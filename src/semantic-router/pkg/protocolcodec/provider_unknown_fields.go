package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sort"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// This file holds the response-side twin of the request carrier in
// unmodeled.go. A client request is the client's document and the Router
// refuses one it cannot read. An upstream response is neither the client's nor
// the Router's: it is a completion that was already generated and paid for, and
// the Router only routes it. Refusing it because the provider named a member
// the wire contract does not is the worse outcome, so the member is removed
// before decoding and its path is reported.
//
// Nothing here widens the client boundary. Only decodeProviderJSON consults it,
// and only under the policy the engine derives for the response leg.

// pruneUnknownProviderFields removes the members of an upstream JSON document
// that the wire struct does not name, and returns their sorted paths. Array
// elements collapse to one path, so a hundred choices report one field once.
// A document with nothing unnamed is returned unchanged, byte for byte.
func pruneUnknownProviderFields(body []byte, targetType reflect.Type) ([]byte, []string) {
	dropped := make(map[string]struct{})
	pruned, changed := pruneJSONValue(body, dereferenceJSONType(targetType), "", dropped)
	if !changed {
		return body, nil
	}
	paths := make([]string, 0, len(dropped))
	for path := range dropped {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return pruned, paths
}

// pruneJSONValue mirrors validateExactJSONValue. The shapes that validation
// leaves open -- a null, a custom unmarshaler, a dynamic map, a RawMessage
// element -- are left exactly as the provider sent them, because a later decode
// of that value applies the same policy again.
func pruneJSONValue(
	body []byte,
	targetType reflect.Type,
	path string,
	dropped map[string]struct{},
) ([]byte, bool) {
	if targetType == nil || bytes.Equal(bytes.TrimSpace(body), []byte("null")) ||
		reflect.PointerTo(targetType).Implements(jsonUnmarshalerType) {
		return body, false
	}
	switch targetType.Kind() {
	case reflect.Struct:
		return pruneJSONObject(body, targetType, path, dropped)
	case reflect.Slice, reflect.Array:
		return pruneJSONArray(body, targetType, path, dropped)
	}
	return body, false
}

func pruneJSONObject(
	body []byte,
	targetType reflect.Type,
	path string,
	dropped map[string]struct{},
) ([]byte, bool) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		// The strict decoder that follows reports the real parse error.
		return body, false
	}
	fields := exactJSONStructFields(targetType)
	changed := false
	for name, value := range object {
		fieldType, named := fields[name]
		if !named {
			dropped[joinFieldPath(path, name)] = struct{}{}
			delete(object, name)
			changed = true
			continue
		}
		pruned, valueChanged := pruneJSONValue(
			value, dereferenceJSONType(fieldType), joinFieldPath(path, name), dropped,
		)
		if valueChanged {
			object[name] = pruned
			changed = true
		}
	}
	if !changed {
		return body, false
	}
	encoded, err := json.Marshal(object)
	if err != nil {
		return body, false
	}
	return encoded, true
}

func pruneJSONArray(
	body []byte,
	targetType reflect.Type,
	path string,
	dropped map[string]struct{},
) ([]byte, bool) {
	if targetType.Elem() == reflect.TypeOf(json.RawMessage{}) {
		return body, false
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(body, &elements); err != nil {
		return body, false
	}
	elementType := dereferenceJSONType(targetType.Elem())
	changed := false
	for index, element := range elements {
		pruned, elementChanged := pruneJSONValue(element, elementType, path+"[]", dropped)
		if elementChanged {
			elements[index] = pruned
			changed = true
		}
	}
	if !changed {
		return body, false
	}
	encoded, err := json.Marshal(elements)
	if err != nil {
		return body, false
	}
	return encoded, true
}

func joinFieldPath(path, name string) string {
	if path == "" {
		return name
	}
	return path + "." + name
}

// reportedProviderFields keeps one line per distinct field path per process. A
// busy router must say which provider member it stopped carrying without
// writing that line on every response.
var reportedProviderFields sync.Map

// reportDroppedProviderFields names the paths and nothing else. It never sees
// a value, so no response content can reach a log through it.
func reportDroppedProviderFields(paths []string) {
	for _, path := range paths {
		if _, seen := reportedProviderFields.LoadOrStore(path, struct{}{}); seen {
			continue
		}
		logging.ComponentWarnEvent("protocolcodec", "upstream_response_field_dropped", map[string]interface{}{
			"field": path,
		})
	}
}
