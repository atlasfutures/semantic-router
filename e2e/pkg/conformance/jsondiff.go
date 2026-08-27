package conformance

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// decodeJSON parses a body into generic values. Numbers stay as json.Number so a
// comparison never loses integer precision and a mismatch reports the literal that
// was actually on the wire.
func decodeJSON(raw []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()

	var value any
	if err := dec.Decode(&value); err != nil {
		return nil, fmt.Errorf("decode json: %w", err)
	}
	if dec.More() {
		return nil, fmt.Errorf("decode json: trailing content after the top-level value")
	}
	return value, nil
}

// parsePointer splits an RFC 6901 JSON Pointer into its unescaped tokens.
func parsePointer(pointer string) ([]string, error) {
	if pointer == "" {
		return nil, nil
	}
	if !strings.HasPrefix(pointer, pointerPrefix) {
		return nil, fmt.Errorf("json pointer %q must start with %q", pointer, pointerPrefix)
	}

	tokens := strings.Split(pointer[1:], "/")
	for i, token := range tokens {
		tokens[i] = strings.ReplaceAll(strings.ReplaceAll(token, "~1", "/"), "~0", "~")
	}
	return tokens, nil
}

// deletePointer removes the value at pointer, if present. A pointer that does not
// resolve is not an error: exact-except exclusions are declared once per case and
// legitimately apply to only one of the two sides.
func deletePointer(root any, pointer string) error {
	tokens, err := parsePointer(pointer)
	if err != nil {
		return err
	}
	if len(tokens) == 0 {
		return fmt.Errorf("json pointer %q cannot address the whole document", pointer)
	}

	parent, ok := resolve(root, tokens[:len(tokens)-1])
	if !ok {
		return nil
	}
	last := tokens[len(tokens)-1]

	switch container := parent.(type) {
	case map[string]any:
		delete(container, last)
	case []any:
		// Deleting from an array would renumber every later index and silently change
		// what the remaining exclusions address.
		return fmt.Errorf("json pointer %q addresses an array element; exclude the array or a field inside the element", pointer)
	}
	return nil
}

func resolve(root any, tokens []string) (any, bool) {
	current := root
	for _, token := range tokens {
		switch container := current.(type) {
		case map[string]any:
			next, ok := container[token]
			if !ok {
				return nil, false
			}
			current = next
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil || index < 0 || index >= len(container) {
				return nil, false
			}
			current = container[index]
		default:
			return nil, false
		}
	}
	return current, true
}

// kindOf names the JSON type of a decoded value, for typed volatile matching.
func kindOf(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case json.Number:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return fmt.Sprintf("%T", value)
}

// diffJSON walks want and got in parallel and appends one Mismatch per difference.
//
// A pointer listed in volatile must carry the same JSON type on both sides but may
// carry a different value: nondeterministic IDs and timestamps are matched by type
// rather than deleted, so a field that disappears entirely is still a failure.
func diffJSON(path string, want, got any, volatile map[string]struct{}, out *[]Mismatch) {
	if _, isVolatile := volatile[pointerOrRoot(path)]; isVolatile {
		if wantKind, gotKind := kindOf(want), kindOf(got); wantKind != gotKind {
			*out = append(*out, Mismatch{
				Path:   pointerOrRoot(path),
				Want:   wantKind,
				Got:    gotKind,
				Reason: "volatile field changed JSON type",
			})
		}
		return
	}

	wantKind, gotKind := kindOf(want), kindOf(got)
	if wantKind != gotKind {
		*out = append(*out, Mismatch{
			Path:   pointerOrRoot(path),
			Want:   render(want),
			Got:    render(got),
			Reason: fmt.Sprintf("type %s vs %s", wantKind, gotKind),
		})
		return
	}

	switch wantValue := want.(type) {
	case map[string]any:
		diffObject(path, wantValue, got.(map[string]any), volatile, out)
	case []any:
		diffArray(path, wantValue, got.([]any), volatile, out)
	default:
		if render(want) != render(got) {
			*out = append(*out, Mismatch{Path: pointerOrRoot(path), Want: render(want), Got: render(got), Reason: "value differs"})
		}
	}
}

func diffObject(path string, want, got map[string]any, volatile map[string]struct{}, out *[]Mismatch) {
	for _, key := range sortedKeys(want, got) {
		wantChild, inWant := want[key]
		gotChild, inGot := got[key]
		child := path + pointerPrefix + escapeToken(key)

		switch {
		case !inGot:
			*out = append(*out, Mismatch{Path: child, Want: render(wantChild), Got: "<absent>", Reason: "field missing"})
		case !inWant:
			*out = append(*out, Mismatch{Path: child, Want: "<absent>", Got: render(gotChild), Reason: "unexpected field"})
		default:
			diffJSON(child, wantChild, gotChild, volatile, out)
		}
	}
}

func diffArray(path string, want, got []any, volatile map[string]struct{}, out *[]Mismatch) {
	// A length change is reported once on the array: per-index noise would bury it.
	if len(want) != len(got) {
		*out = append(*out, Mismatch{
			Path:   pointerOrRoot(path),
			Want:   strconv.Itoa(len(want)) + " elements",
			Got:    strconv.Itoa(len(got)) + " elements",
			Reason: "array length differs",
		})
		return
	}
	for i := range want {
		diffJSON(path+pointerPrefix+strconv.Itoa(i), want[i], got[i], volatile, out)
	}
}

func sortedKeys(maps ...map[string]any) []string {
	seen := map[string]struct{}{}
	for _, m := range maps {
		for key := range m {
			seen[key] = struct{}{}
		}
	}

	keys := make([]string, 0, len(seen))
	for key := range seen {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func escapeToken(token string) string {
	return strings.ReplaceAll(strings.ReplaceAll(token, "~", "~0"), "/", "~1")
}

// pointerOrRoot keeps the empty path printable as the document root.
func pointerOrRoot(path string) string {
	if path == "" {
		return pointerPrefix
	}
	return path
}

func render(value any) string {
	switch typed := value.(type) {
	case nil:
		return "null"
	case string:
		return strconv.Quote(typed)
	case json.Number:
		return typed.String()
	case bool:
		return strconv.FormatBool(typed)
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	return string(encoded)
}
