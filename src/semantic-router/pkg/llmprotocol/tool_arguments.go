package llmprotocol

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"unicode/utf8"
)

// NormalizeToolArguments returns a model's whole tool-call arguments as one
// strict JSON object (ValidateJSONObject), or an error when they are not an
// object at all.
//
// Arguments that are strict already come back as they are. Two departures
// from strict JSON are what Anthropic itself accepts of the arguments its
// models write, and are settled the way its decoder (and JSON.parse,
// encoding/json and Python's json) settles them: a repeated member keeps its
// last value, in the place the name first appeared, and an unpaired surrogate
// escape becomes U+FFFD. Such arguments are re-encoded compactly with those
// two decisions applied; nothing else is relaxed. Invalid UTF-8, raw control
// characters in a string, a value that is not an object, trailing data and
// nesting past maximumDepth are still refused.
func NormalizeToolArguments(arguments []byte, maximumDepth int) ([]byte, error) {
	strictErr := ValidateJSONObject(arguments, maximumDepth)
	if strictErr == nil {
		return arguments, nil
	}
	if !utf8.Valid(arguments) {
		return nil, strictErr
	}
	decoder := json.NewDecoder(bytes.NewReader(arguments))
	decoder.UseNumber()
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '{' {
		return nil, fmt.Errorf("JSON value must be an object")
	}
	normalized, err := normalizeJSONObject(decoder, 0, maximumDepth)
	if err != nil {
		return nil, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("JSON contains an additional document")
		}
		return nil, err
	}
	// The re-encoding holds no surrogate escape and no repeated member, so
	// it is strict; this is the check that it stayed so.
	if err := ValidateJSONObject(normalized, maximumDepth); err != nil {
		return nil, err
	}
	return normalized, nil
}

// normalizeJSONValue re-encodes the value whose first token is next, with
// depth counted as consumeJSONValue counts it.
func normalizeJSONValue(decoder *json.Decoder, depth, maximumDepth int) ([]byte, error) {
	if maximumDepth <= 0 || depth > maximumDepth {
		return nil, fmt.Errorf("JSON nesting exceeds the configured limit")
	}
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	switch value := token.(type) {
	case json.Delim:
		if value == '{' {
			return normalizeJSONObject(decoder, depth, maximumDepth)
		}
		if value == '[' {
			return normalizeJSONArray(decoder, depth, maximumDepth)
		}
		return nil, fmt.Errorf("unexpected JSON delimiter %q", value)
	case string:
		return encodeJSONString(value)
	case json.Number:
		return []byte(value), nil
	case bool:
		if value {
			return []byte("true"), nil
		}
		return []byte("false"), nil
	case nil:
		return []byte("null"), nil
	default:
		return nil, fmt.Errorf("unexpected JSON token %T", token)
	}
}

func normalizeJSONObject(decoder *json.Decoder, depth, maximumDepth int) ([]byte, error) {
	var keys []string
	values := make(map[string][]byte)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		key, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("JSON object key is not a string")
		}
		value, err := normalizeJSONValue(decoder, depth+1, maximumDepth)
		if err != nil {
			return nil, err
		}
		if _, repeated := values[key]; !repeated {
			keys = append(keys, key)
		}
		values[key] = value
	}
	if closing, err := decoder.Token(); err != nil || closing != json.Delim('}') {
		return nil, fmt.Errorf("unterminated JSON object")
	}
	out := []byte{'{'}
	for index, key := range keys {
		if index > 0 {
			out = append(out, ',')
		}
		encodedKey, err := encodeJSONString(key)
		if err != nil {
			return nil, err
		}
		out = append(append(append(out, encodedKey...), ':'), values[key]...)
	}
	return append(out, '}'), nil
}

func normalizeJSONArray(decoder *json.Decoder, depth, maximumDepth int) ([]byte, error) {
	out := []byte{'['}
	for first := true; decoder.More(); first = false {
		value, err := normalizeJSONValue(decoder, depth+1, maximumDepth)
		if err != nil {
			return nil, err
		}
		if !first {
			out = append(out, ',')
		}
		out = append(out, value...)
	}
	if closing, err := decoder.Token(); err != nil || closing != json.Delim(']') {
		return nil, fmt.Errorf("unterminated JSON array")
	}
	return append(out, ']'), nil
}

// encodeJSONString encodes a decoded string without HTML escaping, so the
// re-encoding changes no character the model wrote beyond the two
// normalizations.
func encodeJSONString(value string) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buffer.Bytes(), []byte("\n")), nil
}
