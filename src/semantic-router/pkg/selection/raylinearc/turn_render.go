/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

func renderToolCall(name string, arguments any) string {
	return fmt.Sprintf("[tool_call %s] %s", name, compactJSON(arguments))
}

func renderToolResult(name string, text string, isError bool) string {
	marker := ""
	if isError {
		marker = " [tool error]"
	}
	return fmt.Sprintf("[tool_result %s]%s\n%s", name, marker, text)
}

// compactJSON ports Rayline's compact_json formatting: keys are sorted,
// comma/colon separators carry one space, and every non-ASCII rune is escaped.
func compactJSON(value any) string {
	switch typed := value.(type) {
	case nil:
		return "null"
	case bool:
		return strconv.FormatBool(typed)
	case json.Number:
		return typed.String()
	case string:
		return asciiJSONString(typed)
	case []any:
		rendered := make([]string, len(typed))
		for index := range typed {
			rendered[index] = compactJSON(typed[index])
		}
		return "[" + strings.Join(rendered, ", ") + "]"
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		rendered := make([]string, len(keys))
		for index, key := range keys {
			rendered[index] = asciiJSONString(key) +
				": " + compactJSON(typed[key])
		}
		return "{" + strings.Join(rendered, ", ") + "}"
	default:
		return ""
	}
}

// stringCoerce ports Rayline's non-object Anthropic tool-input coercion.
func stringCoerce(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case bool:
		if typed {
			return "True"
		}
		return "False"
	case string:
		return typed
	default:
		var buffer bytes.Buffer
		encoder := json.NewEncoder(&buffer)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(value); err != nil {
			return ""
		}
		return strings.TrimSuffix(buffer.String(), "\n")
	}
}

func asciiJSONString(value string) string {
	var result strings.Builder
	result.WriteByte('"')
	for _, character := range value {
		switch character {
		case '"', '\\':
			result.WriteByte('\\')
			result.WriteRune(character)
		case '\b':
			result.WriteString(`\b`)
		case '\f':
			result.WriteString(`\f`)
		case '\n':
			result.WriteString(`\n`)
		case '\r':
			result.WriteString(`\r`)
		case '\t':
			result.WriteString(`\t`)
		default:
			writeJSONRune(&result, character)
		}
	}
	result.WriteByte('"')
	return result.String()
}

func writeJSONRune(result *strings.Builder, character rune) {
	if character >= 0x20 && character <= 0x7f {
		result.WriteRune(character)
		return
	}
	if character <= 0xffff {
		fmt.Fprintf(result, `\u%04x`, character)
		return
	}
	high, low := utf16.EncodeRune(character)
	fmt.Fprintf(result, `\u%04x\u%04x`, high, low)
}

func decodeArgumentJSON(arguments string, path string) (any, error) {
	decoder := json.NewDecoder(strings.NewReader(arguments))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, turnError("malformed_tool_arguments", path, "%v", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err != nil {
			return nil, turnError(
				"malformed_tool_arguments",
				path,
				"%v",
				err,
			)
		}
		return nil, turnError(
			"malformed_tool_arguments",
			path,
			"arguments contain multiple JSON values",
		)
	}
	return value, nil
}

func resolveToolName(
	toolNames map[string]string,
	callID string,
	path string,
) (string, error) {
	name, ok := toolNames[callID]
	if !ok || name == "" {
		return "", turnError(
			"unresolved_tool_id",
			path,
			"tool call %q has no recorded name",
			callID,
		)
	}
	return name, nil
}

func recordToolName(
	toolNames map[string]string,
	callID string,
	name string,
	path string,
) error {
	if callID == "" || name == "" {
		return turnError(
			"invalid_field",
			path,
			"tool call id and name must be nonempty",
		)
	}
	if _, duplicate := toolNames[callID]; duplicate {
		return turnError(
			"duplicate_tool_id",
			path,
			"tool call id %q is duplicated",
			callID,
		)
	}
	toolNames[callID] = name
	return nil
}
