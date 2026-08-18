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
)

type InputProtocol string

const (
	ProtocolAnthropicMessages InputProtocol = "anthropic_messages"
	ProtocolOpenAIChat        InputProtocol = "openai_chat"
	ProtocolOpenAIResponses   InputProtocol = "openai_responses"
)

type Turn struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

type TurnNormalizationError struct {
	Code string
	Path string
	// Detail is a bounded, non-sensitive discriminator for the failure, such
	// as the unrecognised block type. It never carries request content, so it
	// is safe to log. Keep it out of metric labels: it is unbounded in
	// cardinality even though it is bounded in sensitivity.
	Detail string
	Err    error
}

func (err *TurnNormalizationError) Error() string {
	if err.Path == "" {
		return fmt.Sprintf("normalize ARC turns: %s: %v", err.Code, err.Err)
	}
	return fmt.Sprintf(
		"normalize ARC turns at %s: %s: %v",
		err.Path,
		err.Code,
		err.Err,
	)
}

func (err *TurnNormalizationError) Unwrap() error {
	return err.Err
}

func TurnNormalizationErrorCode(err error) string {
	var normalizationError *TurnNormalizationError
	if errors.As(err, &normalizationError) {
		return normalizationError.Code
	}
	return ""
}

// TurnNormalizationErrorDetail returns the bounded discriminator for a
// normalization failure, or the empty string when the failure carries none.
func TurnNormalizationErrorDetail(err error) string {
	var normalizationError *TurnNormalizationError
	if errors.As(err, &normalizationError) {
		return normalizationError.Detail
	}
	return ""
}

// TurnNormalizationErrorPath returns the request path that failed to
// normalize, or the empty string when the failure is not path-scoped.
func TurnNormalizationErrorPath(err error) string {
	var normalizationError *TurnNormalizationError
	if errors.As(err, &normalizationError) {
		return normalizationError.Path
	}
	return ""
}

func NormalizeTurns(protocol InputProtocol, requestBody []byte) ([]Turn, error) {
	request, err := decodeRequestObject(requestBody)
	if err != nil {
		return nil, err
	}
	switch protocol {
	case ProtocolAnthropicMessages:
		return normalizeAnthropicTurns(request)
	case ProtocolOpenAIChat:
		return normalizeOpenAIChatTurns(request)
	case ProtocolOpenAIResponses:
		return normalizeOpenAIResponsesTurns(request)
	default:
		return nil, turnError(
			"unsupported_protocol",
			"",
			"unsupported protocol %q",
			protocol,
		)
	}
}

func decodeRequestObject(data []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, turnError("invalid_json", "", "%v", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, turnError(
				"invalid_json",
				"",
				"request contains multiple JSON values",
			)
		}
		return nil, turnError("invalid_json", "", "%v", err)
	}
	request, ok := value.(map[string]any)
	if !ok {
		return nil, turnError(
			"invalid_request",
			"",
			"request body must be an object",
		)
	}
	return request, nil
}

func requiredArray(
	object map[string]any,
	field string,
	path string,
) ([]any, error) {
	value, ok := object[field]
	if !ok {
		return nil, turnError(
			"missing_field",
			path+"."+field,
			"field is required",
		)
	}
	array, ok := value.([]any)
	if !ok {
		return nil, turnError(
			"invalid_field",
			path+"."+field,
			"field must be an array",
		)
	}
	return array, nil
}

func requiredObject(value any, path string) (map[string]any, error) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, turnError(
			"invalid_item",
			path,
			"item must be an object",
		)
	}
	return object, nil
}

func requiredString(
	object map[string]any,
	field string,
	path string,
) (string, error) {
	value, ok := object[field]
	if !ok {
		return "", turnError(
			"missing_field",
			path+"."+field,
			"field is required",
		)
	}
	text, ok := value.(string)
	if !ok {
		return "", turnError(
			"invalid_field",
			path+"."+field,
			"field must be a string",
		)
	}
	return text, nil
}

func optionalBool(
	object map[string]any,
	field string,
	path string,
) (bool, error) {
	value, ok := object[field]
	if !ok || value == nil {
		return false, nil
	}
	result, ok := value.(bool)
	if !ok {
		return false, turnError(
			"invalid_field",
			path+"."+field,
			"field must be a boolean",
		)
	}
	return result, nil
}

func turnError(
	code string,
	path string,
	format string,
	arguments ...any,
) error {
	return &TurnNormalizationError{
		Code: code,
		Path: path,
		Err:  fmt.Errorf(format, arguments...),
	}
}

// turnErrorWithDetail is turnError plus a bounded discriminator that survives
// into logs. Use it wherever the failure names something the caller cannot
// otherwise recover, such as an unrecognised block type.
func turnErrorWithDetail(
	code string,
	path string,
	detail string,
	format string,
	arguments ...any,
) error {
	return &TurnNormalizationError{
		Code:   code,
		Path:   path,
		Detail: detail,
		Err:    fmt.Errorf(format, arguments...),
	}
}
