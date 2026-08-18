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
	"os"
	"reflect"
	"testing"
)

const protocolGoldenSchema = "rayline.arc.protocol-turns.v1"

type protocolGoldenFixture struct {
	SchemaVersion string               `json:"schema_version"`
	Cases         []protocolGoldenCase `json:"cases"`
}

type protocolGoldenCase struct {
	ID        string          `json:"id"`
	Protocol  InputProtocol   `json:"protocol"`
	Request   json.RawMessage `json:"request"`
	Turns     []Turn          `json:"turns,omitempty"`
	ErrorCode string          `json:"error_code,omitempty"`
}

func TestProtocolTurnGoldens(t *testing.T) {
	data, err := os.ReadFile("testdata/protocol_turn_goldens.v1.json")
	if err != nil {
		t.Fatalf("read protocol fixture: %v", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var fixture protocolGoldenFixture
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode protocol fixture: %v", err)
	}
	if fixture.SchemaVersion != protocolGoldenSchema {
		t.Fatalf(
			"fixture schema = %q, want %q",
			fixture.SchemaVersion,
			protocolGoldenSchema,
		)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("protocol fixture has no cases")
	}

	seen := make(map[string]bool, len(fixture.Cases))
	for _, test := range fixture.Cases {
		t.Run(test.ID, func(t *testing.T) {
			assertProtocolGoldenCase(t, test, seen)
		})
	}
}

func assertProtocolGoldenCase(
	t *testing.T,
	test protocolGoldenCase,
	seen map[string]bool,
) {
	t.Helper()
	if test.ID == "" || seen[test.ID] {
		t.Fatalf("fixture case ID is empty or duplicated: %q", test.ID)
	}
	seen[test.ID] = true
	turns, normalizeErr := NormalizeTurns(test.Protocol, test.Request, TurnOptions{})
	if test.ErrorCode != "" {
		if got := TurnNormalizationErrorCode(normalizeErr); got != test.ErrorCode {
			t.Fatalf(
				"error code = %q (%v), want %q",
				got,
				normalizeErr,
				test.ErrorCode,
			)
		}
		return
	}
	if normalizeErr != nil {
		t.Fatalf("NormalizeTurns() error = %v", normalizeErr)
	}
	if !reflect.DeepEqual(turns, test.Turns) {
		t.Fatalf("turns = %#v, want %#v", turns, test.Turns)
	}
}

func TestCrossProtocolToolGoldenCasesAreIdentical(t *testing.T) {
	requests := map[InputProtocol]string{
		ProtocolAnthropicMessages: `{
			"messages": [
				{"role":"user","content":"task"},
				{"role":"assistant","content":[
					{"type":"text","text":"calling"},
					{"type":"tool_use","id":"call_1","name":"run","input":{"z":"é","a":1}}
				]},
				{"role":"user","content":[
					{"type":"text","text":"context"},
					{"type":"tool_result","tool_use_id":"call_1","content":"done"}
				]}
			]
		}`,
		ProtocolOpenAIChat: `{
			"messages": [
				{"role":"user","content":"task"},
				{"role":"assistant","content":"calling","tool_calls":[{
					"id":"call_1","type":"function",
					"function":{"name":"run","arguments":"{\"z\":\"é\",\"a\":1}"}
				}]},
				{"role":"user","content":"context"},
				{"role":"tool","tool_call_id":"call_1","content":"done"}
			]
		}`,
		ProtocolOpenAIResponses: `{
			"input": [
				{"type":"message","role":"user","content":"task"},
				{"type":"message","role":"assistant","content":"calling"},
				{"type":"function_call","call_id":"call_1","name":"run",
				 "arguments":"{\"z\":\"é\",\"a\":1}"},
				{"type":"message","role":"user","content":"context"},
				{"type":"function_call_output","call_id":"call_1","output":"done"}
			]
		}`,
	}

	var expected []Turn
	for _, protocol := range []InputProtocol{
		ProtocolAnthropicMessages,
		ProtocolOpenAIChat,
		ProtocolOpenAIResponses,
	} {
		turns, err := NormalizeTurns(protocol, []byte(requests[protocol]), TurnOptions{})
		if err != nil {
			t.Fatalf("%s normalization failed: %v", protocol, err)
		}
		if expected == nil {
			expected = turns
			continue
		}
		if !reflect.DeepEqual(turns, expected) {
			t.Fatalf("%s turns = %#v, want %#v", protocol, turns, expected)
		}
	}
}

func TestRaylineRenderingTraps(t *testing.T) {
	value := map[string]any{
		"z": "é😀<>&",
		"a": []any{json.Number("1"), true, nil},
	}
	if got, want := compactJSON(value),
		`{"a": [1, true, null], "z": "\u00e9\ud83d\ude00<>&"}`; got != want {
		t.Fatalf("compactJSON() = %q, want %q", got, want)
	}
	if got := stringCoerce(true); got != "True" {
		t.Fatalf("stringCoerce(true) = %q, want True", got)
	}
	if got := stringCoerce(false); got != "False" {
		t.Fatalf("stringCoerce(false) = %q, want False", got)
	}
	if got := stringCoerce(nil); got != "" {
		t.Fatalf("stringCoerce(nil) = %q, want empty", got)
	}
}

func TestNormalizeTurnsRejectsUnsupportedProtocolAndTrailingJSON(t *testing.T) {
	if code := TurnNormalizationErrorCode(
		mustNormalizeError(InputProtocol("unknown"), `{}`),
	); code != "unsupported_protocol" {
		t.Fatalf("unsupported protocol error code = %q", code)
	}
	if code := TurnNormalizationErrorCode(
		mustNormalizeError(ProtocolOpenAIChat, `{"messages":[]} {}`),
	); code != "invalid_json" {
		t.Fatalf("trailing JSON error code = %q", code)
	}
}

func mustNormalizeError(protocol InputProtocol, body string) error {
	_, err := NormalizeTurns(protocol, []byte(body), TurnOptions{})
	return err
}
