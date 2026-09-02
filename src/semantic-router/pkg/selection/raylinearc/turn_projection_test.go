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
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// This file is the oracle for the turn projection.
//
// The fixture holds the fork's protocol-turn goldens verbatim. Each case is a
// raw wire body and the turn list the fork's own parsers produced from it. The
// router no longer parses wire bodies: it decodes once through the public
// codec and the selector reads the neutral request. So each golden input is
// replayed through that codec and the projection is asserted against the
// fork's recorded output.
//
// Three dispositions cover every case, and the table below is the evidence for
// which one each case has.
//
//   - projected: the codec accepts the body and the projection reproduces the
//     fork's turns byte for byte. This is the calibration-preserving set.
//   - codecRejected: the codec refuses the body, so the request answers 400 at
//     ingress and never reaches the selector. The fork's turns are
//     unreachable, not wrong.
//   - codecReshaped: the codec accepts the body but changes the message
//     sequence, so the fork's turns are not reproducible. The turns the
//     projection must now produce are written out beside the reason.
//
// A case may only move between dispositions with a recorded reason. Silent
// movement is what would void the encoder calibration.
type goldenDisposition int

const (
	dispositionProjected goldenDisposition = iota
	dispositionCodecRejected
	dispositionCodecReshaped
)

type goldenExpectation struct {
	disposition goldenDisposition
	// codecError is the protocol error code the codec answers for a rejected
	// case. It is asserted so that a codec change shows up here rather than
	// silently turning a 400 back into a routed request.
	codecError string
	// turns overrides the fixture's turns for a reshaped case.
	turns []Turn
}

// turnProjectionExpectations classifies every case in the fixture.
//
// Counted on 2026-09-02 against the codec at this commit: 14 cases project, 8
// reshape, 13 are rejected at ingress.
//
// Nine cases moved off rejected when the codec began carrying request blocks
// it does not name. Five of them project byte for byte. Four reshape, and the
// reason is recorded beside each: two carry a block the fork read text out of
// (compaction, mid_conv_system), so the projection now emits the turn without
// that text; two carry a block that is the message's only content, so the turn
// survives as empty text where the fork emitted no turn at all. Both are the
// consequence of the carrier being opaque: nothing outside the codec that
// re-emits it may read inside it, including this projection.
var turnProjectionExpectations = map[string]goldenExpectation{
	// Projected: byte-identical to the fork.
	"openai_chat_tool_flow":                          {disposition: dispositionProjected},
	"responses_string_input":                         {disposition: dispositionProjected},
	"responses_system_and_developer_roles_drop":      {disposition: dispositionProjected},
	"responses_empty_input_is_no_turns":              {disposition: dispositionProjected},
	"anthropic_consecutive_same_role_stay_separate":  {disposition: dispositionProjected},
	"anthropic_tool_result_string_content":           {disposition: dispositionProjected},
	"anthropic_tool_result_empty_content_with_error": {disposition: dispositionProjected},
	"responses_message_content_parts_drop_non_text":  {disposition: dispositionProjected},

	// Reshaped: the codec drops a message whose content renders to nothing, so
	// the empty turn the fork emitted no longer exists to project.
	"anthropic_null_content_is_empty_turn": {
		disposition: dispositionCodecReshaped,
		turns:       []Turn{{Role: "assistant", Text: "hi"}},
	},
	"anthropic_absent_content_is_empty_turn": {
		disposition: dispositionCodecReshaped,
		turns:       []Turn{{Role: "assistant", Text: "hi"}},
	},
	"anthropic_empty_content_array_is_empty_turn": {
		disposition: dispositionCodecReshaped,
		turns:       []Turn{},
	},
	// Reshaped: the codec splits one Anthropic user message that mixes text, a
	// tool result and an image into three neutral messages. The image-only
	// message survives as a turn whose blocks all drop, so the projection
	// emits one more turn than the fork did.
	"anthropic_text_and_tool_result_in_one_message": {
		disposition: dispositionCodecReshaped,
		turns: []Turn{
			{Role: "assistant", Text: "[tool_call grep] {}"},
			{Role: "user", Text: "before\n\n[tool_result grep]\nout"},
			{Role: "user", Text: ""},
		},
	},

	// Rejected: the fork also refused these, for its own reasons.
	"anthropic_unresolved_tool_id":  {disposition: dispositionCodecRejected, codecError: "orphan_tool_result"},
	"chat_unresolved_tool_id":       {disposition: dispositionCodecRejected, codecError: "orphan_tool_result"},
	"responses_unresolved_tool_id":  {disposition: dispositionCodecRejected, codecError: "orphan_tool_result"},
	"chat_malformed_tool_arguments": {disposition: dispositionCodecRejected, codecError: "invalid_tool_call"},
	"anthropic_unknown_block": {
		disposition: dispositionCodecReshaped,
		turns:       []Turn{{Role: "user", Text: ""}},
	},
	"chat_unknown_role":      {disposition: dispositionCodecRejected, codecError: "invalid_role"},
	"responses_unknown_item": {disposition: dispositionCodecRejected, codecError: "unsupported_input_item"},
	"anthropic_tool_reference_rejected_at_top_level": {
		disposition: dispositionCodecReshaped,
		turns:       []Turn{{Role: "user", Text: ""}},
	},

	// Rejected: shapes the fork routed. The codec models a narrower request
	// union than the Anthropic and Responses request APIs accept, so every one
	// of these answers 400 at ingress today.
	"anthropic_tool_flow":              {disposition: dispositionProjected},
	"openai_responses_tool_flow":       {disposition: dispositionCodecRejected, codecError: "invalid_json"},
	"anthropic_python_scalar_coercion": {disposition: dispositionCodecRejected, codecError: "invalid_tool_call"},
	"anthropic_compaction_summary": {
		disposition: dispositionCodecReshaped,
		turns:       []Turn{{Role: "user", Text: "continue"}},
	},
	"anthropic_compaction_failed_is_empty":         {disposition: dispositionProjected},
	"anthropic_server_tool_families_drop_as_pairs": {disposition: dispositionProjected},
	"anthropic_tool_reference_in_result":           {disposition: dispositionProjected},
	"anthropic_mid_conv_system_folds": {
		disposition: dispositionCodecReshaped,
		turns:       []Turn{{Role: "user", Text: "summarize"}},
	},
	"anthropic_advisor_and_fallback_drop":               {disposition: dispositionProjected},
	"anthropic_system_role_message_drops_whole":         {disposition: dispositionProjected},
	"responses_host_tool_items_drop":                    {disposition: dispositionCodecRejected, codecError: "unsupported_input_item"},
	"anthropic_unknown_role_drops_whole":                {disposition: dispositionCodecRejected, codecError: "invalid_anthropic_role"},
	"anthropic_all_blocks_dropped_is_empty_turn":        {disposition: dispositionCodecRejected, codecError: "redacted_reasoning"},
	"anthropic_interleaved_rich_blocks_keep_order":      {disposition: dispositionCodecRejected, codecError: "unsupported_document_source"},
	"responses_dropped_item_keeps_same_role_coalescing": {disposition: dispositionCodecRejected, codecError: "empty_message"},
}

type turnProjectionFixture struct {
	SchemaVersion string                     `json:"schema_version"`
	Cases         []turnProjectionGoldenCase `json:"cases"`
}

type turnProjectionGoldenCase struct {
	ID       string          `json:"id"`
	Protocol string          `json:"protocol"`
	Request  json.RawMessage `json:"request"`
	Turns    []Turn          `json:"turns"`
}

func TestTurnProjectionGoldens(t *testing.T) {
	fixture := readTurnProjectionFixture(t)
	engine, err := protocolcodec.NewEngine(
		protocolcodec.NewBuiltinRegistry(),
		llmprotocol.DefaultPolicy(),
	)
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}
	counts := map[goldenDisposition]int{}
	for _, test := range fixture.Cases {
		expectation, known := turnProjectionExpectations[test.ID]
		if !known {
			t.Errorf("case %q has no recorded disposition", test.ID)
			continue
		}
		counts[expectation.disposition]++
		t.Run(test.ID, func(t *testing.T) {
			runTurnProjectionCase(t, engine, test, expectation)
		})
	}
	if counts[dispositionProjected] != 14 ||
		counts[dispositionCodecReshaped] != 8 ||
		counts[dispositionCodecRejected] != 13 {
		t.Fatalf(
			"disposition counts moved: projected=%d reshaped=%d rejected=%d",
			counts[dispositionProjected],
			counts[dispositionCodecReshaped],
			counts[dispositionCodecRejected],
		)
	}
}

func runTurnProjectionCase(
	t *testing.T,
	engine *protocolcodec.Engine,
	test turnProjectionGoldenCase,
	expectation goldenExpectation,
) {
	t.Helper()
	format, body := turnProjectionWireBody(t, test)
	request, _, _, decodeErr := engine.DecodeRequest(format, body)
	if expectation.disposition == dispositionCodecRejected {
		if decodeErr == nil {
			t.Fatalf("codec accepted a body recorded as rejected")
		}
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(decodeErr, &protocolError) {
			t.Fatalf("decode error = %T, want *llmprotocol.ProtocolError", decodeErr)
		}
		if protocolError.Code != expectation.codecError {
			t.Fatalf(
				"codec error = %q, want %q",
				protocolError.Code,
				expectation.codecError,
			)
		}
		return
	}
	if decodeErr != nil {
		t.Fatalf("codec rejected a body recorded as routable: %v", decodeErr)
	}
	want := test.Turns
	if expectation.disposition == dispositionCodecReshaped {
		want = expectation.turns
	}
	got, err := ProjectTurns(&request, TurnOptions{})
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	assertTurnsIdentical(t, got, want)
}

// turnProjectionWireBody completes each fixture case into a request the codec
// will look at. The fixture records only the fields the fork's parsers read;
// the codec additionally requires the routing fields every real client sends.
func turnProjectionWireBody(
	t *testing.T,
	test turnProjectionGoldenCase,
) (llmprotocol.WireFormat, []byte) {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(test.Request, &body); err != nil {
		t.Fatalf("fixture request is not an object: %v", err)
	}
	body["model"] = "golden-model"
	var format llmprotocol.WireFormat
	switch test.Protocol {
	case "anthropic_messages":
		format = llmprotocol.AnthropicMessagesV1
		body["max_tokens"] = 1
	case "openai_chat":
		format = llmprotocol.OpenAIChatV1
	case "openai_responses":
		format = llmprotocol.OpenAIResponsesV1
	default:
		t.Fatalf("fixture case names an unknown protocol %q", test.Protocol)
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("re-encode fixture request: %v", err)
	}
	return format, encoded
}

// assertTurnsIdentical compares the serialised turn lists rather than the Go
// values. The encoder is fed the serialisation, so that is the contract worth
// pinning.
func assertTurnsIdentical(t *testing.T, got []Turn, want []Turn) {
	t.Helper()
	gotJSON, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal projected turns: %v", err)
	}
	wantJSON, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal golden turns: %v", err)
	}
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("turns\n got: %s\nwant: %s", gotJSON, wantJSON)
	}
}

func readTurnProjectionFixture(t *testing.T) turnProjectionFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/turn_projection_goldens.v1.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture turnProjectionFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if fixture.SchemaVersion != "rayline.arc.protocol-turns.v1" {
		t.Fatalf("fixture schema = %q", fixture.SchemaVersion)
	}
	if len(fixture.Cases) != 35 {
		t.Fatalf("fixture has %d cases, want 35", len(fixture.Cases))
	}
	return fixture
}
