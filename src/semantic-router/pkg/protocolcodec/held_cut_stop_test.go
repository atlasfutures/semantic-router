package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// anthropicStop is the terminal Anthropic sends for a turn that stopped for
// reason: message_delta with the stop, then message_stop.
func anthropicStop(reason string) string {
	return "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"" + reason + "\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

// A tool_use block stopped before any input_json_delta: its arguments are
// the start's {} placeholder, which reads whole unless the stop marks it cut.
const anthropicPlaceholderCut = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-fixture\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_fixture\",\"name\":\"bash\",\"input\":{}}}\n\n" +
	anthropicFailureBlockStop

// heldCutCompletion runs a held cut call under an Anthropic stop through one
// client format and returns the run, failing on any error or on a cut call
// presented as whole.
func heldCutCompletion(t *testing.T, target llmprotocol.WireFormat, body string) failureCutRun {
	t.Helper()
	run := runFailureCut(t, llmprotocol.AnthropicMessagesV1, target, body, nil)
	if run.pushErr != nil || run.finalErr != nil {
		t.Fatalf("push error %v, finalize error %v", run.pushErr, run.finalErr)
	}
	if bytes.Contains(run.wire, []byte("invalid_stream_tool_arguments")) || bytes.Contains(run.wire, []byte(`"type":"error"`)) {
		t.Fatalf("a held cut call failed the turn:\n%s", run.wire)
	}
	cut := false
	for _, event := range run.events {
		if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil {
			if !event.ToolCall.Incomplete {
				t.Fatalf("the cut call completed as whole: %+v", event.ToolCall)
			}
			cut = true
		}
		if event.Type == llmprotocol.EventResponseFailed {
			t.Fatalf("the turn failed: %+v", event)
		}
	}
	if !cut {
		t.Fatalf("no cut call completed: %+v", run.events)
	}
	return run
}

// model_context_window_exceeded is a length stop like max_tokens: the
// context window filled while the model wrote the call. The call is held as
// cut and the turn ends as each format's length stop.
func TestAToolCallCutAtTheContextWindowIsALengthStop(t *testing.T) {
	for name, held := range map[string]string{
		"truncated arguments":  anthropicFailureCutStart + anthropicFailureCutPartial + anthropicFailureBlockStop,
		"placeholder argument": anthropicPlaceholderCut,
	} {
		body := held + anthropicStop("model_context_window_exceeded")
		t.Run(name+"/anthropic", func(t *testing.T) {
			run := heldCutCompletion(t, llmprotocol.AnthropicMessagesV1, body)
			if !bytes.Contains(run.wire, []byte(`"stop_reason":"model_context_window_exceeded"`)) ||
				!bytes.Contains(run.wire, []byte("event: message_stop")) {
				t.Fatalf("Messages client did not get the stop as sent:\n%s", run.wire)
			}
		})
		t.Run(name+"/responses", func(t *testing.T) {
			run := heldCutCompletion(t, llmprotocol.OpenAIResponsesV1, body)
			requireResponsesIncompleteCut(t, run.wire, "max_output_tokens")
		})
		t.Run(name+"/chat", func(t *testing.T) {
			run := heldCutCompletion(t, llmprotocol.OpenAIChatV1, body)
			if !bytes.Contains(run.wire, []byte(`"finish_reason":"length"`)) || !bytes.Contains(run.wire, []byte("data: [DONE]")) {
				t.Fatalf("Chat client did not get a length stop:\n%s", run.wire)
			}
		})
	}
}

// anthropicRefusalStop is the recorded refusal terminal: Anthropic's
// streaming classifier stops a forced tool call mid-arguments with
// stop_reason refusal and says why in stop_details.
const anthropicRefusalStop = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_sequence\":null,\"stop_details\":{\"type\":\"refusal\",\"category\":\"cyber\",\"explanation\":\"Flagged.\"}},\"usage\":{\"output_tokens\":9}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// A refusal that stops the model mid-call is a refusal, not an argument
// error. The final tool_use was never finished, whatever its arguments: cut
// mid-object, still the start's {} placeholder (the recorded shape: a forced
// tool_use, no input_json_delta, then the refusal), or a whole object. It
// stays incomplete, and each client gets the refusal as it gets one with no
// call cut (Anthropic stop_reason refusal, Chat content_filter, Responses
// incomplete at content_filter), never a completed function_call.
func TestAToolCallCutByARefusalIsARefusal(t *testing.T) {
	requireRefusalStop := func(t *testing.T, run failureCutRun) {
		t.Helper()
		for _, event := range run.events {
			if event.Type == llmprotocol.EventResponseCompleted {
				if event.StopReason != llmprotocol.StopContentFilter {
					t.Fatalf("stop = %q, want content_filter", event.StopReason)
				}
				return
			}
		}
		t.Fatalf("the refused turn never completed: %+v", run.events)
	}
	for name, held := range map[string]string{
		"truncated arguments":   anthropicFailureCutStart + anthropicFailureCutPartial + anthropicFailureBlockStop,
		"placeholder arguments": anthropicPlaceholderCut,
		"whole arguments":       anthropicFailureCutStart + anthropicFailureWhole + anthropicFailureBlockStop,
	} {
		body := held + anthropicRefusalStop
		t.Run(name+"/anthropic", func(t *testing.T) {
			run := heldCutCompletion(t, llmprotocol.AnthropicMessagesV1, body)
			requireRefusalStop(t, run)
			if !bytes.Contains(run.wire, []byte(`"stop_reason":"refusal"`)) || !bytes.Contains(run.wire, []byte("event: message_stop")) {
				t.Fatalf("Messages client did not get the refusal:\n%s", run.wire)
			}
		})
		t.Run(name+"/responses", func(t *testing.T) {
			run := heldCutCompletion(t, llmprotocol.OpenAIResponsesV1, body)
			requireRefusalStop(t, run)
			requireResponsesIncompleteCut(t, run.wire, "content_filter")
			if bytes.Contains(run.wire, []byte(`"status":"completed"`)) {
				t.Fatalf("Responses client was given a completed item:\n%s", run.wire)
			}
		})
		t.Run(name+"/chat", func(t *testing.T) {
			run := heldCutCompletion(t, llmprotocol.OpenAIChatV1, body)
			requireRefusalStop(t, run)
			if !bytes.Contains(run.wire, []byte(`"finish_reason":"content_filter"`)) || !bytes.Contains(run.wire, []byte("data: [DONE]")) {
				t.Fatalf("Chat client did not get the refusal:\n%s", run.wire)
			}
		})
		t.Run(name+"/buffered", func(t *testing.T) {
			response, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1, []byte(body),
				llmprotocol.StreamContext{Context: t.Context(), PublicModel: "public-model", ProviderModel: "claude-fixture"})
			if err != nil {
				t.Fatal(err)
			}
			call := response.Output[len(response.Output)-1].Content[0].ToolCall
			if response.StopReason != llmprotocol.StopContentFilter || call == nil || !call.Incomplete ||
				response.StopDetails == nil || response.StopDetails.Category != "cyber" {
				t.Fatalf("response = stop %q details %+v call %+v", response.StopReason, response.StopDetails, call)
			}
		})
	}
}

// A buffered Messages refusal whose last block is a tool_use is decoded with
// the call incomplete, by the same rule, and a Responses client is never
// given it as completed. A Messages client keeps the input the provider
// sent.
func TestABufferedRefusalMarksTheLastToolCallIncomplete(t *testing.T) {
	for name, input := range map[string]string{"placeholder": `{}`, "whole": `{"command":"ls -la /tmp"}`} {
		t.Run(name, func(t *testing.T) {
			body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_1","name":"bash","input":` + input + `}],` +
				`"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"cyber"},"usage":{"input_tokens":1,"output_tokens":1}}`)
			engine := NewBuiltinEngine()
			response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
			if err != nil {
				t.Fatal(err)
			}
			if call := response.Output[0].Content[0].ToolCall; call == nil || !call.Incomplete {
				t.Fatalf("decoded call = %+v", call)
			}
			responses, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{})
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(responses.Body, []byte(`"status":"completed"`)) || !bytes.Contains(responses.Body, []byte(`"reason":"content_filter"`)) {
				t.Fatalf("Responses body = %s", responses.Body)
			}
			messages, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, llmprotocol.Envelope{})
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(messages.Body, []byte(`"input":`+input)) || !bytes.Contains(messages.Body, []byte(`"stop_reason":"refusal"`)) {
				t.Fatalf("Messages body = %s", messages.Body)
			}
		})
	}
}

// A buffered reply whose last block is a tool_use under the context window
// stop is decoded with the call marked cut, as under max_tokens.
func TestABufferedContextWindowStopMarksTheLastToolCallCut(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}],"stop_reason":"model_context_window_exceeded","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if call := response.Output[0].Content[0].ToolCall; call == nil || !call.Incomplete {
		t.Fatalf("decoded call = %+v", call)
	}
}

// requireResponsesIncompleteCut holds a Responses stream to a cut call: the
// function_call item ends incomplete and the response ends response.incomplete
// for reason, its output keeping the call incomplete.
func requireResponsesIncompleteCut(t *testing.T, wire []byte, reason string) {
	t.Helper()
	var transcript goldenStreamTranscript
	if err := json.Unmarshal(marshalGoldenStreamTranscript(t, [][]byte{wire}), &transcript); err != nil {
		t.Fatal(err)
	}
	var item, terminal map[string]any
	for _, frame := range transcript.Frames {
		data, _ := frame.Data.(map[string]any)
		switch frame.Event {
		case "response.output_item.done":
			item, _ = data["item"].(map[string]any)
		case "response.incomplete", "response.completed", "response.failed":
			terminal, _ = data["response"].(map[string]any)
			terminal["event"] = frame.Event
		}
	}
	if item["type"] != "function_call" || item["status"] != "incomplete" {
		t.Fatalf("function_call item = %v\n%s", item, wire)
	}
	details, _ := terminal["incomplete_details"].(map[string]any)
	if terminal["event"] != "response.incomplete" || terminal["status"] != "incomplete" || details["reason"] != reason {
		t.Fatalf("terminal = %v, want response.incomplete for %s", terminal, reason)
	}
	output, _ := terminal["output"].([]any)
	if len(output) == 0 || output[len(output)-1].(map[string]any)["status"] != "incomplete" {
		t.Fatalf("terminal output = %v", output)
	}
}
