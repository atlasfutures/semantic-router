package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A provider that fails while a tool call's arguments are still streaming
// leaves the call's arguments a truncated object. Anthropic's own shape for
// it: the block's input_json_delta partials, then an error event. The
// failure is what the client is told; the unfinished arguments are only its
// symptom, and are never reported as invalid_stream_tool_arguments.
const (
	anthropicFailureCutStart = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-fixture\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_fixture\",\"name\":\"bash\",\"input\":{}}}\n\n" +
		"event: ping\ndata: {\"type\": \"ping\"}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\": \"}}\n\n"
	anthropicFailureCutPartial = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls -la /tm\"}}\n\n"
	anthropicFailureWhole      = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls -la /tmp\\\"}\"}}\n\n"
	anthropicFailureBlockStop  = "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	anthropicOverloaded        = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	anthropicEndTurn           = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	chatFailureCutStart = "data: {\"id\":\"gen-fixture\",\"object\":\"chat.completion.chunk\",\"created\":1791137714,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_fixture\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"gen-fixture\",\"object\":\"chat.completion.chunk\",\"created\":1791137714,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"command\\\": \"}}]},\"finish_reason\":null}]}\n\n"
	chatFailureCutPartial = "data: {\"id\":\"gen-fixture\",\"object\":\"chat.completion.chunk\",\"created\":1791137714,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls -la /tm\"}}]},\"finish_reason\":null}]}\n\n"
	chatFailureWhole      = "data: {\"id\":\"gen-fixture\",\"object\":\"chat.completion.chunk\",\"created\":1791137714,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls -la /tmp\\\"}\"}}]},\"finish_reason\":null}]}\n\n"
	// A finish chunk completes the call, so its truncated arguments are held
	// as cut until the stream's terminal.
	chatFailureFinish = "data: {\"id\":\"gen-fixture\",\"object\":\"chat.completion.chunk\",\"created\":1791137714,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
	chatOverloaded    = "data: {\"error\":{\"message\":\"Overloaded\",\"type\":\"server_error\",\"param\":null,\"code\":\"provider_overloaded\"}}\n\n"

	responsesFailureCutStart = "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_fixture\",\"object\":\"response\",\"created_at\":100,\"model\":\"provider-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_fixture\",\"call_id\":\"call_fixture\",\"name\":\"bash\",\"arguments\":\"\",\"status\":\"in_progress\"}}\n\n" +
		"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"fc_fixture\",\"delta\":\"{\\\"command\\\": \\\"ls -la /tm\"}\n\n" +
		"event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"sequence_number\":3,\"output_index\":0,\"item_id\":\"fc_fixture\",\"name\":\"bash\",\"arguments\":\"{\\\"command\\\": \\\"ls -la /tm\"}\n\n" +
		"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":4,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_fixture\",\"call_id\":\"call_fixture\",\"name\":\"bash\",\"arguments\":\"{\\\"command\\\": \\\"ls -la /tm\",\"status\":\"incomplete\"}}\n\n"
	responsesFailed = "event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":5,\"response\":{\"id\":\"resp_fixture\",\"object\":\"response\",\"created_at\":100,\"model\":\"provider-model\",\"status\":\"failed\",\"output\":[],\"error\":{\"code\":\"server_error\",\"message\":\"Overloaded\"}}}\n\n"
)

type failureCutRun struct {
	wire      []byte
	events    []llmprotocol.Event
	pushErr   error
	finalErr  error
	lastFrame string
}

func runFailureCut(t *testing.T, source, target llmprotocol.WireFormat, body string, reason error) failureCutRun {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	var run failureCutRun
	frames, events, _, pushErr := stream.Push([]byte(body))
	run.pushErr = pushErr
	if reason == nil {
		reason = pushErr
	}
	final, finalEvents, _, finalErr := stream.Finalize(reason)
	run.finalErr = finalErr
	run.wire = append(bytes.Join(frames, nil), bytes.Join(final, nil)...)
	run.events = append(events, finalEvents...)
	sseFrames := strings.Split(strings.TrimSpace(string(run.wire)), "\n\n")
	run.lastFrame = sseFrames[len(sseFrames)-1]
	return run
}

// requireFailureNotCut holds what a held cut item may never do to a failed
// turn: become the error, or be presented as a complete call or a finished
// turn.
func requireFailureNotCut(t *testing.T, target llmprotocol.WireFormat, run failureCutRun) {
	t.Helper()
	if run.pushErr != nil || run.finalErr != nil {
		t.Fatalf("push error %v, finalize error %v", run.pushErr, run.finalErr)
	}
	if bytes.Contains(run.wire, []byte("invalid_stream_tool_arguments")) {
		t.Fatalf("the client was told the symptom, not the failure:\n%s", run.wire)
	}
	for _, event := range run.events {
		if event.Type == llmprotocol.EventResponseCompleted {
			t.Fatalf("a failed turn completed: %+v", event)
		}
		if event.ToolCall != nil && event.Type == llmprotocol.EventOutputItemCompleted && !event.ToolCall.Incomplete {
			t.Fatalf("a cut call completed as whole: %+v", event.ToolCall)
		}
	}
	switch target {
	case llmprotocol.AnthropicMessagesV1:
		if bytes.Contains(run.wire, []byte("event: message_stop")) || bytes.Contains(run.wire, []byte("event: message_delta")) {
			t.Fatalf("Messages client was given a finished message:\n%s", run.wire)
		}
	case llmprotocol.OpenAIChatV1:
		if bytes.Contains(run.wire, []byte("data: [DONE]")) || bytes.Contains(run.wire, []byte(`"finish_reason":"`)) {
			t.Fatalf("Chat client was given a finished choice:\n%s", run.wire)
		}
	case llmprotocol.OpenAIResponsesV1:
		for _, terminal := range []string{"event: response.completed", "event: response.incomplete", `"status":"completed"`} {
			if bytes.Contains(run.wire, []byte(terminal)) {
				t.Fatalf("Responses client was given %s:\n%s", terminal, run.wire)
			}
		}
	}
}

// failureFrame reads the client's last frame as the failure it carries: the
// event name and the error's type (Messages) or code (Chat, Responses).
func failureFrame(t *testing.T, target llmprotocol.WireFormat, frame string) (string, string) {
	t.Helper()
	var name, data string
	for _, line := range strings.Split(frame, "\n") {
		if value, found := strings.CutPrefix(line, "event: "); found {
			name = value
		}
		if value, found := strings.CutPrefix(line, "data: "); found {
			data = value
		}
	}
	var wire struct {
		Code  string `json:"code"`
		Error *struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
		Response *struct {
			Error *struct {
				Code string `json:"code"`
			} `json:"error"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(data), &wire); err != nil {
		t.Fatalf("last frame is not a failure: %q: %v", frame, err)
	}
	switch {
	case target == llmprotocol.AnthropicMessagesV1 && wire.Error != nil:
		return name, wire.Error.Type
	case wire.Response != nil && wire.Response.Error != nil:
		return name, wire.Response.Error.Code
	case wire.Error != nil:
		return name, wire.Error.Code
	default:
		return name, wire.Code
	}
}

// requireFailureFrame holds the client's last frame to the failure in its
// format: Messages' error event, Chat's error frame, and Responses' error
// event, or response.failed when the provider failed the response itself.
func requireFailureFrame(t *testing.T, target llmprotocol.WireFormat, run failureCutRun, wantCode string) {
	t.Helper()
	name, code := failureFrame(t, target, run.lastFrame)
	switch target {
	case llmprotocol.AnthropicMessagesV1:
		if name != "error" {
			t.Fatalf("Messages last event = %q, want error:\n%s", name, run.wire)
		}
	case llmprotocol.OpenAIChatV1:
		if name != "" || !strings.HasPrefix(run.lastFrame, `data: {"error":`) {
			t.Fatalf("Chat last frame is not its error frame:\n%s", run.wire)
		}
	case llmprotocol.OpenAIResponsesV1:
		if name != "error" && name != "response.failed" {
			t.Fatalf("Responses last event = %q, want error or response.failed:\n%s", name, run.wire)
		}
	}
	if code != wantCode {
		t.Fatalf("%s failure code = %q, want %q:\n%s", target, code, wantCode, run.wire)
	}
}

// withoutSequence drops a Responses frame's sequence_number, which counts
// the frames before it and so differs with the frames a held item adds.
func withoutSequence(frame string) string {
	start := strings.Index(frame, `"sequence_number":`)
	if start < 0 {
		return frame
	}
	end := start + len(`"sequence_number":`)
	for end < len(frame) && frame[end] >= '0' && frame[end] <= '9' {
		end++
	}
	return frame[:start] + frame[end:]
}

// A provider failure while a tool call is cut reaches every client format as
// that failure, exactly as it does when the call's arguments were whole.
func TestAProviderFailureDuringACutToolCallReachesTheClient(t *testing.T) {
	type upstream struct {
		source              llmprotocol.WireFormat
		cut, whole, failure string
		// wantCode is the failure's code (or Messages' error type) as each
		// client format carries it.
		wantCode map[llmprotocol.WireFormat]string
	}
	anthropicCodes := map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: "overloaded_error",
		llmprotocol.OpenAIChatV1:        "overloaded_error",
		llmprotocol.OpenAIResponsesV1:   "overloaded_error",
	}
	chatCodes := map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: "api_error",
		llmprotocol.OpenAIChatV1:        "provider_overloaded",
		llmprotocol.OpenAIResponsesV1:   "provider_overloaded",
	}
	for name, test := range map[string]upstream{
		// The block is still open when the error arrives.
		"anthropic, block open": {
			source: llmprotocol.AnthropicMessagesV1,
			cut:    anthropicFailureCutStart + anthropicFailureCutPartial, whole: anthropicFailureCutStart + anthropicFailureWhole,
			failure: anthropicOverloaded, wantCode: anthropicCodes,
		},
		// The block stopped, so the call is held as cut when the error
		// arrives in place of message_delta.
		"anthropic, block stopped": {
			source:  llmprotocol.AnthropicMessagesV1,
			cut:     anthropicFailureCutStart + anthropicFailureCutPartial + anthropicFailureBlockStop,
			whole:   anthropicFailureCutStart + anthropicFailureWhole + anthropicFailureBlockStop,
			failure: anthropicOverloaded, wantCode: anthropicCodes,
		},
		"chat, call open": {
			source: llmprotocol.OpenAIChatV1,
			cut:    chatFailureCutStart + chatFailureCutPartial, whole: chatFailureCutStart + chatFailureWhole,
			failure: chatOverloaded, wantCode: chatCodes,
		},
		"chat, call finished": {
			source:  llmprotocol.OpenAIChatV1,
			cut:     chatFailureCutStart + chatFailureCutPartial + chatFailureFinish,
			whole:   chatFailureCutStart + chatFailureWhole + chatFailureFinish,
			failure: chatOverloaded, wantCode: chatCodes,
		},
	} {
		for _, target := range builtinFormats {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				run := runFailureCut(t, test.source, target, test.cut+test.failure, nil)
				requireFailureNotCut(t, target, run)
				requireFailureFrame(t, target, run, test.wantCode[target])
				whole := runFailureCut(t, test.source, target, test.whole+test.failure, nil)
				if withoutSequence(run.lastFrame) != withoutSequence(whole.lastFrame) {
					t.Fatalf("the failure differs from the one a whole call gets:\ncut:   %s\nwhole: %s", run.lastFrame, whole.lastFrame)
				}
			})
		}
	}
}

// A Responses provider that marks its cut call incomplete and then fails the
// response: a Responses client receives response.failed with the provider's
// code, and the others its failure in their own format.
func TestAResponsesFailureAfterAnIncompleteCallReachesTheClient(t *testing.T) {
	for target, want := range map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: "api_error",
		llmprotocol.OpenAIChatV1:        "server_error",
		llmprotocol.OpenAIResponsesV1:   "server_error",
	} {
		t.Run(string(target), func(t *testing.T) {
			run := runFailureCut(t, llmprotocol.OpenAIResponsesV1, target, responsesFailureCutStart+responsesFailed, nil)
			requireFailureNotCut(t, target, run)
			requireFailureFrame(t, target, run, want)
			if target == llmprotocol.OpenAIResponsesV1 && !strings.HasPrefix(run.lastFrame, "event: response.failed") {
				t.Fatalf("Responses client did not receive response.failed:\n%s", run.wire)
			}
		})
	}
}

// When the Router ends the stream itself while a cut call is held (its own
// deadline cut, a deadline, a transport end), the client is told that, as
// it would be with no cut call held.
func TestARouterEndDuringACutToolCallReachesTheClient(t *testing.T) {
	truncated := llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, "stream_truncated",
		"the router ended this stream at its own deadline, below the platform's", nil)
	held := anthropicFailureCutStart + anthropicFailureCutPartial + anthropicFailureBlockStop
	for name, test := range map[string]struct {
		reason error
		code   map[llmprotocol.WireFormat]string
	}{
		"router deadline cut": {reason: truncated, code: map[llmprotocol.WireFormat]string{
			llmprotocol.AnthropicMessagesV1: "timeout_error", llmprotocol.OpenAIChatV1: "stream_truncated", llmprotocol.OpenAIResponsesV1: "stream_truncated",
		}},
		"deadline": {reason: context.DeadlineExceeded, code: map[llmprotocol.WireFormat]string{
			llmprotocol.AnthropicMessagesV1: "api_error", llmprotocol.OpenAIChatV1: "stream_timeout", llmprotocol.OpenAIResponsesV1: "stream_timeout",
		}},
		"transport end": {reason: io.ErrUnexpectedEOF, code: map[llmprotocol.WireFormat]string{
			llmprotocol.AnthropicMessagesV1: "api_error", llmprotocol.OpenAIChatV1: "stream_incomplete", llmprotocol.OpenAIResponsesV1: "stream_incomplete",
		}},
	} {
		for _, target := range builtinFormats {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				run := runFailureCut(t, llmprotocol.AnthropicMessagesV1, target, held, test.reason)
				requireFailureNotCut(t, target, run)
				requireFailureFrame(t, target, run, test.code[target])
				var failed *llmprotocol.ProtocolError
				for _, event := range run.events {
					if event.Type == llmprotocol.EventResponseFailed {
						failed = event.Error
					}
				}
				var wantErr *llmprotocol.ProtocolError
				if !errors.As(test.reason, &wantErr) {
					wantErr = streamFinalizationError(test.reason, "")
				}
				if failed == nil || failed.Code != wantErr.Code {
					t.Fatalf("failed event error = %+v, want code %q", failed, wantErr.Code)
				}
			})
		}
	}
}

// Controls: without a failure, a held cut call is decided as before. Under
// end_turn, or a stream that ends cleanly with no terminal, its arguments
// are malformed.
func TestACutToolCallWithoutAFailureStillFails(t *testing.T) {
	held := anthropicFailureCutStart + anthropicFailureCutPartial + anthropicFailureBlockStop
	for name, body := range map[string]string{"end_turn": held + anthropicEndTurn, "clean end": held} {
		for _, target := range builtinFormats {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				run := runFailureCut(t, llmprotocol.AnthropicMessagesV1, target, body, nil)
				err := run.pushErr
				if err == nil {
					err = run.finalErr
				}
				requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
			})
		}
	}
}
