package protocolcodec

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// A turn on claude-haiku-5.5 with max_tokens 1 spent its one completion
// token on a part the provider never emitted: stop_reason max_tokens and no
// output at all. The response validator read that as a malformed success
// (empty_response_output) and the Router served a 502, so health checks that
// probe with a tiny max_tokens read the route as down. A length stop before
// the first part is a normal, if empty, reply; every target renders it as
// the success it is.

// emptyBufferedBody is a buffered upstream response with no output under the
// given source stop (the source protocol's own spelling).
func emptyBufferedBody(source llmprotocol.WireFormat, stop string) []byte {
	switch source {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"id":"gen-empty","object":"chat.completion","created":1793318400,"model":"m",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"` + stop + `"}],` +
			`"usage":{"prompt_tokens":12,"completion_tokens":1,"total_tokens":13}}`)
	case llmprotocol.OpenAIResponsesV1:
		status, details := "completed", `null`
		if stop != "" {
			status, details = "incomplete", `{"reason":"`+stop+`"}`
		}
		return []byte(`{"id":"resp-empty","object":"response","created_at":1793318400,"model":"m","status":"` + status +
			`","incomplete_details":` + details + `,"output":[],"usage":{"input_tokens":12,"output_tokens":1,"total_tokens":13}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"id":"msg-empty","type":"message","role":"assistant","model":"m","content":[],` +
			`"stop_reason":"` + stop + `","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1}}`)
	default:
		panic("unknown source format")
	}
}

// emptyStreamBody is an upstream stream that opens, writes no output part and
// ends under the given source stop.
func emptyStreamBody(source llmprotocol.WireFormat, stop string) []byte {
	switch source {
	case llmprotocol.OpenAIChatV1:
		return []byte(`data: {"id":"gen-empty","object":"chat.completion.chunk","created":1793318400,"model":"m",` +
			`"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n" +
			`data: {"id":"gen-empty","object":"chat.completion.chunk","created":1793318400,"model":"m",` +
			`"choices":[{"index":0,"delta":{},"finish_reason":"` + stop + `"}],` +
			`"usage":{"prompt_tokens":12,"completion_tokens":1,"total_tokens":13}}` + "\n\n" +
			"data: [DONE]\n\n")
	case llmprotocol.OpenAIResponsesV1:
		event, status, details := "response.completed", "completed", `null`
		if stop != "" {
			event, status, details = "response.incomplete", "incomplete", `{"reason":"`+stop+`"}`
		}
		return []byte("event: response.created\n" +
			`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp-empty","object":"response","created_at":1,"model":"m","status":"in_progress","output":[]}}` + "\n\n" +
			"event: " + event + "\n" +
			`data: {"type":"` + event + `","sequence_number":1,"response":{"id":"resp-empty","object":"response","created_at":1,"model":"m","status":"` + status +
			`","incomplete_details":` + details + `,"output":[],"usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":13}}}` + "\n\n")
	case llmprotocol.AnthropicMessagesV1:
		return []byte("event: message_start\n" +
			`data: {"type":"message_start","message":{"id":"msg-empty","type":"message","role":"assistant","content":[],"model":"m","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1}}}` + "\n\n" +
			"event: message_delta\n" +
			`data: {"type":"message_delta","delta":{"stop_reason":"` + stop + `","stop_sequence":null},"usage":{"output_tokens":1}}` + "\n\n" +
			"event: message_stop\n" + `data: {"type":"message_stop"}` + "\n\n")
	default:
		panic("unknown source format")
	}
}

// maxTokensSourceStop is each source protocol's spelling of a max_tokens stop.
var maxTokensSourceStop = map[llmprotocol.WireFormat]string{
	llmprotocol.OpenAIChatV1:        "length",
	llmprotocol.OpenAIResponsesV1:   "max_output_tokens",
	llmprotocol.AnthropicMessagesV1: "max_tokens",
}

// emptyLengthStopWant is what each client format must carry for a reply that
// wrote nothing under a length stop, buffered and streamed.
var emptyLengthStopWant = map[llmprotocol.WireFormat]struct{ buffered, stream []string }{
	llmprotocol.OpenAIChatV1: {
		buffered: []string{`"message":{"role":"assistant","content":""}`, `"finish_reason":"length"`},
		stream:   []string{`"finish_reason":"length"`, "data: [DONE]"},
	},
	llmprotocol.AnthropicMessagesV1: {
		buffered: []string{`"content":[]`, `"stop_reason":"max_tokens"`},
		stream:   []string{"event: message_delta", `"stop_reason":"max_tokens"`, "event: message_stop"},
	},
	llmprotocol.OpenAIResponsesV1: {
		buffered: []string{`"status":"incomplete"`, `"incomplete_details":{"reason":"max_output_tokens"}`, `"output":[]`},
		stream:   []string{"event: response.incomplete", `"incomplete_details":{"reason":"max_output_tokens"}`},
	},
}

func TestEmptyOutputUnderMaxTokensIsASuccessOnEveryFormat(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		want := emptyLengthStopWant[target]
		t.Run("buffered", func(t *testing.T) {
			body := translateEmptyBuffered(t, engine, source, target, maxTokensSourceStop[source])
			assertWireContains(t, body, want.buffered)
		})
		t.Run("stream", func(t *testing.T) {
			wire, err := runEmptyStream(engine, source, target, emptyStreamBody(source, maxTokensSourceStop[source]))
			if err != nil {
				t.Fatalf("stream failed: %v\n%s", err, wire)
			}
			assertWireContains(t, wire, want.stream)
			assertNoStreamErrorFrame(t, wire)
		})
	})
}

// Anthropic's model_context_window_exceeded is the other length stop: a
// prompt that fills the window leaves no room to write, so it may end with
// nothing written just as max_tokens does, and is served the same way.
func TestEmptyOutputUnderContextWindowStopIsASuccess(t *testing.T) {
	engine := NewBuiltinEngine()
	source := llmprotocol.AnthropicMessagesV1
	for _, target := range builtinFormats {
		t.Run(string(target), func(t *testing.T) {
			body := translateEmptyBuffered(t, engine, source, target, "model_context_window_exceeded")
			wire, err := runEmptyStream(engine, source, target, emptyStreamBody(source, "model_context_window_exceeded"))
			if err != nil {
				t.Fatalf("stream failed: %v\n%s", err, wire)
			}
			assertNoStreamErrorFrame(t, wire)
			assertWireContains(t, wire, emptyLengthStopWant[target].stream[len(emptyLengthStopWant[target].stream)-1:])
			if target == llmprotocol.AnthropicMessagesV1 {
				assertWireContains(t, body, []string{`"content":[]`, `"stop_reason":"model_context_window_exceeded"`})
				assertWireContains(t, wire, []string{`"stop_reason":"model_context_window_exceeded"`})
			}
		})
	}
}

// Every other stop with nothing written is still a malformed success: the
// provider said the reply finished and sent no reply.
func TestEmptyOutputUnderOtherStopsStillFails(t *testing.T) {
	engine := NewBuiltinEngine()
	buffered := []struct {
		source llmprotocol.WireFormat
		stop   string
	}{
		{llmprotocol.OpenAIChatV1, "stop"},
		{llmprotocol.OpenAIChatV1, "tool_calls"},
		{llmprotocol.OpenAIResponsesV1, ""},
		{llmprotocol.AnthropicMessagesV1, "end_turn"},
		{llmprotocol.AnthropicMessagesV1, "tool_use"},
		{llmprotocol.AnthropicMessagesV1, "pause_turn"},
	}
	for _, test := range buffered {
		for _, target := range builtinFormats {
			t.Run(fmt.Sprintf("buffered/%s/%s/%s", test.source, test.stop, target), func(t *testing.T) {
				result, err := engine.TranslateResponse(test.source, target, emptyBufferedBody(test.source, test.stop), nil)
				if err == nil {
					t.Fatalf("empty output under %q was served as a success: %s", test.stop, result.Body)
				}
			})
		}
	}
	streams := []struct {
		source llmprotocol.WireFormat
		stop   string
	}{
		{llmprotocol.OpenAIResponsesV1, ""},
		{llmprotocol.AnthropicMessagesV1, "end_turn"},
		{llmprotocol.AnthropicMessagesV1, "tool_use"},
	}
	for _, test := range streams {
		for _, target := range builtinFormats {
			t.Run(fmt.Sprintf("stream/%s/%s/%s", test.source, test.stop, target), func(t *testing.T) {
				wire, err := runEmptyStream(engine, test.source, target, emptyStreamBody(test.source, test.stop))
				if err == nil && !streamCarriesFailure(target, wire) {
					t.Fatalf("empty stream under %q was served as a success:\n%s", test.stop, wire)
				}
				assertStreamHasNoSuccessTerminal(t, wire)
			})
		}
	}
}

func translateEmptyBuffered(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat, stop string) []byte {
	t.Helper()
	// The Router always rewrites the model, so the body is encoded rather
	// than handed back as the provider's bytes.
	result, err := engine.TranslateResponse(source, target, emptyBufferedBody(source, stop), func(response *llmprotocol.Response) error {
		response.Model = "public-model"
		return nil
	})
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	// The client's own codec reads the body back as the same empty success.
	decoded, _, _, err := engine.DecodeResponse(target, result.Body)
	if err != nil {
		t.Fatalf("target codec rejects its own body: %v\n%s", err, result.Body)
	}
	if err := llmprotocol.ValidateResponse(decoded, engine.policy.Limits); err != nil {
		t.Fatalf("target body fails validation: %v\n%s", err, result.Body)
	}
	if len(decoded.Output) != 0 || !llmprotocol.LengthStop(decoded.StopReason) {
		t.Fatalf("target body decoded as output=%+v stop=%q:\n%s", decoded.Output, decoded.StopReason, result.Body)
	}
	return result.Body
}

func runEmptyStream(engine *Engine, source, target llmprotocol.WireFormat, body []byte) ([]byte, error) {
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		return nil, err
	}
	frames, _, _, pushErr := stream.Push(body)
	finalFrames, _, _, finalErr := stream.Finalize(nil)
	wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
	if pushErr != nil {
		return wire, pushErr
	}
	return wire, finalErr
}

func streamCarriesFailure(target llmprotocol.WireFormat, wire []byte) bool {
	switch target {
	case llmprotocol.OpenAIResponsesV1:
		return bytes.Contains(wire, []byte("response.failed")) || bytes.Contains(wire, []byte("event: error"))
	default:
		return bytes.Contains(wire, []byte(`"error"`))
	}
}

func assertNoStreamErrorFrame(t *testing.T, wire []byte) {
	t.Helper()
	for _, marker := range []string{"event: error", "response.failed", `"error":{`} {
		if bytes.Contains(wire, []byte(marker)) {
			t.Fatalf("an empty length-stopped stream carried an error frame (%s):\n%s", marker, wire)
		}
	}
}

func assertWireContains(t *testing.T, wire []byte, want []string) {
	t.Helper()
	for _, fragment := range want {
		if !strings.Contains(string(wire), fragment) {
			t.Fatalf("wire lacks %s:\n%s", fragment, wire)
		}
	}
}
