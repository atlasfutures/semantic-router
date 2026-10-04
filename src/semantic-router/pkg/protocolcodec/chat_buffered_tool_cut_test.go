package protocolcodec

import (
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const chatBufferedCutArguments = `{"command":"cat > /tmp/probe3.js <<'EOF'\n$(for i in $(seq 1 300); do`

// chatBufferedToolCall is a non-streaming Chat reply holding one tool call,
// with the given finish reason and, when not empty, OpenRouter's native one.
func chatBufferedToolCall(arguments, finish, native string) []byte {
	nativeMember := ""
	if native != "" {
		nativeMember = `,"native_finish_reason":` + strconv.Quote(native)
	}
	return []byte(`{"id":"gen-fixture","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
		`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_fixture","type":"function",` +
		`"function":{"name":"bash","arguments":` + strconv.Quote(arguments) + `}}]},` +
		`"finish_reason":` + strconv.Quote(finish) + nativeMember + `}],` +
		`"usage":{"prompt_tokens":92,"completion_tokens":40,"total_tokens":132}}`)
}

// A non-streaming Chat upstream cut off mid tool call is a length stop, in
// OpenAI's form (finish_reason length) and OpenRouter's (tool_calls with a
// native length reason), and every client format receives it as one.
func TestABufferedChatToolCutIsALengthStop(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, body := range map[string][]byte{
		"finish_reason length":          chatBufferedToolCall(chatBufferedCutArguments, "length", ""),
		"native max_output_tokens":      chatBufferedToolCall(chatBufferedCutArguments, "tool_calls", "max_output_tokens"),
		"native max_tokens":             chatBufferedToolCall(chatBufferedCutArguments, "tool_calls", "max_tokens"),
		"native length":                 chatBufferedToolCall(chatBufferedCutArguments, "tool_calls", "length"),
		"length with native length too": chatBufferedToolCall(chatBufferedCutArguments, "length", "length"),
	} {
		t.Run(name, func(t *testing.T) {
			response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body)
			if err != nil {
				t.Fatal(err)
			}
			call := response.Output[0].Content[len(response.Output[0].Content)-1].ToolCall
			if response.StopReason != llmprotocol.StopMaxTokens || call == nil || !call.Incomplete ||
				call.Arguments != chatBufferedCutArguments {
				t.Fatalf("decoded response = %+v, call = %+v", response, call)
			}
			for _, target := range []llmprotocol.WireFormat{
				llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
			} {
				if _, err := engine.EncodeResponse(target, response, llmprotocol.Envelope{}); err != nil {
					t.Fatalf("encode to %s: %v", target, err)
				}
			}
		})
	}
}

// The controls: truncated arguments under any stop but a length stop, a
// whole non-object, or text no bytes could complete, stay malformed.
func TestABufferedChatToolCutWithoutALengthStopStillFails(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, body := range map[string][]byte{
		"finish_reason stop":          chatBufferedToolCall(chatBufferedCutArguments, "stop", ""),
		"tool_calls, no native":       chatBufferedToolCall(chatBufferedCutArguments, "tool_calls", ""),
		"tool_calls, native stop":     chatBufferedToolCall(chatBufferedCutArguments, "tool_calls", "stop"),
		"length, whole non-object":    chatBufferedToolCall(`[1, 2]`, "length", ""),
		"length, no possible object":  chatBufferedToolCall(`{"x":]`, "length", ""),
		"length, not an object start": chatBufferedToolCall(`run ls`, "length", ""),
	} {
		t.Run(name, func(t *testing.T) {
			_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body)
			requireProtocolErrorCode(t, err, "invalid_tool_call")
		})
	}
	// Only the last call can be the one the limit cut: a truncated call
	// followed by a whole one is malformed.
	notLast := []byte(`{"id":"gen-fixture","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
		`"message":{"role":"assistant","content":null,"tool_calls":[` +
		`{"id":"call_1","type":"function","function":{"name":"bash","arguments":` + strconv.Quote(chatBufferedCutArguments) + `}},` +
		`{"id":"call_2","type":"function","function":{"name":"bash","arguments":"{}"}}]},` +
		`"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, notLast)
	requireProtocolErrorCode(t, err, "invalid_tool_call")
}
