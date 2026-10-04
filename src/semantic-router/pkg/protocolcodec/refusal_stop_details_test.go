package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A refusal says why it refused. The provider's own stop string
// (OpenRouter's native_finish_reason) and Anthropic's stop_details (its type
// and category) are kept on the neutral response as Router telemetry, so a
// refused turn's diagnostics can tell a classifier's refusal from another.
// No codec publishes them to a client.
func TestRefusalStopDetailsAreDecoded(t *testing.T) {
	engine := NewBuiltinEngine()
	stopDetails := `"stop_details":{"type":"refusal","category":"cyber","explanation":"Flagged."}`
	cyber := &llmprotocol.StopDetails{Type: "refusal", Category: "cyber"}
	chatRefusal := func(details string) string {
		return `{"id":"gen-1","object":"chat.completion","created":1,"model":"anthropic/claude-opus-5","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":null,"refusal":"Declined."},` +
			`"finish_reason":"content_filter","native_finish_reason":"refusal"` + details + `}]}`
	}
	chunk := func(choice string) string {
		return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[` + choice + `]}` + "\n\n"
	}
	chatStream := chunk(`{"index":0,"delta":{"role":"assistant","content":"I can"},"finish_reason":null}`) +
		chunk(`{"index":0,"delta":{},"finish_reason":"content_filter","native_finish_reason":"refusal",`+stopDetails+`}`) +
		"data: [DONE]\n\n"
	messagesRefusal := `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5","content":[],` +
		`"stop_reason":"refusal","stop_sequence":null,` + stopDetails + `,"usage":{"input_tokens":1,"output_tokens":0}}`
	messagesStream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_sequence\":null," + stopDetails + "},\"usage\":{\"output_tokens\":0}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	for _, tc := range []struct {
		name         string
		format       llmprotocol.WireFormat
		body         string
		stream       bool
		wantProvider string
		wantDetails  *llmprotocol.StopDetails
	}{
		{"Chat, with stop_details", llmprotocol.OpenAIChatV1, chatRefusal("," + stopDetails), false, "refusal", cyber},
		{"Chat, native finish reason only", llmprotocol.OpenAIChatV1, chatRefusal(""), false, "refusal", nil},
		{"Chat stream", llmprotocol.OpenAIChatV1, chatStream, true, "refusal", cyber},
		{"Messages", llmprotocol.AnthropicMessagesV1, messagesRefusal, false, "", cyber},
		{"Messages stream", llmprotocol.AnthropicMessagesV1, messagesStream, true, "", cyber},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var response llmprotocol.Response
			var err error
			if tc.stream {
				response, _, err = engine.DecodeResponseStream(tc.format, []byte(tc.body), llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "m", ProviderModel: "m",
				})
			} else {
				response, _, _, err = engine.DecodeResponse(tc.format, []byte(tc.body))
			}
			if err != nil {
				t.Fatal(err)
			}
			if response.ProviderStopReason != tc.wantProvider {
				t.Fatalf("ProviderStopReason = %q, want %q", response.ProviderStopReason, tc.wantProvider)
			}
			if (response.StopDetails == nil) != (tc.wantDetails == nil) ||
				(tc.wantDetails != nil && *response.StopDetails != *tc.wantDetails) {
				t.Fatalf("StopDetails = %+v, want %+v", response.StopDetails, tc.wantDetails)
			}
		})
	}
}
