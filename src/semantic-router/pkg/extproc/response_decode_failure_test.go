package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A response the router cannot decode still tells the operator why, in its
// code and its message. Answering only "invalid response" throws that away at
// the one boundary where it is the whole diagnosis.
func TestUpstreamDecodeFailureCarriesItsCodeAndMessage(t *testing.T) {
	for name, format := range map[string]llmprotocol.WireFormat{
		"chat":      llmprotocol.OpenAIChatV1,
		"anthropic": llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			requestContext := &RequestContext{
				SourceFormat: format, TargetFormat: llmprotocol.OpenAIChatV1,
				TraceContext: context.Background(),
			}
			decodeErr := llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"negative_usage",
				"upstream usage cannot be negative",
				nil,
			)
			_ = decodeErr
			// The real path: an upstream body the codec cannot decode.
			response := router.handleNonStreamingResponseBody(
				undecodableUpstreamBody(), requestContext, 0,
			)
			encoded := router.encodeImmediateResponseForClient(response, requestContext)
			immediate := encoded.GetImmediateResponse()
			if immediate == nil {
				t.Fatalf("response is not an immediate refusal: %+v", encoded)
			}
			if got := int(immediate.GetStatus().GetCode()); got != 502 {
				t.Fatalf("status = %d, want 502 for an unusable upstream response", got)
			}
			var body struct {
				Error struct {
					Message string `json:"message"`
					Type    string `json:"type"`
					Code    any    `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal(immediate.GetBody(), &body); err != nil {
				t.Fatalf("refusal body is not JSON: %s", immediate.GetBody())
			}
			if !strings.Contains(body.Error.Message, "usage cannot be negative") {
				t.Fatalf("message = %q, want the decode failure's own message", body.Error.Message)
			}
		})
	}
}

// undecodableUpstreamBody is a Chat response whose usage the contract refuses:
// the provider states more reasoning tokens than completion tokens, which is
// the shape arm 4 returns. What matters here is only that the decode fails
// with a named protocol error.
func undecodableUpstreamBody() []byte {
	return []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":-3,"completion_tokens":64,"total_tokens":61}}`)
}
