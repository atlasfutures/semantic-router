package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var clientFormats = []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1}

// undecodableUpstreamBody is a Chat response whose usage the contract refuses:
// a negative prompt count, answered "negative_usage: upstream usage cannot be negative".
const undecodableUpstreamBody = `{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
	`"choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"hi"}}],` +
	`"usage":{"prompt_tokens":-3,"completion_tokens":64,"total_tokens":61}}`

// TestUpstreamDecodeFailureCarriesItsCodeAndMessage is the ask. The response
// path answers every codec refusal with one fixed sentence, so a truncated
// body, an unnamed field and a negative token count all read alike, past the
// boundary where the upstream body stops being inspectable.
func TestUpstreamDecodeFailureCarriesItsCodeAndMessage(t *testing.T) {
	for _, format := range clientFormats {
		t.Run(string(format), func(t *testing.T) {
			_, message := refuseUndecodableResponse(t, format)
			if !strings.Contains(message, "usage cannot be negative") {
				t.Errorf("message = %q, want the decode failure's own message", message)
			}
		})
	}
}

// TestUpstreamDecodeFailureIsGenericToday pins the present behaviour so a
// change to it shows up in the diff rather than only in the test above.
func TestUpstreamDecodeFailureIsGenericToday(t *testing.T) {
	for _, format := range clientFormats {
		t.Run(string(format), func(t *testing.T) {
			status, message := refuseUndecodableResponse(t, format)
			if status != 502 || message != "model service unavailable" {
				t.Errorf("%d %q: the recorded behaviour has changed", status, message)
			}
		})
	}
}

// refuseUndecodableResponse drives the real non-streaming response path with a
// refused body, and returns the status and error message the client reads.
func refuseUndecodableResponse(t *testing.T, format llmprotocol.WireFormat) (int, string) {
	t.Helper()
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	requestContext := &RequestContext{SourceFormat: format, TargetFormat: llmprotocol.OpenAIChatV1, TraceContext: context.Background()}
	response := router.handleNonStreamingResponseBody([]byte(undecodableUpstreamBody), requestContext, 0)
	immediate := router.encodeImmediateResponseForClient(response, requestContext).GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("response is not an immediate refusal: %+v", response)
	}
	var payload struct {
		Error struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal(immediate.GetBody(), &payload); err != nil {
		t.Fatalf("refusal body is not JSON: %s", immediate.GetBody())
	}
	return int(immediate.GetStatus().GetCode()), payload.Error.Message
}
