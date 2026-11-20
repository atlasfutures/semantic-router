package extproc

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A turn no window can hold is the caller's to shorten: a 400 that says the
// prompt is too long in the provider's own words, in every wire format, with
// the estimate and the largest window, never the 503 a client retries.
func TestRaylineARCContextTooLongIsAPromptTooLong400(t *testing.T) {
	for _, class := range []string{arcFailureNoContextArm, arcFailureOverContextLimit} {
		for _, format := range []llmprotocol.WireFormat{
			llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
		} {
			router := &OpenAIRouter{}
			ctx := routingTestContext(format, testNeutralRequest("auto", "hi"))
			ctx.RaylineARCContextTokens, ctx.RaylineARCContextLimit = 1100000, 1050000
			response := router.authoritativeSelectionFailureResponse(
				&modelSelectionFailure{algorithm: "rayline_arc", class: class}, ctx,
			)
			if got := immediateHeaderValue(response, selectionFailureHeader); got != selectionFailureContextTooLong {
				t.Fatalf("%s %s: failure header = %q, want %s", class, format, got, selectionFailureContextTooLong)
			}
			encoded := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse()
			if status := int(encoded.GetStatus().GetCode()); status != http.StatusBadRequest {
				t.Fatalf("%s %s: status = %d, want 400", class, format, status)
			}
			body := string(encoded.GetBody())
			var decoded interface{}
			if err := json.Unmarshal(encoded.GetBody(), &decoded); err != nil {
				t.Fatalf("%s %s: body is not JSON: %s", class, format, body)
			}
			if !strings.Contains(fmt.Sprint(decoded), "prompt is too long: 1100000 tokens > 1050000 maximum") {
				t.Fatalf("%s %s: body = %s, want the prompt-too-long message", class, format, body)
			}
			if format == llmprotocol.AnthropicMessagesV1 && !strings.Contains(body, `"invalid_request_error"`) {
				t.Fatalf("%s: Anthropic body = %s, want invalid_request_error", class, body)
			}
			if got := selectionTurnFailureClass(class); got != turnFailureContextOverflow {
				t.Fatalf("%s: turn failure class = %q, want %s", class, got, turnFailureContextOverflow)
			}
		}
	}
}

// With no estimate or no window the message still starts as a provider's does.
func TestContextTooLongMessageFor(t *testing.T) {
	if got := contextTooLongMessageFor(0, 1050000); got != "prompt is too long" {
		t.Fatalf("no estimate: %q", got)
	}
	if got := contextTooLongMessageFor(1100000, 0); got != "prompt is too long" {
		t.Fatalf("no limit: %q", got)
	}
}

// The stated window is the largest arm card; an arm with no declared window
// holds any length, so none is stated.
func TestLargestContextWindow(t *testing.T) {
	window := func(n int) config.ModelParams { return config.ModelParams{ContextWindowSize: n} }
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.Config.ModelConfig = map[string]config.ModelParams{"a": window(1000000), "b": window(1050000), "c": {}}
	if got := router.largestContextWindow([]config.ModelRef{{Model: "a"}, {Model: "b"}}); got != 1050000 {
		t.Fatalf("largest = %d, want 1050000", got)
	}
	if got := router.largestContextWindow([]config.ModelRef{{Model: "a"}, {Model: "c"}}); got != 0 {
		t.Fatalf("with an undeclared window = %d, want 0", got)
	}
}
