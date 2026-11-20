package extproc

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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

// The stated window is the largest card among the arms the turn's other
// constraints admit: an image turn ignores a text-only arm's larger window, an
// incapable arm is ignored, and an admitted arm with no declared window holds
// any length, so none is stated.
func TestAdmissibleContextWindow(t *testing.T) {
	window := func(n int) config.ModelParams { return config.ModelParams{ContextWindowSize: n} }
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.Config.ModelConfig = map[string]config.ModelParams{"a": window(4000), "b": window(128000), "c": {}}
	refs := []config.ModelRef{{Model: "a"}, {Model: "b"}}
	if got := router.admissibleContextWindow(refs, &selection.RaylineARCSelectionContext{}); got != 128000 {
		t.Fatalf("text turn = %d, want 128000", got)
	}
	image := &selection.RaylineARCSelectionContext{ImageBearing: true, NonVisionArms: []bool{false, true}}
	if got := router.admissibleContextWindow(refs, image); got != 4000 {
		t.Fatalf("image turn = %d, want the vision arm's 4000", got)
	}
	incapable := &selection.RaylineARCSelectionContext{IncapableArms: []bool{false, true}}
	if got := router.admissibleContextWindow(refs, incapable); got != 4000 {
		t.Fatalf("with an incapable arm = %d, want 4000", got)
	}
	if got := router.admissibleContextWindow([]config.ModelRef{{Model: "a"}, {Model: "c"}}, &selection.RaylineARCSelectionContext{}); got != 0 {
		t.Fatalf("with an undeclared window = %d, want 0", got)
	}
}
