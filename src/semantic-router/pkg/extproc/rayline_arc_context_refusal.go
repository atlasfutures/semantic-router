package extproc

import (
	"net/http"
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// contextTooLongCode is the public error code of a turn longer than every
// window it could be served in (no_context_arm, over_context_limit).
const contextTooLongCode = "context_length_exceeded"

// contextTooLongMessage is the refusal text when no count or limit can be
// stated. It starts the way a provider's own over-length refusal does, so a
// client that compacts its conversation on that refusal does so here too.
const contextTooLongMessage = "prompt is too long"

// contextTooLongMessageFor states the turn's estimated length and the largest
// window it could be served in, in the provider's own form.
func contextTooLongMessageFor(tokens, limit int) string {
	if tokens <= 0 || limit <= 0 {
		return contextTooLongMessage
	}
	return contextTooLongMessage + ": " + strconv.Itoa(tokens) + " tokens > " + strconv.Itoa(limit) + " maximum"
}

// largestContextWindow is the largest card context window among the arms;
// zero when an arm declares none (that arm holds any length, so no turn is
// over-length) or there are no arms.
func (r *OpenAIRouter) largestContextWindow(modelRefs []config.ModelRef) int {
	if r == nil || r.Config == nil {
		return 0
	}
	largest := 0
	for _, ref := range modelRefs {
		params, known := r.Config.ModelConfig[strings.TrimSpace(ref.Model)]
		if !known || params.ContextWindowSize <= 0 {
			return 0
		}
		if params.ContextWindowSize > largest {
			largest = params.ContextWindowSize
		}
	}
	return largest
}

// contextTooLongResponse refuses the turn as the caller's error: a 400
// invalid request saying the prompt is too long. It is not a 503: no backend
// was called, the router is healthy, and the same request is refused
// identically every time, so a 503 reads as an outage and invites a retry that
// cannot succeed. The gateway passes a cell's refusal to the client as it is
// and replays no cell turn, so the 400 is what lets a client shorten the
// conversation instead of retrying it. The protocol error rides the request
// so the client's final encoding keeps the message.
func (r *OpenAIRouter) contextTooLongResponse(ctx *RequestContext) *ext_proc.ProcessingResponse {
	message := contextTooLongMessage
	if ctx != nil {
		message = contextTooLongMessageFor(ctx.RaylineARCContextTokens, ctx.RaylineARCContextLimit)
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			contextTooLongCode,
			message,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusBadRequest, message)
}
