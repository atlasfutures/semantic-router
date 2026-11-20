package extproc

import (
	"net/http"
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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

// admissibleContextWindow is the largest card context window among the arms
// the request's other constraints admit: vision on an image turn and
// capability, the masks no_context_arm is combined with. It is the length a
// shortened turn must fit. Zero when no arm is admitted or an admitted arm
// declares no window (that arm holds any length).
func (r *OpenAIRouter) admissibleContextWindow(modelRefs []config.ModelRef, arc *selection.RaylineARCSelectionContext) int {
	if r == nil || r.Config == nil || arc == nil {
		return 0
	}
	marked := func(mask []bool, index int) bool { return index < len(mask) && mask[index] }
	largest := 0
	for index, ref := range modelRefs {
		if (arc.ImageBearing && marked(arc.NonVisionArms, index)) || marked(arc.IncapableArms, index) {
			continue
		}
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
