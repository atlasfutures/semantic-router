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

// arcFailureNoOutputArm is the bounded class for a turn that asks for more
// output than every arm its own constraints admit can produce. It is the
// caller's request that cannot be served, so it answers 400
// invalid_request_error with code max_tokens_exceeds_limit and the limit it
// may ask for; the router does not clamp the caller's max_tokens to fit an arm.
// The mask runs before the operator's disables and the privacy allow-list and
// reads no health state, so a refusal here is never an outage.
const arcFailureNoOutputArm = "no_output_arm"

// noOutputArmMessage is the refusal text for arcFailureNoOutputArm when no
// limit can be stated. It carries nothing from the request.
const noOutputArmMessage = "max_tokens exceeds every arm's output limit."

// noOutputArmCode is the public error code of the refusal.
const noOutputArmCode = "max_tokens_exceeds_limit"

// noOutputArmMessageFor names the largest output this request may ask for.
func noOutputArmMessageFor(limit int64) string {
	if limit <= 0 {
		return noOutputArmMessage
	}
	return "max_tokens exceeds the output limit of every model this request can use; the largest is " +
		strconv.FormatInt(limit, 10) + " tokens."
}

// admissibleOutputLimit is the largest card output limit among the arms the
// request's own constraints admit: vision on an image turn, capability and
// context window. Zero when no arm is admitted or an admitted arm declares no
// limit (a request then always fits that arm).
func (r *OpenAIRouter) admissibleOutputLimit(modelRefs []config.ModelRef, arc *selection.RaylineARCSelectionContext) int64 {
	if r == nil || r.Config == nil || arc == nil {
		return 0
	}
	marked := func(mask []bool, index int) bool { return index < len(mask) && mask[index] }
	var largest int64
	for index, ref := range modelRefs {
		if (arc.ImageBearing && marked(arc.NonVisionArms, index)) ||
			marked(arc.IncapableArms, index) || marked(arc.OverContextArms, index) {
			continue
		}
		params, known := r.Config.ModelConfig[strings.TrimSpace(ref.Model)]
		if !known || params.MaxOutputTokens <= 0 {
			return 0
		}
		if limit := int64(params.MaxOutputTokens); limit > largest {
			largest = limit
		}
	}
	return largest
}

// requestedOutputTokens reads the output allowance the caller asked for, from
// the neutral request, so Anthropic max_tokens, Chat max_tokens or
// max_completion_tokens, and Responses max_output_tokens all read the same.
// It answers zero when the caller stated no limit, which masks nothing.
//
// A limit the router itself wrote (a card default, RouterSetMaxOutputTokens)
// is not the caller's, and a router-raised allowance reads back the caller's
// own number (ClientMaxOutputTokens).
//
// A thinking budget counts as output: on Anthropic Messages thinking is billed
// and bounded inside max_tokens, and OpenRouter's reasoning.max_tokens counts
// toward output too. The requested output is the larger of the two, not their
// sum: each wire format already requires the budget to sit inside the
// allowance, so adding them would mask arms the request fits.
func requestedOutputTokens(request *llmprotocol.Request) int64 {
	if request == nil {
		return 0
	}
	var allowance int64
	switch {
	case request.ClientMaxOutputTokens != nil:
		allowance = *request.ClientMaxOutputTokens
	case request.RouterSetMaxOutputTokens:
	case request.Sampling.MaxOutputTokens != nil:
		allowance = *request.Sampling.MaxOutputTokens
	}
	if budget := request.ReasoningBudgetTokens; budget != nil && *budget > allowance {
		allowance = *budget
	}
	if allowance < 0 {
		return 0
	}
	return allowance
}

// overOutputArms reads each candidate's declared max_output_tokens against the
// requested output. It returns nil when nothing was requested or no arm is
// under the request, which leaves selection exactly as it was. An arm whose
// card declares no output limit is never masked.
//
// The exclusion is a refusal, like the context window: an arm whose output
// limit is below max_tokens answers 400 from the provider, so routing there
// produces no completion at all.
func (r *OpenAIRouter) overOutputArms(modelRefs []config.ModelRef, requested int64) []bool {
	if r == nil || r.Config == nil || requested <= 0 || len(modelRefs) == 0 {
		return nil
	}
	marked := false
	arms := make([]bool, len(modelRefs))
	for index, ref := range modelRefs {
		params, known := r.Config.ModelConfig[strings.TrimSpace(ref.Model)]
		if known && params.MaxOutputTokens > 0 && int64(params.MaxOutputTokens) < requested {
			arms[index] = true
			marked = true
		}
	}
	if !marked {
		return nil
	}
	return arms
}

// withOverOutputArms folds the output-length constraint into the mask. It is
// a fact about this turn, so it runs beside the context window and before the
// operator's standing verdict.
func withOverOutputArms(
	arcContext *selection.RaylineARCSelectionContext,
	armCount int,
	excluded []bool,
) ([]bool, error) {
	if len(arcContext.OverOutputArms) == 0 {
		return excluded, nil
	}
	if len(arcContext.OverOutputArms) != armCount {
		return nil, arcSelectionFailure("output_arm_mapping")
	}
	combined := make([]bool, armCount)
	eligible := 0
	for index := range combined {
		combined[index] = arcContext.OverOutputArms[index] ||
			(len(excluded) == armCount && excluded[index])
		if !combined[index] {
			eligible++
		}
	}
	if eligible == 0 {
		return nil, arcSelectionFailure(arcFailureNoOutputArm)
	}
	return combined, nil
}

// maskedArmIndices lists the marked ordinals of a mask, for the log.
func maskedArmIndices(mask []bool) []int {
	indices := []int{}
	for index, marked := range mask {
		if marked {
			indices = append(indices, index)
		}
	}
	return indices
}

// addOutputMaskFields puts the output-length mask on rayline_arc_selection:
// the requested output and the masked ordinals, numbers only. A turn that
// stated no output limit carries neither.
func addOutputMaskFields(fields map[string]interface{}, ctx *RequestContext) {
	if ctx.RaylineARCRequestedOutputTokens <= 0 {
		return
	}
	fields["requested_output_tokens"] = ctx.RaylineARCRequestedOutputTokens
	fields["over_output_arms"] = maskedArmIndices(ctx.RaylineARCOverOutputArms)
	fields["over_output_arms_count"] = len(maskedArmIndices(ctx.RaylineARCOverOutputArms))
}

// noOutputArmResponse refuses the turn as the caller's error: a 400 with code
// max_tokens_exceeds_limit and the largest limit it may ask for. It is not a
// 503: no backend was called, the router is healthy, and the same request is
// refused identically every time, so a 503 would read as provider downtime and
// invite a retry that cannot succeed. The protocol error rides the request so
// the client's final encoding keeps the message.
func (r *OpenAIRouter) noOutputArmResponse(ctx *RequestContext) *ext_proc.ProcessingResponse {
	message := noOutputArmMessage
	if ctx != nil {
		message = noOutputArmMessageFor(ctx.RaylineARCOutputLimit)
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			noOutputArmCode,
			message,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusBadRequest, message)
}
