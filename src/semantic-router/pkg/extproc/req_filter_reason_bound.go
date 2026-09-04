package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// applyOpenRouterReasoningBound states how many tokens a reasoning turn may
// spend, on the one backend whose wire has a control for it.
//
// max_completion_tokens does not bound a reasoning model: the models behind
// the thinking-on arms do not count reasoning against it. Unbounded, a
// thinking turn runs until it finishes or until the platform closes the
// connection, which is the failure this removes.
//
// The Chat encoder already sends OpenRouter's reasoning.max_tokens for a turn
// the client asked to reason. An arm that reasons by its own configuration is
// the case the encoder cannot see: the client's request says nothing about
// thinking, so the encoder derives nothing, and this boundary is where the
// arm's setting is applied. The bound is derived the same way -- what the
// client allowed for output, floored -- so the two are one rule.
//
// The bound travels alone. OpenRouter reads a top-level reasoning_effort as
// reasoning.effort, and its documentation puts "One of the following (not
// both):" above effort and max_tokens; a body stating both is refused with
//
//	Only one of "reasoning.effort" and "reasoning.max_tokens" can be specified
//
// which answered every thinking-arm turn on the dev cell 2026-09-04. So the
// arm's configured effort is dropped for a bounded turn, and the bound is the
// only reasoning control the request carries. Read 2026-09-04:
//
//	https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
//
// A request with no output allowance keeps its effort level: there is nothing
// to derive a bound from, and capping a client that asked for no cap is not
// the Router's to do.
func applyOpenRouterReasoningBound(
	mutation *reasoningRequestMutation,
	dialect openAIBackendDialect,
	clientAllowance *int64,
) {
	if !dialect.usesReasoningObjectBound() || !mutation.reasoningApplied {
		return
	}
	bound := reasoningBoundForRequest(mutation.requestMap, clientAllowance)
	if bound == nil {
		return
	}
	mutation.requestMap["reasoning"] = encodeReasoningBound(mutation.requestMap["reasoning"], *bound)
	delete(mutation.requestMap, "reasoning_effort")
	mutation.appliedEffort = ""
	mutation.reasoningBound = bound
}

// dropReasoningRequestFromDisabledArm removes the reasoning controls from a
// turn the decision routed to an arm that must not reason.
//
// The controls on the body at this point are the client's, rendered by the
// Chat encoder. Claude Code sends adaptive thinking with output_config.effort
// high on every turn, which the encoder renders as a bound with an effort
// beside it; measured on the dev cell 2026-09-04, a
// deepseek-v4-pro@thinking-off turn was dispatched with reasoning.max_tokens
// 32000. Whether the turn reasons is the Router's decision, not the client's,
// and the arm is the answer.
//
// The reasoning object goes whatever the backend is: it is OpenRouter's
// control and it can only ask a model to reason. The top-level effort goes
// only where OpenRouter reads it, because the other dialects preserve a
// client-supplied effort deliberately -- a vLLM chat template needs the
// argument, and OpenAI takes the level as the client's own.
func dropReasoningRequestFromDisabledArm(
	mutation *reasoningRequestMutation,
	dialect openAIBackendDialect,
	ctx *RequestContext,
) {
	_, hadBound := mutation.requestMap["reasoning"]
	delete(mutation.requestMap, "reasoning")

	droppedEffort := false
	if dialect.usesReasoningObjectBound() {
		droppedEffort = reasoningEffortAsksToReason(mutation.requestMap["reasoning_effort"])
		delete(mutation.requestMap, "reasoning_effort")
		mutation.appliedEffort = ""
	}
	if hadBound || droppedEffort {
		recordDroppedReasoningRequest(ctx)
	}
}

// reasoningEffortAsksToReason reports whether an effort level is a request to
// reason. "none" is the off-signal, so dropping it loses nothing.
func reasoningEffortAsksToReason(raw json.RawMessage) bool {
	var effort string
	if len(raw) == 0 || json.Unmarshal(raw, &effort) != nil {
		return false
	}
	return effort != "" && effort != "none"
}

// recordDroppedReasoningRequest counts a client control the Router removed,
// through the diagnostics the response header and the lossy counter already
// read. The turn still runs: a thinking-off answer is the routable outcome and
// refusing the conversation is not.
func recordDroppedReasoningRequest(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, llmprotocol.Diagnostic{
		Source: ctx.SourceFormat,
		Target: ctx.TargetFormat,
		Field:  "reasoning",
		Action: llmprotocol.DiagnosticDropped,
		Reason: "reasoning_disabled_by_selected_model",
	})
}

// reasoningBoundForRequest reads the bound the request already carries, or
// derives one from the output allowance, or reports that there is none.
//
// The allowance is the client's own where the two differ. The request_params
// floor raises max_completion_tokens on the dispatched body so an answer has
// room beside the thinking, and a bound derived from the raised number would
// bound nothing: measured on the dev cell 2026-09-04, a client asking for 512
// output tokens against the arms' 65,536 floor was told it could spend 65,536
// tokens reasoning.
func reasoningBoundForRequest(
	requestMap map[string]json.RawMessage,
	clientAllowance *int64,
) *int64 {
	if carried := carriedReasoningBound(requestMap["reasoning"]); carried != nil {
		return carried
	}
	allowance := clientAllowance
	if allowance == nil {
		allowance = outputAllowance(requestMap)
	}
	if allowance == nil {
		return nil
	}
	bound := protocolcodec.ReasoningBoundForOutputAllowance(*allowance)
	return &bound
}

func carriedReasoningBound(raw json.RawMessage) *int64 {
	if len(raw) == 0 {
		return nil
	}
	var object struct {
		MaxTokens *int64 `json:"max_tokens"`
	}
	if json.Unmarshal(raw, &object) != nil {
		return nil
	}
	return object.MaxTokens
}

// outputAllowance is what the client allowed the turn to write. Chat spells it
// max_completion_tokens; max_tokens is the older name the same wire still
// accepts, and a request that reaches here through another leg may carry it.
func outputAllowance(requestMap map[string]json.RawMessage) *int64 {
	for _, field := range []string{"max_completion_tokens", "max_tokens"} {
		raw, present := requestMap[field]
		if !present {
			continue
		}
		var allowance int64
		if json.Unmarshal(raw, &allowance) != nil || allowance <= 0 {
			continue
		}
		return &allowance
	}
	return nil
}

// encodeReasoningBound writes max_tokens into the reasoning object the request
// already has, keeping whatever else it holds and dropping the effort level
// the object form of the same control would carry.
func encodeReasoningBound(existing json.RawMessage, bound int64) json.RawMessage {
	object := map[string]json.RawMessage{}
	if len(existing) > 0 {
		if json.Unmarshal(existing, &object) != nil {
			object = map[string]json.RawMessage{}
		}
	}
	encoded, err := json.Marshal(bound)
	if err != nil {
		return existing
	}
	object["max_tokens"] = encoded
	delete(object, "effort")
	rendered, err := json.Marshal(object)
	if err != nil {
		return existing
	}
	return rendered
}

// appliedControl names the reasoning control the request ends up carrying, so
// the mutation log says what was sent rather than what was configured. Exactly
// one control travels, so the line names one.
func (mutation *reasoningRequestMutation) appliedControl() string {
	if mutation.reasoningBound != nil {
		return fmt.Sprintf("a bound of %d reasoning tokens", *mutation.reasoningBound)
	}
	return fmt.Sprintf("effort (%s)", mutation.appliedEffort)
}
