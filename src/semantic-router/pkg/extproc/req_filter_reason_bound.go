package extproc

import (
	"encoding/json"
	"fmt"

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
// An effort level travels beside the bound, because a bound alone is ignored.
// Measured on the dev cell 2026-09-04, reasoning.max_tokens 1024 by itself left
// xiaomi/mimo-v2.5-pro@thinking-on spending 21,674 reasoning tokens and
// deepseek/deepseek-v4-flash@thinking-on spending 20,974; the build that sent
// an effort level beside the bound spent 16,030 on the same shape, and was
// answered 200, so the docs' "One of the following (not both)" is not enforced.
//
// The level is not the arm's. An arm-chosen dial above the bound is what made
// the bound inert, so the level is derived from the bound by OpenRouter's own
// documented conversion and the two controls then say the same thing. An
// effort the client stated is the exception: it is the client's number and it
// stands. Read 2026-09-04:
//
//	https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
//
// A request with no output allowance keeps its effort level: there is nothing
// to derive a bound from, and capping a client that asked for no cap is not
// the Router's to do.
func applyOpenRouterReasoningBound(mutation *reasoningRequestMutation, dialect openAIBackendDialect) {
	if !dialect.usesReasoningObjectBound() || !mutation.reasoningApplied {
		return
	}
	bound := reasoningBoundForRequest(mutation.requestMap)
	if bound == nil {
		return
	}
	mutation.requestMap["reasoning"] = encodeReasoningBound(mutation.requestMap["reasoning"], *bound)
	mutation.appliedEffort = mutation.effortBesideBound(*bound)
	mutation.requestMap["reasoning_effort"] = reasoningStringValue(mutation.appliedEffort)
	mutation.reasoningBound = bound
}

// effortBesideBound is the effort level that travels with the bound: the
// client's own when the request stated one, otherwise the level OpenRouter's
// documented conversion gives that bound. With no output allowance there is
// nothing to convert against, and the arm's own level stands.
func (mutation *reasoningRequestMutation) effortBesideBound(bound int64) string {
	if stated := mutation.statedReasoningEffort(); stated != "" {
		return stated
	}
	if allowance := outputAllowance(mutation.requestMap); allowance != nil {
		return protocolcodec.ReasoningEffortForBound(bound, *allowance)
	}
	return mutation.appliedEffort
}

// statedReasoningEffort is the effort the request carried into the boundary,
// which is the client's own: parseReasoningRequestMutation lifts it off the
// body before the arm's setting is applied, and defaults it only when the body
// had none. "none" is an off-signal, not a level, and a turn that reasons has
// already overruled it.
func (mutation *reasoningRequestMutation) statedReasoningEffort() string {
	if !mutation.hasOriginalEffort {
		return ""
	}
	var effort string
	if json.Unmarshal(mutation.originalReasoningEffort, &effort) != nil || effort == "none" {
		return ""
	}
	return effort
}

// reasoningBoundForRequest reads the bound the request already carries, or
// derives one from the output allowance, or reports that there is none.
func reasoningBoundForRequest(requestMap map[string]json.RawMessage) *int64 {
	if carried := carriedReasoningBound(requestMap["reasoning"]); carried != nil {
		return carried
	}
	allowance := outputAllowance(requestMap)
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

// appliedControl names the reasoning controls the request ends up carrying, so
// the mutation log says what was sent rather than what was configured. A bound
// travels with the effort level that bound buys, so the line names both.
func (mutation *reasoningRequestMutation) appliedControl() string {
	if mutation.reasoningBound != nil {
		return fmt.Sprintf("a bound of %d reasoning tokens at effort (%s)",
			*mutation.reasoningBound, mutation.appliedEffort)
	}
	return fmt.Sprintf("effort (%s)", mutation.appliedEffort)
}
