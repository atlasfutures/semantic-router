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
func applyOpenRouterReasoningBound(mutation *reasoningRequestMutation, dialect openAIBackendDialect) {
	if !dialect.usesReasoningObjectBound() || !mutation.reasoningApplied {
		return
	}
	bound := reasoningBoundForRequest(mutation.requestMap)
	if bound == nil {
		return
	}
	mutation.requestMap["reasoning"] = encodeReasoningBound(mutation.requestMap["reasoning"], *bound)
	delete(mutation.requestMap, "reasoning_effort")
	mutation.appliedEffort = ""
	mutation.reasoningBound = bound
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

// appliedControl names the reasoning control the request ends up carrying, so
// the mutation log says what was sent rather than what was configured. Exactly
// one control travels, so the line names one.
func (mutation *reasoningRequestMutation) appliedControl() string {
	if mutation.reasoningBound != nil {
		return fmt.Sprintf("a bound of %d reasoning tokens", *mutation.reasoningBound)
	}
	return fmt.Sprintf("effort (%s)", mutation.appliedEffort)
}
