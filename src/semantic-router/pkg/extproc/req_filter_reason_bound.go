package extproc

import (
	"encoding/json"
	"fmt"
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"

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
	transport modelcatalog.ReasoningTransport,
	client clientReasoningRequest,
	clientAllowance *int64,
) {
	if !usesReasoningObjectTransport(transport) || !mutation.reasoningApplied {
		return
	}
	bound := reasoningBoundForRequest(mutation.requestMap, client, clientAllowance)
	if bound == nil {
		return
	}
	mutation.requestMap["reasoning"] = encodeReasoningBound(mutation.requestMap["reasoning"], *bound)
	delete(mutation.requestMap, "reasoning_effort")
	mutation.appliedEffort = ""
	mutation.reasoningBound = bound
}

// clientReasoningRequest is what the client's body said about reasoning
// before the provider mutation rewrote it: the bound it carried, if any, and
// whether any control on it asked the model to reason at all.
type clientReasoningRequest struct {
	bound         *int64
	askedToReason bool
}

// snapshotClientReasoningRequest reads the client's reasoning controls off
// the request as it arrived. The catalog-driven mutation clears a carried
// max_tokens and moves the effort into the reasoning object, so anything the
// boundary wants to honour or to count has to be read before it runs. The
// top-level reasoning_effort is already gone from the map by then: the parser
// lifts it into the mutation before normalising, so it is read from there.
func snapshotClientReasoningRequest(mutation *reasoningRequestMutation) clientReasoningRequest {
	requestMap := mutation.requestMap
	snapshot := clientReasoningRequest{bound: carriedReasoningBound(requestMap["reasoning"])}
	snapshot.askedToReason = snapshot.bound != nil ||
		(mutation.hasOriginalEffort && reasoningEffortAsksToReason(mutation.originalReasoningEffort)) ||
		reasoningObjectAsksToReason(requestMap["reasoning"])
	return snapshot
}

// reasoningObjectAsksToReason reports whether the client's reasoning object
// itself asks to reason: an effort level that is not the off-signal, or an
// explicit enabled true.
func reasoningObjectAsksToReason(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var object struct {
		Effort  json.RawMessage `json:"effort"`
		Enabled *bool           `json:"enabled"`
	}
	if json.Unmarshal(raw, &object) != nil {
		return false
	}
	return reasoningEffortAsksToReason(object.Effort) || (object.Enabled != nil && *object.Enabled)
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
//
// Removing the client's request is not the same as saying no. On OpenRouter
// the arm's own off-signal takes its place: see offSignalForDisabledArm.
func dropReasoningRequestFromDisabledArm(
	mutation *reasoningRequestMutation,
	transport modelcatalog.ReasoningTransport,
	client clientReasoningRequest,
	ctx *RequestContext,
) {
	if usesReasoningObjectTransport(transport) {
		// The catalog mutation already wrote enabled false for a family that
		// can be disabled; a family that cannot is still routed to an arm that
		// must not reason, so the off-signal is stated here either way, on the
		// object the mutation left, keeping the presentation members it kept.
		delete(mutation.requestMap, "reasoning_effort")
		mutation.appliedEffort = ""
		mutation.requestMap["reasoning"] = offSignalForDisabledArm(mutation.requestMap["reasoning"])
	} else {
		// reasoning is OpenRouter's object; every other backend would see an
		// unknown member.
		delete(mutation.requestMap, "reasoning")
	}
	if client.askedToReason {
		recordDroppedReasoningRequest(ctx)
	}
}

// offSignalForDisabledArm is how a thinking-off arm says so to OpenRouter.
//
// chat_template_kwargs.enable_thinking false is what the Router sent before,
// and it is a vLLM chat-template argument: the providers OpenRouter picks for
// the ARC arms never see it. Measured against OpenRouter 2026-09-04 on
// deepseek-v4-pro, deepseek-v4-flash, mimo-v2.5-pro, qwen3.6-35b-a3b and
// glm-5.2, a body carrying only that flag reasoned on every arm -- the whole
// 64-token budget spent on reasoning, empty content, finish_reason "length".
// The same five with reasoning.enabled false returned reasoning_tokens 0 and
// content. The users of a thinking-off arm were paying the output rate for
// reasoning they never asked for and never saw. Read 2026-09-04:
//
//	https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
//
// The flag stays where it is. It is the control a vLLM-backed arm reads, and
// an OpenRouter provider that does not read it ignores it.
func offSignalForDisabledArm(existing json.RawMessage) json.RawMessage {
	object := map[string]json.RawMessage{}
	if len(existing) > 0 {
		if json.Unmarshal(existing, &object) != nil {
			object = map[string]json.RawMessage{}
		}
	}
	delete(object, "effort")
	delete(object, "max_tokens")
	object["enabled"] = json.RawMessage("false")
	rendered, err := json.Marshal(object)
	if err != nil {
		return json.RawMessage(`{"enabled":false}`)
	}
	return rendered
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
	client clientReasoningRequest,
	clientAllowance *int64,
) *int64 {
	if client.bound != nil {
		return client.bound
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
