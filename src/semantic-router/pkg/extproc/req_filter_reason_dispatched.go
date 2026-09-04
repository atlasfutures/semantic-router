package extproc

import "encoding/json"

// What the request actually asks the upstream to do about reasoning.
//
// routing_decision recomputed the arm's configured effort and reported it as
// the request's, so on build 16 it named a dial the wire did not carry. The
// controls are read back off the rendered body instead, at the provider
// boundary and after every mutation, so the record cannot drift from the
// request: it is the same bytes.
//
// Two places carry an effort. OpenRouter and the other backends that take a
// top-level field carry reasoning_effort; a vLLM-compatible backend takes it
// through chat_template_kwargs instead. reasoning.max_tokens is OpenRouter's
// alone. A body that carries none of them records none, which is the honest
// report for a turn that was told nothing about reasoning.
func recordDispatchedReasoningControls(ctx *RequestContext, body []byte) {
	if ctx == nil {
		return
	}
	ctx.DispatchedReasoningEffort = ""
	ctx.DispatchedReasoningBound = nil

	var wire struct {
		ReasoningEffort    string `json:"reasoning_effort"`
		ChatTemplateKwargs struct {
			ReasoningEffort string `json:"reasoning_effort"`
		} `json:"chat_template_kwargs"`
		Reasoning struct {
			MaxTokens *int64 `json:"max_tokens"`
		} `json:"reasoning"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return
	}
	ctx.DispatchedReasoningEffort = wire.ReasoningEffort
	if ctx.DispatchedReasoningEffort == "" {
		ctx.DispatchedReasoningEffort = wire.ChatTemplateKwargs.ReasoningEffort
	}
	ctx.DispatchedReasoningBound = wire.Reasoning.MaxTokens
}
