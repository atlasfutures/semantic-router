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
// Three places carry an effort. OpenRouter and the other backends that take a
// top-level field carry reasoning_effort; OpenRouter's reasoning object can
// carry it as reasoning.effort; a vLLM-compatible backend takes it through
// chat_template_kwargs instead. reasoning.max_tokens is OpenRouter's alone. A body that carries none of them records none, which is the honest
// report for a turn that was told nothing about reasoning.
func recordDispatchedReasoningControls(ctx *RequestContext, body []byte) {
	if ctx == nil {
		return
	}
	ctx.DispatchedReasoningEffort = ""
	ctx.DispatchedReasoningBound = nil

	var wire dispatchedReasoningWire
	if json.Unmarshal(body, &wire) != nil {
		return
	}
	wire.record(ctx)
}

// recordDispatchedControls is recordDispatchedReasoningControls and
// recordDispatchedProviderPin over one decode of the final body, which is the
// whole transcript: each read-back otherwise scans every message to find a
// few top-level members. A body the joint decode refuses -- a member of one
// group of the wrong type -- is read by each read-back on its own, so one
// group's error never clears the other's record.
func recordDispatchedControls(ctx *RequestContext, body []byte) {
	if ctx == nil {
		return
	}
	var wire struct {
		dispatchedReasoningWire
		dispatchedProviderWire
	}
	if json.Unmarshal(body, &wire) != nil {
		recordDispatchedReasoningControls(ctx, body)
		recordDispatchedProviderPin(ctx, body)
		return
	}
	ctx.DispatchedReasoningEffort = ""
	ctx.DispatchedReasoningBound = nil
	ctx.DispatchedProviderOrder = nil
	ctx.DispatchedProviderPrivacy = dispatchedProviderPrivacy{}
	wire.dispatchedReasoningWire.record(ctx)
	wire.dispatchedProviderWire.record(ctx)
}

// dispatchedReasoningWire is the body's reasoning controls.
type dispatchedReasoningWire struct {
	ReasoningEffort    string `json:"reasoning_effort"`
	ChatTemplateKwargs struct {
		ReasoningEffort string `json:"reasoning_effort"`
	} `json:"chat_template_kwargs"`
	Reasoning struct {
		Effort    string `json:"effort"`
		MaxTokens *int64 `json:"max_tokens"`
	} `json:"reasoning"`
	// Messages carries its effort in output_config (pathfinder #2655).
	OutputConfig struct {
		Effort string `json:"effort"`
	} `json:"output_config"`
}

func (wire *dispatchedReasoningWire) record(ctx *RequestContext) {
	ctx.DispatchedReasoningEffort = wire.ReasoningEffort
	if ctx.DispatchedReasoningEffort == "" {
		ctx.DispatchedReasoningEffort = wire.Reasoning.Effort
	}
	if ctx.DispatchedReasoningEffort == "" {
		ctx.DispatchedReasoningEffort = wire.ChatTemplateKwargs.ReasoningEffort
	}
	if ctx.DispatchedReasoningEffort == "" {
		ctx.DispatchedReasoningEffort = wire.OutputConfig.Effort
	}
	// The record publishes it: a value outside the effort enum is a digest.
	ctx.DispatchedReasoningEffort = knownOrDigest(ctx.DispatchedReasoningEffort, knownEfforts)
	ctx.DispatchedReasoningBound = wire.Reasoning.MaxTokens
}
