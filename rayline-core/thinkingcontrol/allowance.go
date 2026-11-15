package thinkingcontrol

import "github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

// RaiseMessagesAllowance keeps the caller's output allowance on top of a
// thinking budget: Messages counts thinking inside max_tokens and refuses a
// budget that leaves no room below it.
func RaiseMessagesAllowance(request *llmprotocol.Request, budget *int64, defaultAllowance int64) bool {
	if budget == nil {
		return false
	}
	// The dispatch output bound has already set a limit on a Messages
	// request; the fallback only keeps a direct caller valid.
	allowance := defaultAllowance
	if request.Sampling.MaxOutputTokens != nil {
		allowance = *request.Sampling.MaxOutputTokens
	}
	if allowance > *budget {
		return false
	}
	raised := *budget + allowance
	request.Sampling.MaxOutputTokens = &raised
	return true
}

// ClampAllowance is the outcome of lowering an allowance to a deployment's
// limit.
type ClampAllowance struct {
	// From is the allowance the caller stated, before the clamp.
	From int64
	// To is the allowance the request is sent with.
	To int64
}

// ClampMessagesAllowance lowers an allowance above limit to limit, for a
// caller that cannot state a smaller one. It reports whether it clamped.
//
// The admitted control's thinking budget is read, never lowered: a lowered one
// would serve a control the policy did not choose.
func ClampMessagesAllowance(request *llmprotocol.Request, budget *int64, limit int64) (ClampAllowance, bool) {
	if request == nil || request.Sampling.MaxOutputTokens == nil || limit <= 0 {
		return ClampAllowance{}, false
	}
	stated := *request.Sampling.MaxOutputTokens
	if stated <= limit {
		return ClampAllowance{}, false
	}
	// Messages counts thinking inside max_tokens and Anthropic requires the
	// budget strictly below it, so a limit the budget reaches cannot be
	// clamped into: that turn keeps the deployment-limit refusal rather than
	// becoming a request the codec would reject.
	if budget != nil && *budget >= limit {
		return ClampAllowance{}, false
	}
	// Kept so a downstream bound is not derived from the limit as though the
	// caller had asked for it.
	if request.ClientMaxOutputTokens == nil {
		request.ClientMaxOutputTokens = &stated
	}
	clamped := ClampAllowance{From: stated, To: limit}
	request.Sampling.MaxOutputTokens = &clamped.To
	request.RouterSetMaxOutputTokens = true
	return clamped, true
}
