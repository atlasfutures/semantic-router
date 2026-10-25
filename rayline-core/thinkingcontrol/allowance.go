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
