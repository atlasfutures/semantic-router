package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// selectionOwnsDispatch reports whether an authoritative selector owns this
// request's dispatch, in which case upstream-error fallback must not re-send
// it to another candidate. ARC commits its episode on the primary's response
// headers and aborts on a non-2xx; a fallback 200 from another arm would reach
// the client while the episode records nothing for the turn. Config
// validation refuses the combination too; this is the runtime guard for a
// fallback policy inherited by default.
func selectionOwnsDispatch(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	if ctx.SelectionTransaction != nil || ctx.RaylineARCTransaction != nil {
		return true
	}
	decision := ctx.VSRSelectedDecision
	return decision != nil && decision.Algorithm != nil &&
		decision.Algorithm.Type == config.RaylineARCAlgorithmType
}
