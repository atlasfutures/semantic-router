package extproc

import (
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// A streamed turn's re-send after its upstream stream ended early, inside the
// first completion tokens (providers.models[].head_retry).
//
// An upstream stream sometimes ends without its terminal event: the codec
// closes the turn with stream_incomplete, and the client is left holding a
// broken partial answer. Most of these cuts come within the first few tokens.
// So an arm with the window on holds its stream's frames from the client
// until head_retry_tokens completion tokens have arrived, exactly as host
// retry holds them until the first content (stream_host_retry.go). When the
// stream ends early while still held, the Router re-sends the dispatched body
// once, unchanged, and the new call's stream becomes the rest of the response
// the client already has open. Once the window is passed, or the turn ends
// cleanly inside it, what was held is sent and the turn streams as before.
//
// The re-send rides the host-retry plan and its splice. It is the turn's one
// re-send: after it, no host retry follows, and the re-send's own frames are
// not held. A re-send that fails before any content ends the turn with
// stream_head_retry_failed; one that is cut after content ends it as any cut
// does.
//
// The cut call writes its own usage line (head_retry: head_attempt), settled
// as the upstream_5xx it was with nothing sent; the re-send writes the turn's
// line (head_retry: head_retry).

const (
	headRetryAttempt = "head_attempt"
	headRetryServed  = "head_retry"

	// streamHeadRetryFailedCode is the in-band error of a turn whose head
	// re-send failed before any content.
	streamHeadRetryFailedCode = "stream_head_retry_failed"
)

// streamHeadTokens is how many completion tokens the call in flight has
// produced: the larger of the count the stream stated and the Router's own
// estimate of what it decoded.
func streamHeadTokens(state *semanticResponseStreamState) int {
	if state == nil {
		return 0
	}
	tokens := estimatedTokensForBytes(generatedTextBytes(state))
	if stated := state.usage.OutputTotal.Value; stated != nil && int(*stated) > tokens {
		tokens = int(*stated)
	}
	return tokens
}

// headRetryEligible reports that the call in flight is the turn's first, its
// frames are all still held, and the head re-send has not been made. A
// refusal re-serve, an endpoint re-send or a host retry under way owns the
// turn's hold instead.
func headRetryEligible(ctx *RequestContext) bool {
	plan := ctx.HostRetry
	return plan != nil && plan.headTokens > 0 && !plan.headRetried && plan.endpoint == nil &&
		plan.splice == nil && len(plan.attempts) == 1 && !plan.exposed && ctx.RefusalReserve == nil &&
		ctx.FullDuplexResponseBody && ctx.IsStreamingResponse && !ctx.StreamingComplete
}

// holdingStreamHead reports that the call in flight's frames are held for
// the head re-send: it has produced fewer completion tokens than the window.
func (r *OpenAIRouter) holdingStreamHead(ctx *RequestContext) bool {
	return headRetryEligible(ctx) && streamHeadTokens(ctx.SemanticStreamState) < ctx.HostRetry.headTokens
}

// streamEndedIncomplete reports that the codec finalized the stream as
// stream_incomplete: it ended without its terminal event, and nothing the
// upstream said or the Router cut ended it first. The codec's own failure
// event is observed as the stream's failure (upstreamFailed), so a stream
// the provider failed carries the provider's code there instead.
func streamEndedIncomplete(ctx *RequestContext) bool {
	state := ctx.SemanticStreamState
	return ctx.UpstreamBodyCut == nil && state != nil && state.upstreamFailed != nil &&
		state.upstreamFailed.Code == "stream_incomplete"
}

// retryStreamHead starts the head re-send of a turn whose upstream stream
// ended early while its frames were held, or reports that there is none: the
// turn then ends as it would without it.
func (r *OpenAIRouter) retryStreamHead(ctx *RequestContext) bool {
	if !r.holdingStreamHead(ctx) || !streamEndedIncomplete(ctx) {
		return false
	}
	plan := ctx.HostRetry
	buffered := streamHeadTokens(ctx.SemanticStreamState)
	fields := map[string]interface{}{
		"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel,
		"buffered_tokens": buffered, "window_tokens": plan.headTokens,
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	if r.responseStreamOverran(ctx) {
		fields["reason"] = "deadline_passed"
		logging.ComponentWarnEvent("extproc", "stream_head_retry_unavailable", fields)
		return false
	}
	call, unavailable := r.buildResendCall(ctx, plan.dispatch.logicalModel, plan.dispatch, plan.dispatched)
	if call == nil {
		fields["reason"] = unavailable
		logging.ComponentWarnEvent("extproc", "stream_head_retry_unavailable", fields)
		return false
	}
	logging.ComponentEvent("extproc", "stream_head_retry", fields)

	plan.headBuffered = buffered
	r.settleReplacedCall(ctx, turnFailureUpstream5xx, "stream_incomplete", headAttemptUsage(ctx), func(record *llmUsageRecord) {
		record.HeadRetry = nonEmpty(headRetryAttempt)
		record.HeadRetryBufferedTokens = &buffered
	})
	plan.headRetried = true
	plan.held, plan.exposed = nil, false
	resetStreamForHostRetry(ctx, plan.dispatched, plan.dispatch)
	// The cut call's content never reached the client: the re-send's first
	// content is the turn's.
	ctx.StreamContentSeen, ctx.StreamFirstContentAt = false, time.Time{}
	r.initializeSemanticResponseStream(ctx)
	plan.splice = startHostRetrySplice(ctx, call, 0)
	return true
}

// headAttemptUsage is what the cut call is settled at: the counts it stated,
// with its output taken from the Router's own estimate of what it decoded
// when that is larger, as a cut turn's line is.
func headAttemptUsage(ctx *RequestContext) responseUsageMetrics {
	usage := partialStreamUsage(ctx.SemanticStreamState.usage)
	estimated, carried := estimatedTruncatedStreamUsage(ctx)
	if !carried || usage.completionTokensReported && estimated.completionTokens <= usage.completionTokens {
		return usage
	}
	estimated.providerCost = usage.providerCost
	return estimated
}

// headRetryFailedError is the in-band error of a turn whose head re-send
// failed before any content, classed by how the re-send failed.
func headRetryFailedError(class string) *llmprotocol.ProtocolError {
	category := llmprotocol.ErrorUpstreamUnavailable
	switch class {
	case turnFailureRateLimited:
		category = llmprotocol.ErrorRateLimited
	case turnFailureTimeout:
		category = llmprotocol.ErrorUpstreamTimeout
	}
	return llmprotocol.NewError(category, streamHeadRetryFailedCode,
		"the model's stream ended early and its re-send failed before any content ("+class+")", nil)
}

// addHeadRetryFields writes a turn's head-retry window on its usage line and,
// once the re-send was made, marks the line of the call that served.
func addHeadRetryFields(record *llmUsageRecord, ctx *RequestContext) {
	plan := ctx.HostRetry
	if plan == nil || plan.headTokens == 0 {
		return
	}
	window := plan.headTokens
	record.HeadRetryTokens = &window
	if plan.headRetried {
		buffered := plan.headBuffered
		record.HeadRetry = nonEmpty(headRetryServed)
		record.HeadRetryBufferedTokens = &buffered
	}
}

// addHeadRetryTurnFields writes the same on a turn_failed line.
func addHeadRetryTurnFields(fields map[string]interface{}, ctx *RequestContext) {
	plan := ctx.HostRetry
	if plan == nil || plan.headTokens == 0 {
		return
	}
	fields["head_retry_tokens"] = plan.headTokens
	if plan.headRetried {
		fields["head_retry"] = headRetryServed
		fields["head_retry_buffered_tokens"] = plan.headBuffered
	}
}
