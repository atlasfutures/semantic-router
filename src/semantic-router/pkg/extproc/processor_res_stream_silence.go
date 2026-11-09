package extproc

import (
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// A streamed turn's silence (semantic-router #210).
//
// An arm can accept a turn, send its 200 and then send nothing but keepalive
// comments: kimi-k3 on the dev cell 2026-10-06 sat 73 to 313 s before its
// first content. The stream deadline only ends such a turn at 590 s, and
// until then the turn holds its episode lease, so the client's retry is
// refused as session_busy, and nothing tells the fallback the arm is stalled.
//
// The first-content and idle deadlines end it sooner. The wait is measured
// from the upstream's response headers to the first content event, and from
// each content event to the next. Keepalives, provider-opaque frames and the
// response-started event are not content. A turn ended before any content
// reached the client is a timeout with nothing sent: its arm is excluded from
// the cell's next decisions, and the episode lease is released at once, so
// the client's retry is decided afresh on another arm.

// streamSilenceLimit is how long the turn may now go without content. Zero
// means no limit applies.
func (r *OpenAIRouter) streamSilenceLimit(ctx *RequestContext) time.Duration {
	if r == nil || r.Config == nil || ctx == nil {
		return 0
	}
	// An arm may set its own first-content and idle limits.
	seconds := r.Config.FirstContentSecForModel(ctx.RequestModel)
	if ctx.StreamContentSeen {
		seconds = r.Config.IdleSecForModel(ctx.RequestModel)
	}
	if seconds <= 0 {
		return 0
	}
	limit := time.Duration(seconds) * time.Second
	if left, capped := r.hostRetryBudgetLeft(ctx); capped && left < limit {
		// A re-send waits for its first content only as long as the turn's
		// host-retry budget has left.
		limit = max(left, time.Millisecond)
	}
	return limit
}

// hostRetryBudgetLeft is how long a re-send in flight, from its own start,
// may wait for its first content before the turn's host-retry budget
// (response_stream.host_retry_budget_sec) is spent; capped is false when no
// budget applies.
func (r *OpenAIRouter) hostRetryBudgetLeft(ctx *RequestContext) (time.Duration, bool) {
	budget := r.Config.ResponseStreamHostRetryBudgetSec
	if budget <= 0 || ctx.StreamContentSeen || !ctx.HostRetry.started() || ctx.StartTime.IsZero() ||
		refusalReserveServing(ctx) {
		return 0, false
	}
	return ctx.StartTime.Add(time.Duration(budget) * time.Second).Sub(ctx.StreamContentSince), true
}

// streamSilenceWait is how long the loop may wait for a message before the
// turn's silence is past its limit. Zero means it may wait indefinitely.
func (r *OpenAIRouter) streamSilenceWait(ctx *RequestContext) time.Duration {
	if !streamSilenceWatched(ctx) {
		return 0
	}
	limit := r.streamSilenceLimit(ctx)
	if limit <= 0 {
		return 0
	}
	remaining := limit - time.Since(ctx.StreamContentSince)
	if remaining <= 0 {
		return time.Millisecond
	}
	return remaining
}

// streamSilenceExceeded reports that the turn has gone without content for
// longer than it may.
func (r *OpenAIRouter) streamSilenceExceeded(ctx *RequestContext) bool {
	if !streamSilenceWatched(ctx) {
		return false
	}
	limit := r.streamSilenceLimit(ctx)
	return limit > 0 && time.Since(ctx.StreamContentSince) >= limit
}

func streamSilenceWatched(ctx *RequestContext) bool {
	return ctx != nil && ctx.IsStreamingResponse && !ctx.StreamingComplete && !ctx.StreamContentSince.IsZero()
}

// streamSilenceTimerArmed reports that the loop's own timer may end the turn.
// That sends a body reply with no body message to answer, which only a
// full-duplex exchange accepts, and only once Envoy is streaming the body: a
// reply before the first body message would arrive while the body is idle.
func streamSilenceTimerArmed(ctx *RequestContext) bool {
	return ctx != nil && ctx.FullDuplexResponseBody && ctx.ProtocolResponseStream != nil
}

// observeStreamContent restarts the silence clock on a content event.
func observeStreamContent(ctx *RequestContext, events []llmprotocol.Event) {
	if ctx == nil {
		return
	}
	for _, event := range events {
		if streamContentEvent(event.Type) && !bareToolCallStart(event) {
			ctx.StreamContentSince = time.Now()
			if !ctx.StreamContentSeen {
				ctx.StreamFirstContentAt = ctx.StreamContentSince
			}
			ctx.StreamContentSeen = true
			return
		}
	}
}

// bareToolCallStart is a tool-call delta that carries no arguments yet: a
// Chat stream names the call (id and name) in its first delta, which is an
// item's start, not content.
func bareToolCallStart(event llmprotocol.Event) bool {
	return event.Type == llmprotocol.EventToolCallDelta && (event.ToolCall == nil || event.ToolCall.Arguments == "")
}

func streamContentEvent(eventType llmprotocol.EventType) bool {
	switch eventType {
	// An item's start is not content: a Responses reasoning model announces
	// its reasoning item at once and then thinks in silence, and a Messages
	// content_block_start carries no payload. The first delta is the content.
	case llmprotocol.EventOutputTextDelta,
		llmprotocol.EventReasoningDelta, llmprotocol.EventToolCallDelta,
		llmprotocol.EventImageGenerationProgress, llmprotocol.EventOutputItemCompleted,
		llmprotocol.EventResponseCompleted, llmprotocol.EventResponseFailed:
		return true
	}
	return false
}

// silentStreamError is what the client is told, and what classes the turn a
// timeout.
func silentStreamError(ctx *RequestContext) *llmprotocol.ProtocolError {
	if ctx != nil && ctx.HostRetry != nil && ctx.HostRetry.failedClass != "" {
		return reservedArmFailedError(ctx.HostRetry.failedClass)
	}
	if ctx != nil && ctx.StreamContentSeen {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, streamIdleTimeoutCode,
			"the model stopped sending content, so the router ended this stream", nil)
	}
	return llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, streamFirstContentTimeoutCode,
		"the model sent no content in time, so the router ended this stream", nil)
}

const (
	streamFirstContentTimeoutCode = "stream_first_content_timeout"
	streamIdleTimeoutCode         = "stream_idle_timeout"
	// reservedArmFailedCode is the in-band error of a refusal re-serve's arm
	// that failed before any content: no host retry follows it, and the
	// client is told how it failed rather than a timeout.
	reservedArmFailedCode = "reserved_model_failed"
)

// reservedArmFailedError is the error a refusal re-serve's arm that failed
// before any content ends the stream with, in the category of its class.
func reservedArmFailedError(class string) *llmprotocol.ProtocolError {
	category := llmprotocol.ErrorUpstreamUnavailable
	switch class {
	case turnFailureRateLimited:
		category = llmprotocol.ErrorRateLimited
	case turnFailureTimeout:
		category = llmprotocol.ErrorUpstreamTimeout
	}
	return llmprotocol.NewError(category, reservedArmFailedCode,
		"the model the turn was re-served on failed before any content ("+class+")", nil)
}

// routerStreamCutCodes are the in-band errors of a stream the Router itself
// ended: each is a timeout.
var routerStreamCutCodes = map[string]bool{
	"stream_truncated":            true,
	streamFirstContentTimeoutCode: true,
	streamIdleTimeoutCode:         true,
}

// routerStreamCutCauses is what a client is told about each cut the Router
// makes, in the vocabulary the codec renders (llmprotocol.Truncation*). The
// codes are the Router's own, so the map lives here beside them rather than
// in the codec: the codec knows the three words and nothing about why this
// host cuts a stream. Every code in routerStreamCutCodes has an entry, and so
// does the body cut (standard_resend_body_cut), which is not a timeout and is
// classed apart. A cut added without an entry here reaches the client as a
// bare length stop, which is why a test holds the two in step.
//
// "deadline" rather than "timeout" even though the category is a timeout: a
// timeout on this path already means something else -- an upstream timeout,
// the route timer, a socket ETIMEDOUT -- and a caller reading it infers the
// model stopped responding, which on a deadline cut it had not.
var routerStreamCutCauses = map[string]string{
	"stream_truncated":            llmprotocol.TruncationDeadline,
	streamFirstContentTimeoutCode: llmprotocol.TruncationSilence,
	streamIdleTimeoutCode:         llmprotocol.TruncationSilence,
	"standard_resend_body_cut":    llmprotocol.TruncationTransport,
}

// routerStreamCutCause names the cut an in-band error is, or nothing for an
// error that is not one of the Router's own.
func routerStreamCutCause(cut *llmprotocol.ProtocolError) string {
	if cut == nil {
		return ""
	}
	return routerStreamCutCauses[cut.Code]
}

func (r *OpenAIRouter) logResponseStreamSilence(ctx *RequestContext) {
	logging.ComponentWarnEvent("extproc", "response_stream_silent", map[string]interface{}{
		"request_id":    ctx.RequestID,
		"model":         ctx.RequestModel,
		"selected":      ctx.VSRSelectedModel,
		"content_seen":  ctx.StreamContentSeen,
		"limit_ms":      r.streamSilenceLimit(ctx).Milliseconds(),
		"silent_ms":     time.Since(ctx.StreamContentSince).Milliseconds(),
		"elapsed_ms":    time.Since(ctx.StartTime).Milliseconds(),
		"response_ends": ctx.FullDuplexResponseBody,
	})
}

// releaseCutTurn ends the selection of a turn the Router cut. A cut turn
// never commits, and its lease would otherwise be renewed until the loop
// exits, which is when the upstream or the platform finally closes the
// exchange; the client's retry would be refused as session_busy until then.
func releaseCutTurn(ctx *RequestContext) {
	ensureSelectionTransactionBound(ctx)
	if ctx == nil || ctx.SelectionTransaction == nil || ctx.SelectionTransaction.isCommitted() {
		return
	}
	finalizeSelectionAbort(ctx, processAbortClass(ctx))
}
