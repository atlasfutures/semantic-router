package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	seconds := r.Config.ResponseStreamFirstContentSec
	if ctx.StreamContentSeen {
		seconds = r.Config.ResponseStreamIdleSec
	}
	if seconds <= 0 {
		return 0
	}
	return time.Duration(seconds) * time.Second
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

// observeStreamContent restarts the silence clock on a content event.
func observeStreamContent(ctx *RequestContext, events []llmprotocol.Event) {
	if ctx == nil {
		return
	}
	for _, event := range events {
		if streamContentEvent(event.Type) {
			ctx.StreamContentSince = time.Now()
			ctx.StreamContentSeen = true
			return
		}
	}
}

func streamContentEvent(eventType llmprotocol.EventType) bool {
	switch eventType {
	case llmprotocol.EventOutputItemStarted, llmprotocol.EventOutputTextDelta,
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
)

// routerStreamCutCodes are the in-band errors of a stream the Router itself
// ended: each is a timeout.
var routerStreamCutCodes = map[string]bool{
	"stream_truncated":            true,
	streamFirstContentTimeoutCode: true,
	streamIdleTimeoutCode:         true,
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
