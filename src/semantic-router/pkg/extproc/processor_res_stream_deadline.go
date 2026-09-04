package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The platform's timeout ladder above one routed turn, and the Router's own
// deadline at the bottom of it.
//
// A turn that reaches the top of the ladder is ended by the platform, and the
// Router never gets to speak: by the time the cut arrives the ext_proc stream
// is gone and the frames that would close the message honestly have nowhere to
// go. That is what three thinking-arm turns on the dev cell 2026-09-03 looked
// like from the client -- message_start, content_block_start, deltas, then a
// clean close, with curl exiting 0.
//
// So the Router ends the turn first, while the connection to the client is
// still open. 590 s leaves 20 s below the lowest rung for the closing frames
// to travel.
//
//	Cloud Run request timeout    630 s
//	Envoy stream_idle_timeout    620 s
//	ext_proc message_timeout     610 s
//	this deadline                590 s
//
// The cell's Envoy configuration is authoritative for the three rungs above;
// extProcMessageTimeoutSeconds mirrors it so the relationship can be asserted
// here rather than only remembered.
const (
	defaultResponseStreamDeadlineSeconds = 590
	extProcMessageTimeoutSeconds         = 610
)

// responseStreamDeadline is how long this turn may stream before the Router
// ends it. Zero means the deadline is off.
func (r *OpenAIRouter) responseStreamDeadline() time.Duration {
	seconds := defaultResponseStreamDeadlineSeconds
	if r != nil && r.Config != nil && r.Config.ResponseStreamDeadlineSec != 0 {
		seconds = r.Config.ResponseStreamDeadlineSec
	}
	if seconds < 0 {
		return 0
	}
	return time.Duration(seconds) * time.Second
}

// responseStreamOverran reports that this turn has streamed for longer than
// the Router allows one to. It is measured from the start of the request,
// because that is what every rung of the platform ladder above it measures.
func (r *OpenAIRouter) responseStreamOverran(ctx *RequestContext) bool {
	if ctx == nil || ctx.StartTime.IsZero() || ctx.StreamingComplete {
		return false
	}
	deadline := r.responseStreamDeadline()
	return deadline > 0 && time.Since(ctx.StartTime) >= deadline
}

// truncatedStreamError is what the client is told. Neither vendor defines a
// stop reason for a turn ended by infrastructure, so the failure travels as an
// error frame; the category is a timeout because that is what it is.
func (r *OpenAIRouter) truncatedStreamError() *llmprotocol.ProtocolError {
	return llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout,
		"stream_truncated",
		"the router ended this stream at its own deadline, below the platform's",
		nil,
	)
}

func (r *OpenAIRouter) logResponseStreamTruncation(ctx *RequestContext) {
	logging.ComponentWarnEvent("extproc", "response_stream_deadline_exceeded", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       ctx.RequestModel,
		"deadline_ms": r.responseStreamDeadline().Milliseconds(),
		"elapsed_ms":  time.Since(ctx.StartTime).Milliseconds(),
		// Whether the frames were the end of the turn or only the end of the
		// message. False means this connection is still the platform's to cut,
		// tens of seconds later, and the client is still waiting.
		"response_ended": ctx.FullDuplexResponseBody,
	})
}
