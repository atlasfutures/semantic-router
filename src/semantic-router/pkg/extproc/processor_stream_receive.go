package extproc

import (
	"io"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Acting on a deadline while nothing is arriving.
//
// The ext_proc loop is driven entirely by stream.Recv, so an upstream that
// stops mid-body is not a message the Router is ever sent, and a deadline read
// only when the next chunk arrives is a deadline that never fires. Under
// FULL_DUPLEX_STREAMED there is nothing above to cover it either: ext_proc's
// message_timeout does not apply in that mode, and the Router's own 590 s
// response deadline is measured only on a streamed turn. Envoy's
// stream_idle_timeout, 620 s on the cell, does eventually tear the stream
// down, so the held bytes are not kept forever -- but the client waits out the
// platform for a bare cut, which is the outcome decision 27 exists to remove.
//
// So the receive is moved off the loop goroutine and the loop selects on it
// against a timer. Only the receive moves: every Send stays on the loop
// goroutine, which is what keeps this free of the usual concurrent-writer
// problem on a gRPC stream.
type receivedMessage struct {
	request *ext_proc.ProcessingRequest
	err     error
	// panicValue carries a panic raised inside Recv -- a CGO OOM surfaces
	// that way -- back to the loop goroutine, where the recover in
	// processWithContext still turns it into the same gRPC Internal error it
	// always did. Recovering it on the reader instead would swallow it and
	// end the turn as a clean EOF.
	panicValue any
	panicked   bool
}

// receiveOnce isolates one Recv so a panic inside it can be carried rather
// than killing the reader.
func receiveOnce(stream ext_proc.ExternalProcessor_ProcessServer) (message receivedMessage) {
	defer func() {
		if recovered := recover(); recovered != nil {
			message = receivedMessage{panicValue: recovered, panicked: true}
		}
	}()
	request, err := stream.Recv()
	return receivedMessage{request: request, err: err}
}

// repanicOnLoopGoroutine re-raises on the caller what Recv raised on the
// reader, so the panic is handled exactly where it was before.
func repanicOnLoopGoroutine(message receivedMessage) receivedMessage {
	if message.panicked {
		panic(message.panicValue)
	}
	return message
}

type streamReceiver struct {
	messages chan receivedMessage
}

// startStreamReceiver reads the stream until it fails. The buffer of one lets
// the reader stay a message ahead of the loop; the context guard is what stops
// it blocking on a send after the loop has gone.
func startStreamReceiver(stream ext_proc.ExternalProcessor_ProcessServer) *streamReceiver {
	receiver := &streamReceiver{messages: make(chan receivedMessage, 1)}
	goSafely("extproc_stream_receiver", func() {
		defer close(receiver.messages)
		for {
			message := receiveOnce(stream)
			select {
			case receiver.messages <- message:
			case <-stream.Context().Done():
				return
			}
			if message.err != nil || message.panicked {
				return
			}
		}
	})
	return receiver
}

// next returns the next message, or reports that the wait ran out first. A
// timeout of zero waits for as long as it takes, which is what a router with
// no accumulation deadline configured does.
func (receiver *streamReceiver) next(timeout time.Duration) (receivedMessage, bool) {
	if timeout <= 0 {
		return receiver.receive(), false
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case message, ok := <-receiver.messages:
		return repanicOnLoopGoroutine(closedAsEOF(message, ok)), false
	case <-timer.C:
		return receivedMessage{}, true
	}
}

func (receiver *streamReceiver) receive() receivedMessage {
	message, ok := <-receiver.messages
	return repanicOnLoopGoroutine(closedAsEOF(message, ok))
}

// A closed channel is the reader having given up on a stream that is gone.
// EOF is what the loop already knows how to end on.
func closedAsEOF(message receivedMessage, ok bool) receivedMessage {
	if !ok {
		return receivedMessage{err: io.EOF}
	}
	return message
}

// heldResponseBodyWait is how long the loop may wait before the bytes it is
// holding are past their deadline. Zero means it may wait indefinitely: no
// deadline is configured, or nothing is being held.
func (r *OpenAIRouter) heldResponseBodyWait(ctx *RequestContext) time.Duration {
	if r == nil || r.Config == nil || ctx == nil ||
		len(ctx.ResponseBodyChunks) == 0 || ctx.ResponseBodyHeldSince.IsZero() {
		return 0
	}
	timeout := time.Duration(r.Config.ResponseBodyTimeoutSec) * time.Second
	if timeout <= 0 {
		return 0
	}
	remaining := timeout - time.Since(ctx.ResponseBodyHeldSince)
	if remaining <= 0 {
		// Already past it. Wake immediately rather than not at all.
		return time.Millisecond
	}
	return remaining
}

// endStalledResponseBody refuses the turn the upstream stopped serving. The
// held bytes are released and the body is marked ended, so the chunks and the
// trailer that may still arrive add nothing to a response that is over.
func (r *OpenAIRouter) endStalledResponseBody(
	stream ext_proc.ExternalProcessor_ProcessServer,
	ctx *RequestContext,
) error {
	logging.ComponentWarnEvent("extproc", "response_body_stalled", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      ctx.RequestModel,
		"held_bytes": len(ctx.ResponseBodyChunks),
		"held_ms":    time.Since(ctx.ResponseBodyHeldSince).Milliseconds(),
	})
	ctx.ResponseBodyChunks = nil
	ctx.ResponseBodyEnded = true
	protocolError := llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout,
		"response_body_timeout",
		"the model service stopped sending its response body",
		nil,
	)
	return sendResponse(stream, r.responseBodyGuardResponse(ctx, protocolError), "response body")
}
