package extproc

import (
	"context"
	"errors"
	"io"
	"net/http"
	"runtime/debug"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// handleRequestBodyDispatch routes body messages to the correct handler.
//
// BUFFERED mode (default): the message goes straight to handleRequestBody.
//
// STREAMED mode (streamed_body_mode: true in config): Envoy sends multiple
// body messages. A StreamedBodyHandler accumulates chunks, detects the model
// from the first few KB, and either passes through or accumulates for the
// full pipeline on end_of_stream.
func (r *OpenAIRouter) handleRequestBodyDispatch(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Honor x-vsr-skip-processing before allocating a streamed-body handler.
	// This guarantees no chunk accumulation, model detection, or buffered
	// pipeline runs for opted-out requests, regardless of streamed_body_mode.
	// A route lookup is served by this router, not forwarded by it, so the
	// generic processing opt-out does not apply: honouring it here would hand
	// the caller's body to the upstream and execute a turn the endpoint
	// promises not to execute. The replay surface takes the same exception at
	// the header phase, for the same reason.
	if ctx.SkipProcessing && !isRaylineRoutesRequest(ctx) {
		if ctx.FullDuplexRequestBody {
			return newFullDuplexRequestBodyResponse(v.RequestBody.GetBody(), v.RequestBody.GetEndOfStream()), nil
		}
		return newContinueRequestBodyResponse(), nil
	}

	eos := v.RequestBody.GetEndOfStream()

	// If we already have a handler from a previous chunk, continue streaming
	if ctx.StreamedBody != nil {
		resp, err := ctx.StreamedBody.HandleChunk(v.RequestBody, ctx)
		if eos {
			ctx.StreamedBody.Release()
			ctx.StreamedBody = nil
		}
		return resp, err
	}

	// Decide mode based on config: only use streaming handler when explicitly enabled
	streamedMode := r.Config != nil && r.Config.StreamedBodyMode
	if ctx.FullDuplexRequestBody && !streamedMode {
		// A route lookup must never take this branch. It forwards the body
		// untouched to the upstream, and this endpoint's whole contract is
		// that it reaches no provider -- so falling through here would turn a
		// promise of no execution into a billed turn. Refusing is the only
		// safe answer: the lookup needs the whole body, and this mode hands
		// the router chunks it is not accumulating.
		if isRaylineRoutesRequest(ctx) {
			return r.createRaylineRoutesError(
				ctx,
				503,
				"api_error",
				"route lookup is unavailable while request bodies stream full duplex",
			), nil
		}
		return newFullDuplexRequestBodyResponse(v.RequestBody.GetBody(), eos), nil
	}
	if streamedMode && (!eos || ctx.FullDuplexRequestBody) {
		ctx.StreamedBody = newStreamedBodyHandler(r, ctx)
		resp, err := ctx.StreamedBody.HandleChunk(v.RequestBody, ctx)
		if eos {
			ctx.StreamedBody.Release()
			ctx.StreamedBody = nil
		}
		return resp, err
	}

	// BUFFERED mode or single-message STREAMED — use classic pipeline
	return r.handleRequestBody(v, ctx)
}

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	return r.processWithContext(stream, &RequestContext{
		Headers:      make(map[string]string),
		TraceContext: stream.Context(),
	})
}

// processWithContext runs one ext_proc stream over an already-built request
// context. Every exit from this loop is terminal for that context, which is
// why the authoritative selector's lifecycle is finalized here: EOF, a
// receive error, a cancel and a recovered panic all leave a prepared
// selection holding state that nothing else will release.
func (r *OpenAIRouter) processWithContext(
	stream ext_proc.ExternalProcessor_ProcessServer,
	ctx *RequestContext,
) (retErr error) {
	logging.Debugf("Processing at stage [init]")

	defer finalizeSelectionProcessTerminal(ctx)

	// Recover from any panic (including OOM kills surfaced as runtime panics from
	// CGO inference calls) so a single bad request cannot take down the gRPC server.
	defer func() {
		if rec := recover(); rec != nil {
			r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, "processor_panic")
			logging.Errorf("Process: recovered panic: %v\n%s", rec, debug.Stack())
			retErr = status.Errorf(codes.Internal, "internal error: %v", rec)
		}
	}()

	receiver := startStreamReceiver(stream)
	for {
		message, stalled := receiver.next(r.loopWait(ctx))
		if stalled {
			if waiting, ok := receiver.poll(); ok {
				message, stalled = waiting, false
			}
		}
		if stalled {
			if streamSilenceTimerArmed(ctx) && r.streamSilenceExceeded(ctx) {
				// Nothing arrived, not even a keepalive, and the streamed
				// turn's silence is past its limit.
				if err := r.endSilentStream(stream, ctx); err != nil {
					return err
				}
				continue
			}
			// Nothing arrived and the body being held is past its deadline.
			if err := r.endStalledResponseBody(stream, ctx); err != nil {
				return err
			}
			continue
		}
		if message.err != nil {
			return r.handleProcessReceiveError(ctx, message.err)
		}

		if err := r.handleProcessRequest(stream, message.request, ctx); err != nil {
			if sendCanceled(err) {
				// The client or the proxy went away while a chunk was being
				// sent: the stream ended as it does on a receive error, and
				// the chunk that failed to send delivered nothing.
				ctx.StreamEndedAtSend = true
				r.finalizeEndedStream(ctx, err)
			}
			// A request the router refused mid-flight never reaches the
			// response path that would release its inflight slot.
			releaseInflight(ctx)
			state, reason := replayLifecycleForProcessError(err)
			r.finalizeRouterReplay(ctx, state, reason)
			return err
		}
		if state := ctx.SemanticStreamState; state != nil {
			ctx.DeliveredStreamItems = len(state.items)
		}

	}
}

func (r *OpenAIRouter) handleProcessReceiveError(ctx *RequestContext, err error) error {
	r.finalizeEndedStream(ctx, err)
	releaseInflight(ctx)

	state, reason := replayLifecycleForReceiveError(err)
	r.finalizeRouterReplay(ctx, state, reason)

	if errors.Is(err, io.EOF) {
		logging.Debugf("Stream ended gracefully")
		return nil
	}

	if handled := handleProcessStatusError(ctx, err); handled {
		return nil
	}

	if handled := handleProcessContextError(ctx, err); handled {
		return nil
	}

	logging.Errorf("Error receiving request: %v", err)
	return err
}

func replayLifecycleForProcessError(err error) (string, string) {
	if errors.Is(err, context.Canceled) || status.Code(err) == codes.Canceled {
		return routerreplay.LifecycleAborted, "request_canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) || status.Code(err) == codes.DeadlineExceeded {
		return routerreplay.LifecycleAborted, "request_deadline_exceeded"
	}
	return routerreplay.LifecycleFailed, "request_processing_failed"
}

func replayLifecycleForReceiveError(err error) (string, string) {
	if errors.Is(err, io.EOF) {
		return routerreplay.LifecycleAborted, "stream_ended_before_terminal_response"
	}
	if errors.Is(err, context.Canceled) || status.Code(err) == codes.Canceled {
		return routerreplay.LifecycleAborted, "client_canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) || status.Code(err) == codes.DeadlineExceeded {
		return routerreplay.LifecycleAborted, "deadline_exceeded"
	}
	return routerreplay.LifecycleFailed, "extproc_receive_failed"
}

func handleProcessStatusError(ctx *RequestContext, err error) bool {
	s, ok := status.FromError(err)
	if !ok {
		return false
	}

	switch s.Code() {
	case codes.Canceled:
		return true
	case codes.DeadlineExceeded:
		recordProcessTimeout(ctx)
		return true
	default:
		return false
	}
}

func handleProcessContextError(ctx *RequestContext, err error) bool {
	if errors.Is(err, context.Canceled) {
		logging.Debugf("Stream canceled gracefully")
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		recordProcessTimeout(ctx)
		return true
	}
	return false
}

func recordProcessTimeout(ctx *RequestContext) {
	logging.Infof("Stream deadline exceeded")
	metrics.RecordRequestError(ctx.RequestModel, "timeout")
}

func (r *OpenAIRouter) handleProcessRequest(
	stream ext_proc.ExternalProcessor_ProcessServer,
	req *ext_proc.ProcessingRequest,
	ctx *RequestContext,
) error {
	if protocolConfig := req.GetProtocolConfig(); protocolConfig != nil {
		ctx.FullDuplexRequestBody = protocolConfig.GetRequestBodyMode() == http_ext.ProcessingMode_FULL_DUPLEX_STREAMED
		ctx.FullDuplexResponseBody = protocolConfig.GetResponseBodyMode() == http_ext.ProcessingMode_FULL_DUPLEX_STREAMED
	}

	switch v := req.Request.(type) {
	case *ext_proc.ProcessingRequest_RequestHeaders:
		return r.processRequestHeaders(stream, v, ctx)
	case *ext_proc.ProcessingRequest_RequestBody:
		return r.processRequestBody(stream, v, ctx)
	case *ext_proc.ProcessingRequest_ResponseHeaders:
		return r.processResponseHeaders(stream, v, ctx)
	case *ext_proc.ProcessingRequest_ResponseBody:
		return r.processResponseBody(stream, v, ctx)
	case *ext_proc.ProcessingRequest_RequestTrailers:
		return processRequestTrailers(stream, v)
	case *ext_proc.ProcessingRequest_ResponseTrailers:
		return r.processResponseTrailers(stream, v, ctx)
	default:
		return processUnknownRequest(stream, v)
	}
}

func (r *OpenAIRouter) processRequestHeaders(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) error {
	response, err := r.handleRequestHeaders(v, ctx)
	if err != nil {
		logging.Errorf("handleRequestHeaders failed: %v", err)
		return err
	}
	response = r.encodeImmediateResponseForClient(response, ctx)
	if err := sendResponse(stream, response, "request header"); err != nil {
		logging.Errorf("sendResponse for headers failed: %v", err)
		return err
	}
	return nil
}

func (r *OpenAIRouter) processRequestBody(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_RequestBody,
	ctx *RequestContext,
) error {
	response, err := r.handleRequestBodyDispatch(v, ctx)
	if err != nil {
		var ok bool
		if response, ok = r.processBodyRoutingError(err, ctx); !ok {
			logging.Errorf("handleRequestBody failed: %v", err)
			return err
		}
	}
	response = r.encodeImmediateResponseForClient(response, ctx)
	r.persistImmediateResponseObject(response, ctx)
	// FULL_DUPLEX_STREAMED explicitly permits the processor to buffer any
	// number of input chunks before sending a StreamedBodyResponse. A nil
	// response here means this chunk was retained for the eventual EOS reply.
	if response == nil && ctx.FullDuplexRequestBody {
		return nil
	}
	if err := sendResponse(stream, response, "request body"); err != nil {
		logging.Errorf("sendResponse for body failed: %v", err)
		return err
	}
	return nil
}

// processBodyRoutingError converts a *llmprotocol.ProtocolError raised during
// routing or dispatch into an immediate client-facing response. Capability
// mismatches (a request requiring capabilities the chosen backend wire cannot
// express, e.g. image output on chat completions) are client errors, not
// server failures; every other error keeps the caller's generic path.
func (r *OpenAIRouter) processBodyRoutingError(err error, ctx *RequestContext) (*ext_proc.ProcessingResponse, bool) {
	if err == nil {
		return nil, false
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		return nil, false
	}
	if ctx != nil {
		ctx.ImmediateProtocolError = protocolError
	}
	return r.createErrorResponse(http.StatusBadRequest, protocolError.Message), true
}

func (r *OpenAIRouter) processResponseHeaders(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) error {
	response, err := r.handleResponseHeaders(v, ctx)
	if err != nil {
		return err
	}
	return sendResponse(stream, response, "response header")
}

func (r *OpenAIRouter) processResponseBody(
	stream ext_proc.ExternalProcessor_ProcessServer,
	v *ext_proc.ProcessingRequest_ResponseBody,
	ctx *RequestContext,
) error {
	complete, guardBreach := r.completeResponseBody(v, ctx)
	if guardBreach != nil {
		return sendResponse(stream, r.responseBodyGuardResponse(ctx, guardBreach), "response body")
	}
	if complete == nil {
		return sendResponse(stream, heldResponseBodyChunk(), "response body")
	}
	response, err := r.handleResponseBody(complete, ctx)
	if err != nil {
		return err
	}
	response = r.normalizeFullDuplexResponseBody(response, ctx, complete.ResponseBody)
	if err := sendResponse(stream, response, "response body"); err != nil {
		return err
	}
	runPendingSelectionCompletion(ctx)
	return nil
}

func processUnknownRequest(
	stream ext_proc.ExternalProcessor_ProcessServer,
	request interface{},
) error {
	logging.Warnf("Unknown request type: %v", request)

	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	return sendResponse(stream, response, "unknown")
}

// finalizeEndedStream settles a streamed turn whose ext_proc exchange ended
// before the stream did, at a receive or at a send. This is the only place a
// stream the platform cut is ever seen ending. Returning without finalizing
// leaves the turn with no usage record: the upstream charged for every token
// it generated and the Router counted none of them.
func (r *OpenAIRouter) finalizeEndedStream(ctx *RequestContext, err error) {
	if !ctx.IsStreamingResponse || ctx.StreamingComplete || ctx.FailedCallSettled {
		return
	}
	ctx.StreamingAborted = true
	ctx.StreamEndedByReceiveError = true
	logging.Debugf("Streaming response aborted before completion, will not cache")
	// A codec error an earlier chunk raised is the stream's own failure; the
	// exchange ending after it does not replace it.
	streamErr := err
	if ctx.SemanticStreamErr != nil {
		streamErr = ctx.SemanticStreamErr
	}
	r.finalizeSemanticStreamingResponse(ctx, streamErr)
}

// sendCanceled reports a send that failed because the exchange was ended by
// the client or the proxy, not by the Router.
func sendCanceled(err error) bool {
	code := status.Code(err)
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, io.EOF) ||
		code == codes.Canceled || code == codes.DeadlineExceeded || code == codes.Unavailable
}

// releaseInflight returns the request's inflight slot, once; a request that
// ends without reaching the response path would otherwise hold its model's
// count up for good.
func releaseInflight(ctx *RequestContext) {
	if ctx == nil || ctx.InflightToken == 0 {
		return
	}
	inflight.End(ctx.InflightModel, ctx.InflightToken)
	ctx.InflightToken = 0
}
