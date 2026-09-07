package extproc

import (
	"bytes"
	"encoding/json"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The one shape a response-body reply may take under FULL_DUPLEX_STREAMED.
//
// Envoy's processor_state.cc refuses a plain body mutation in that mode and
// refuses a streamed_response in every other one, so the two are not
// interchangeable and the Router has to send whichever the data plane
// declared. A CommonResponse carrying no mutation is worse than either: in
// full duplex the reply *is* the response body rather than a mutation of it,
// so a reply that mutates nothing forwards nothing, and the chunk is dropped.
//
// Four response-body paths sent one of those illegal shapes:
//
//   - req_filter_skip_processing.go, a skipped turn: bare CONTINUE
//   - processor_res_body.go, the looper capture: bare CONTINUE
//   - processor_res_transport_error.go, a translated upstream failure: plain
//     body mutation, via setResponseBodyMutation
//   - processor_res_body_pipeline.go, every rewritten non-streaming response:
//     plain body mutation, via setResponseBodyMutation
//
// Rather than teach each of them the mode, the shape is fixed once where the
// reply leaves: processResponseBody. That keeps the mode out of the paths that
// have nothing to do with it, and it holds for any reply added later.
//
// The semantic streaming path is left alone. It already builds its own
// streamed_response through responseStreamBodyMutation, and it is the only
// path that knows whether the turn it is ending is over -- an end-of-stream
// invented here would end a response mid-turn.
//
// A body-phase ImmediateResponse is the fifth shape, and it is legal only
// while Envoy still owns the response headers. Under BUFFERED it does: a
// decode failure, a jailbreak refusal, a failed context recovery and the rest
// reach the client as the 502 or 403 they say they are. Under full duplex the
// headers have already gone downstream, where sendLocalReply on a started
// response resets the stream -- so the refusal would become a truncated 200
// carrying nothing.
func (r *OpenAIRouter) normalizeFullDuplexResponseBody(
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
	chunk *ext_proc.HttpBody,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.FullDuplexResponseBody || response == nil {
		return response
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return r.endResponseWithImmediateBody(ctx, immediate)
	}
	bodyResponse := response.GetResponseBody()
	if bodyResponse == nil {
		// A reply in another phase, which this does not shape.
		return response
	}
	if bodyResponse.Response == nil {
		bodyResponse.Response = &ext_proc.CommonResponse{Status: ext_proc.CommonResponse_CONTINUE}
	}
	common := bodyResponse.Response
	if common.GetBodyMutation().GetStreamedResponse() != nil {
		return response
	}

	// Nothing was rewritten, so what travels is what arrived. Outside full
	// duplex Envoy would have forwarded it without being told to.
	body := chunk.GetBody()
	if mutation := common.GetBodyMutation(); mutation != nil {
		switch mutation := mutation.GetMutation().(type) {
		case *ext_proc.BodyMutation_Body:
			body = mutation.Body
		case *ext_proc.BodyMutation_ClearBody:
			if mutation.ClearBody {
				body = nil
			}
		}
	}
	common.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_StreamedResponse{
			StreamedResponse: &ext_proc.StreamedBodyResponse{
				Body: bytes.Clone(body),
				// Envoy said whether this was the last chunk. Ending the
				// response any earlier would truncate it, and any later leaves
				// the client waiting on the platform.
				EndOfStream: chunk.GetEndOfStream(),
			},
		},
	}
	return response
}

// endResponseWithImmediateBody converts a body-phase refusal into the last
// body of the response.
//
// The status is already spent, which is what "the headers have gone
// downstream" means: the client will read 200 whatever this says. What can
// still be preserved is the refusal itself and the end. The header mutation
// is dropped because Envoy scrubbed those headers at encodeHeaders, long
// before this.
//
// The body cannot travel as it was built. Every body-phase producer builds
// its ImmediateResponse through createErrorResponse, which is always
// OpenAI-shaped; on the request and header phases
// encodeImmediateResponseForClient re-encodes that for the client, and this
// seam had no such step. Forwarded verbatim, and with the status already
// downgraded to 200, an Anthropic client would parse a foreign error object
// as content rather than as a failure. So it is re-encoded here into the
// client's own wire format.
//
// The turn is over, so the body is marked ended and the trailer that may
// follow neither flushes nor ends anything a second time.
func (r *OpenAIRouter) endResponseWithImmediateBody(
	ctx *RequestContext,
	immediate *ext_proc.ImmediateResponse,
) *ext_proc.ProcessingResponse {
	logging.ComponentWarnEvent("extproc", "body_phase_refusal_downgraded", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      ctx.RequestModel,
		// What the client would have read under BUFFERED, and will not read
		// here. The body still says it; the status line cannot.
		"refused_status": immediate.GetStatus().GetCode(),
	})
	ctx.ResponseBodyEnded = true
	if ctx.IsStreamingResponse {
		// A streamed turn would otherwise be ended a second time by the
		// trailer. The turn is accounted by whichever refusal built this
		// reply, not by the streaming finalizer.
		ctx.StreamingComplete = true
	}
	return buildResponseBodyContinueResponse(
		responseStreamBodyMutation(ctx, r.clientEncodedRefusal(ctx, immediate), true), nil)
}

// clientEncodedRefusal re-encodes a body-phase refusal for the client.
//
// The typed error is preferred where the producer left one on the context;
// otherwise it is rebuilt from what the refusal itself says, its HTTP status
// naming the category and its OpenAI-shaped body the message. If the client
// speaks OpenAI, or if anything about the re-encode fails, the original body
// travels: it is the same bytes the client would have read under BUFFERED.
func (r *OpenAIRouter) clientEncodedRefusal(
	ctx *RequestContext,
	immediate *ext_proc.ImmediateResponse,
) []byte {
	body := bytes.Clone(immediate.GetBody())
	_, target := responseWireFormats(ctx)
	if target == llmprotocol.OpenAIChatV1 {
		return body
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return body
	}
	encoded, err := engine.EncodeError(target, refusalProtocolError(ctx, immediate))
	if err != nil {
		logging.ComponentErrorEvent("extproc", "body_phase_refusal_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     target,
			"error":      err.Error(),
		})
		return body
	}
	return encoded
}

func refusalProtocolError(
	ctx *RequestContext,
	immediate *ext_proc.ImmediateResponse,
) *llmprotocol.ProtocolError {
	if ctx.ImmediateProtocolError != nil {
		return ctx.ImmediateProtocolError
	}
	status := int(immediate.GetStatus().GetCode())
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	message := "the router refused this response"
	if err := json.Unmarshal(immediate.GetBody(), &envelope); err == nil &&
		envelope.Error.Message != "" {
		message = envelope.Error.Message
	}
	return llmprotocol.NewError(refusalCategory(status), "response_refused", message, nil)
}

func refusalCategory(status int) llmprotocol.ErrorCategory {
	switch {
	case status == 403:
		return llmprotocol.ErrorPermission
	case status == 408 || status == 504:
		return llmprotocol.ErrorUpstreamTimeout
	case status == 429:
		return llmprotocol.ErrorRateLimited
	case status >= 500:
		return llmprotocol.ErrorUpstreamUnavailable
	default:
		return llmprotocol.ErrorInvalidRequest
	}
}
