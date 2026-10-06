package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func isUpstreamTransportError(ctx *RequestContext) bool {
	return ctx != nil && ctx.UpstreamStatusCode != 0 &&
		(ctx.UpstreamStatusCode < 200 || ctx.UpstreamStatusCode >= 300)
}

// handleUpstreamTransportError is the response-body boundary for HTTP failures.
// A non-2xx body is never decoded as a model response: it is translated through
// the neutral transport-error contract while Envoy preserves the upstream HTTP
// status observed during the response-header phase.
func (r *OpenAIRouter) handleUpstreamTransportError(
	body []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	engine, err := r.protocolEngine()
	if err != nil {
		return r.createErrorResponse(503, "protocol runtime unavailable")
	}
	source, target := responseWireFormats(ctx)
	// The failure is classed by what the provider said; the client is told
	// only its public form.
	var upstreamError *llmprotocol.ProtocolError
	translated, err := engine.TranslateTransportError(source, target, body, func(transportError *llmprotocol.TransportError) error {
		upstreamError = transportError.Error
		transportError.Error = publicUpstreamError(ctx, transportError.Error, ctx.UpstreamStatusCode, "transport")
		return nil
	})
	if err != nil {
		metrics.RecordRequestError(ctx.RequestModel, "invalid_upstream_error")
		logging.ComponentErrorEvent("extproc", "neutral_transport_error_decode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     source,
			"status":     ctx.UpstreamStatusCode,
			"error":      err.Error(),
		})
		protocolError := upstreamTransportFallback(ctx.UpstreamStatusCode, err)
		encoded, encodeErr := engine.EncodeError(target, protocolError)
		if encodeErr != nil {
			return r.createErrorResponse(502, "The selected model returned an invalid error response")
		}
		translated.Body = encoded
	}
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, translated.Diagnostics...)
	if upstreamError == nil {
		upstreamError = translated.TransportError.Error
	}
	r.recordUpstreamErrorTurn(ctx, upstreamError)
	response := buildResponseBodyContinueResponse(nil, nil)
	setResponseBodyMutation(response, translated.Body)
	setResponseContentType(response, "application/json")
	r.attachRouterReplayResponse(ctx, translated.Body, true)
	return response
}

func responseWireFormats(ctx *RequestContext) (llmprotocol.WireFormat, llmprotocol.WireFormat) {
	source, target := llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
	if ctx == nil {
		return source, target
	}
	if ctx.TargetFormat != "" {
		source = ctx.TargetFormat
	} else if ctx.SourceFormat != "" {
		source = ctx.SourceFormat
	}
	if ctx.SourceFormat != "" {
		target = ctx.SourceFormat
	}
	return source, target
}

func upstreamTransportFallback(status int, cause error) *llmprotocol.ProtocolError {
	category, code := llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_error"
	if status == 429 {
		category, code = llmprotocol.ErrorRateLimited, "rate_limited"
	} else if status == 408 || status == 504 {
		category, code = llmprotocol.ErrorUpstreamTimeout, "upstream_timeout"
	}
	return llmprotocol.NewError(category, code, "model service returned an invalid error response", cause)
}

// recordUpstreamErrorTurn classes a provider error and settles its usage
// through the response-usage owner, as any other call is. A failed call used
// to leave no usage line at all, so the usage stream could not be joined
// one-to-one with the gateway's rows, which record errors. The provider
// stated no usage, so the line's counts are null, its usage_source unknown
// and its pricing no_usage.
func (r *OpenAIRouter) recordUpstreamErrorTurn(ctx *RequestContext, protocolError *llmprotocol.ProtocolError) {
	if ctx == nil {
		return
	}
	recordTurnFailure(ctx, upstreamFailureClass(ctx.UpstreamStatusCode, protocolError), false)
	r.reportFailedCallUsage(ctx)
}

// reportFailedCallUsage settles a call that failed without stated usage.
func (r *OpenAIRouter) reportFailedCallUsage(ctx *RequestContext) {
	latency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		latency = time.Since(ctx.StartTime)
	}
	r.reportNonStreamingUsage(ctx, latency, responseUsageMetrics{})
	// The failed call is settled: its usage line is written, so the exchange
	// ending after it finalizes nothing more.
	ctx.FailedCallSettled = true
}
