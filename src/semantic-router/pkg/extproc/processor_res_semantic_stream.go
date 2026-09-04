package extproc

import (
	"fmt"
	"sort"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
)

// semanticResponseStreamState is request-scoped reconstruction state for
// telemetry, replay, memory, and cache policy. It consumes neutral events from
// the same codec contract as buffered responses and never inspects client SSE JSON.
type semanticResponseStreamState struct {
	responseID string
	model      string
	stop       llmprotocol.StopReason
	usage      llmprotocol.Usage
	items      map[int]*semanticStreamItem
	order      []int
	terminal   bool
	failed     *llmprotocol.ProtocolError
}

type semanticStreamItem struct {
	id           string
	role         llmprotocol.Role
	text         string
	refusal      string
	reasoning    string
	reasoningSig string
	toolCall     *llmprotocol.ToolCall
	completed    bool
}

func (r *OpenAIRouter) handleSemanticStreamingResponseBody(
	responseBody []byte,
	endOfStream bool,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	// A turn the Router has already ended is over for the client. Whatever the
	// upstream keeps sending would arrive on a message it has been told is
	// finished, so none of it travels.
	if ctx != nil && ctx.StreamingComplete {
		// The response has already been ended, so this chunk ends nothing: a
		// second end-of-stream would be a second terminal on a finished
		// response.
		return buildResponseBodyContinueResponse(responseStreamBodyMutation(ctx, nil, false), nil)
	}
	recordStreamingTTFT(ctx)
	r.initializeSemanticResponseStream(ctx)
	buffers := semanticStreamBuffers{}
	buffers.push(responseBody, ctx)
	overran := r.responseStreamOverran(ctx)
	if overran || endOfStream {
		if overran {
			r.logResponseStreamTruncation(ctx)
			buffers.recordError(ctx, r.truncatedStreamError(), true)
		}
		buffers.finalize(ctx)
		r.finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
	}
	// The turn is over either way here: the upstream reached its end, or the
	// Router ended it. Both have to end the response, or the client waits for
	// an EOF that only the platform will send.
	return buffers.processingResponse(ctx, overran || endOfStream)
}

func (r *OpenAIRouter) initializeSemanticResponseStream(ctx *RequestContext) {
	if err := r.ensureSemanticResponseStream(ctx); err != nil {
		ctx.StreamingAborted = true
		logging.ComponentErrorEvent("extproc", "neutral_stream_init_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
	}
}

type semanticStreamBuffers struct {
	translated []byte
	public     []byte
	// upstream is what arrived, kept only for the full-duplex reply, where a
	// chunk nothing rewrote still has to be handed back rather than left to
	// pass through.
	upstream  []byte
	streamErr error
}

func (buffers *semanticStreamBuffers) push(responseBody []byte, ctx *RequestContext) {
	if len(responseBody) == 0 {
		return
	}
	if ctx.FullDuplexResponseBody {
		buffers.upstream = append(buffers.upstream, responseBody...)
	}
	if ctx.PublicChatUsageFilter != nil {
		filtered, err := ctx.PublicChatUsageFilter.Push(responseBody)
		buffers.public = append(buffers.public, filtered...)
		buffers.recordError(ctx, err, true)
	}
	if ctx.ProtocolResponseStream != nil {
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Push(responseBody)
		buffers.translated = appendProtocolFrames(buffers.translated, frames)
		observeProtocolStream(ctx, events, diagnostics)
		buffers.recordError(ctx, err, true)
	}
}

func (buffers *semanticStreamBuffers) finalize(ctx *RequestContext) {
	if ctx.ProtocolResponseStream != nil {
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Finalize(buffers.streamErr)
		buffers.translated = appendProtocolFrames(buffers.translated, frames)
		observeProtocolStream(ctx, events, diagnostics)
		buffers.recordError(ctx, err, true)
	}
	if ctx.PublicChatUsageFilter != nil {
		filtered, err := ctx.PublicChatUsageFilter.Finalize()
		buffers.public = append(buffers.public, filtered...)
		buffers.recordError(ctx, err, false)
	}
}

func (buffers *semanticStreamBuffers) recordError(ctx *RequestContext, err error, overwrite bool) {
	if err == nil || !overwrite && buffers.streamErr != nil {
		return
	}
	buffers.streamErr = err
	ctx.StreamingAborted = true
}

func appendProtocolFrames(body []byte, frames [][]byte) []byte {
	for _, frame := range frames {
		body = append(body, frame...)
	}
	return body
}

func observeProtocolStream(
	ctx *RequestContext,
	events []llmprotocol.Event,
	diagnostics []llmprotocol.Diagnostic,
) {
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	ctx.SemanticStreamState.observe(events)
}

func (buffers *semanticStreamBuffers) processingResponse(
	ctx *RequestContext,
	endOfStream bool,
) *ext_proc.ProcessingResponse {
	if ctx.ProtocolResponseStream != nil &&
		(requiresClientResponseRewrite(ctx) || ctx.StreamingAborted) {
		return buildResponseBodyContinueResponse(
			responseStreamBodyMutation(ctx, buffers.translated, endOfStream), nil)
	}
	if ctx.PublicChatUsageFilter != nil {
		return buildResponseBodyContinueResponse(
			responseStreamBodyMutation(ctx, buffers.public, endOfStream), nil)
	}
	// Nothing was rewritten, so the upstream chunk travels as it is. In the
	// full-duplex mode it still has to be handed back, because there the reply
	// is the response body rather than a mutation of it.
	if ctx.FullDuplexResponseBody {
		return buildResponseBodyContinueResponse(
			responseStreamBodyMutation(ctx, buffers.upstream, endOfStream), nil)
	}
	return buildResponseBodyContinueResponse(nil, nil)
}

func recordStreamingTTFT(ctx *RequestContext) {
	if ctx == nil || ctx.TTFTRecorded || ctx.ProcessingStartTime.IsZero() || ctx.RequestModel == "" {
		return
	}

	ttft := time.Since(ctx.ProcessingStartTime).Seconds()
	if ttft <= 0 {
		return
	}

	metrics.RecordModelTTFT(ctx.RequestModel, ttft)
	ctx.TTFTSeconds = ttft
	ctx.TTFTRecorded = true
	latency.UpdateTTFT(ctx.RequestModel, ttft)
	ctx.CacheWarmthEstimate = latency.EstimateCacheProbability(latency.CacheEstimationInput{
		Model:       ctx.RequestModel,
		TTFTSeconds: ttft,
	})
	maybeEmitTransitionEvent(ctx)
	logging.Debugf("Recorded TTFT on first streamed body chunk: model=%q, TTFT=%.4fs", ctx.RequestModel, ttft)
}

func (r *OpenAIRouter) ensureSemanticResponseStream(ctx *RequestContext) error {
	if ctx == nil {
		return fmt.Errorf("request context is unavailable")
	}
	if ctx.ProtocolResponseStream != nil {
		return nil
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return err
	}
	source, target := responseWireFormats(ctx)
	streamContext := llmprotocol.StreamContext{
		Context: ctx.TraceContext, Source: source, Target: target,
		Options:     clientStreamOptions(ctx),
		PublicModel: ctx.RequestModel, PreviousResponseID: responseObjectPreviousID(ctx),
	}
	var mutation protocolcodec.StreamEventMutation
	if responseID := responseObjectPublicID(ctx); responseID != "" {
		streamContext.ResponseID = responseID
		mutation = func(event *llmprotocol.Event) error {
			event.ResponseID = responseID
			return nil
		}
	}
	stream, err := engine.NewStreamWithMutation(source, target, streamContext, mutation)
	if err != nil {
		return err
	}
	ctx.ProtocolResponseStream = stream
	if source == llmprotocol.OpenAIChatV1 && target == llmprotocol.OpenAIChatV1 && !streamUsageRequestedByClient(ctx) {
		ctx.PublicChatUsageFilter = protocolcodec.NewChatUsageStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	}
	ctx.SemanticStreamState = &semanticResponseStreamState{
		usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		items: make(map[int]*semanticStreamItem),
	}
	return nil
}

func streamUsageRequestedByClient(ctx *RequestContext) bool {
	options := clientStreamOptions(ctx)
	return options.IncludeUsage != nil && *options.IncludeUsage
}

//nolint:gocognit,cyclop,funlen // Observation is an exhaustive reducer over the closed neutral event contract.
func (state *semanticResponseStreamState) observe(events []llmprotocol.Event) {
	if state == nil {
		return
	}
	for _, event := range events {
		if event.ResponseID != "" {
			state.responseID = event.ResponseID
		}
		if event.Model != "" {
			state.model = event.Model
		}
		if event.Usage != nil {
			state.usage = *event.Usage
		}
		if event.StopReason != "" {
			state.stop = event.StopReason
		}
		switch event.Type {
		case llmprotocol.EventOutputItemStarted:
			item := state.item(event.ItemIndex)
			item.id = event.ItemID
			item.role = event.Role
			if event.ToolCall != nil {
				call := *event.ToolCall
				item.toolCall = &call
			}
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
				item.reasoningSig = event.Content.Signature
			}
		case llmprotocol.EventOutputTextDelta:
			item := state.item(event.ItemIndex)
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
				item.refusal += event.Delta
			} else {
				item.text += event.Delta
			}
		case llmprotocol.EventReasoningDelta:
			item := state.item(event.ItemIndex)
			item.reasoning += event.Delta
			if event.Content != nil && event.Content.Signature != "" {
				item.reasoningSig = event.Content.Signature
			}
		case llmprotocol.EventToolCallDelta:
			item := state.item(event.ItemIndex)
			if item.toolCall == nil {
				item.toolCall = &llmprotocol.ToolCall{}
			}
			if event.ToolCall != nil {
				if event.ToolCall.ID != "" {
					item.toolCall.ID = event.ToolCall.ID
				}
				if event.ToolCall.Name != "" {
					item.toolCall.Name = event.ToolCall.Name
				}
				item.toolCall.Arguments += event.ToolCall.Arguments
			}
		case llmprotocol.EventOutputItemCompleted:
			item := state.item(event.ItemIndex)
			item.completed = true
			if event.ToolCall != nil {
				call := *event.ToolCall
				item.toolCall = &call
			}
		case llmprotocol.EventResponseCompleted:
			state.terminal = true
		case llmprotocol.EventResponseFailed:
			state.terminal = true
			state.failed = event.Error
		}
	}
}

func (state *semanticResponseStreamState) item(index int) *semanticStreamItem {
	item := state.items[index]
	if item != nil {
		return item
	}
	item = &semanticStreamItem{role: llmprotocol.RoleAssistant}
	state.items[index] = item
	state.order = append(state.order, index)
	return item
}

//nolint:cyclop // Response assembly validates every terminal stream invariant in one place.
func (state *semanticResponseStreamState) response() (*llmprotocol.Response, error) {
	if state == nil || state.failed != nil || !state.terminal {
		return nil, fmt.Errorf("semantic stream did not complete successfully")
	}
	indices := append([]int(nil), state.order...)
	sort.SliceStable(indices, func(left, right int) bool { return indices[left] < indices[right] })
	output := make([]llmprotocol.OutputItem, 0, len(indices))
	for _, index := range indices {
		item := state.items[index]
		if item == nil || !item.completed {
			return nil, fmt.Errorf("semantic stream output item is incomplete")
		}
		contents := make([]llmprotocol.Content, 0, 4)
		if item.reasoning != "" || item.reasoningSig != "" {
			contents = append(contents, llmprotocol.Content{
				Kind: llmprotocol.ContentReasoning, Text: item.reasoning,
				Signature: item.reasoningSig,
			})
		}
		if item.refusal != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: item.refusal})
		}
		if item.text != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: item.text})
		}
		if item.toolCall != nil {
			call := *item.toolCall
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &call})
		}
		if len(contents) == 0 {
			// A model that answers with only a stop token completes its turn
			// like any other. Refusing here bills the turn, reports it
			// successful, and leaves settlement holding no response at all.
			//
			// The client is served no content block for such a turn -- an
			// Anthropic message whose content is empty is how that outcome is
			// said on the wire. The neutral contract cannot say it the same
			// way: ValidateResponse refuses an output item that names no
			// content (empty_output_item), so the record kept for settlement
			// holds an empty text block instead. The two differ on purpose.
			logging.ComponentWarnEvent("extproc", "upstream_completion_was_empty", map[string]interface{}{
				"item_index": index,
			})
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentText})
		}
		itemID := item.id
		if itemID == "" {
			itemID = fmt.Sprintf("item_%d", index)
		}
		output = append(output, llmprotocol.OutputItem{ID: itemID, Role: item.role, Content: contents})
	}
	if state.stop == "" {
		state.stop = llmprotocol.StopUnknown
	}
	return &llmprotocol.Response{
		Generation: 1, ID: state.responseID, CreatedAt: time.Now().UTC(),
		Model: state.model, Output: output, StopReason: state.stop, Usage: state.usage,
	}, nil
}

func (r *OpenAIRouter) finalizeSemanticStreamingResponse(ctx *RequestContext, streamErr error) {
	if ctx == nil || ctx.StreamingComplete {
		return
	}
	ctx.StreamingComplete = true
	if streamErr != nil || ctx.SemanticStreamState == nil || !ctx.SemanticStreamState.terminal {
		ctx.StreamingAborted = true
	}
	semanticResponse, responseErr := ctx.SemanticStreamState.response()
	if responseErr == nil {
		ctx.SemanticResponse = semanticResponse
	} else {
		ctx.StreamingAborted = true
	}
	completionLatency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		completionLatency = time.Since(ctx.StartTime)
		if ctx.RequestModel != "" {
			metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency.Seconds())
		}
	}
	inflight.End(ctx.RequestModel, ctx.InflightToken)
	ctx.InflightToken = 0

	usage := truncatedStreamUsage(ctx, r.takeNeutralResponseUsage(ctx))
	r.reportSemanticStreamingUsage(ctx, completionLatency, usage)
	r.calibrateTokenEstimator(ctx, usage.promptTokens)

	if responseErr != nil {
		logging.ComponentWarnEvent("extproc", "neutral_stream_reconstruction_skipped", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      responseErr.Error(),
		})
		return
	}
	encoded, err := r.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "neutral_stream_replay_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return
	}
	r.updateResponseCache(ctx, encoded)
	r.scheduleSemanticResponseMemoryStore(ctx, semanticResponse)
	r.persistResponseObject(ctx)
	r.attachRouterReplayResponse(ctx, encoded, true)
}

// truncatedStreamUsage settles what a turn the platform cut actually cost.
//
// A stream that never reached its terminal event has no reconstructed response
// to read usage from, but the counts the upstream did send before the cut are
// still authoritative, and they are what the turn cost. When none arrived
// there is nothing to put in an llm_usage line, so the turn is named as
// uncounted instead: an estimate there would be an invented number about
// money.
func truncatedStreamUsage(ctx *RequestContext, usage responseUsageMetrics) responseUsageMetrics {
	if !usage.invalid || !ctx.StreamingAborted {
		return usage
	}
	if ctx.SemanticStreamState != nil {
		usage = responseUsageFromSemanticUsage(ctx.SemanticStreamState.usage)
	}
	if usage.invalid {
		logging.ComponentWarnEvent("extproc", "stream_truncated_uncounted", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      ctx.RequestModel,
			"reason":     usage.invalidReason,
		})
	}
	return usage
}

func (r *OpenAIRouter) reportSemanticStreamingUsage(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) {
	if ctx == nil || usage.invalid {
		return
	}
	totalTokens := responseUsageTotal(usage)
	if r.RateLimiter != nil && ctx.RateLimitCtx != nil && totalTokens > 0 {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:  usage.promptTokens,
			OutputTokens: usage.completionTokens,
			TotalTokens:  totalTokens,
		})
	}
	if totalTokens > 0 {
		recordSessionTurn(ctx, usage, r.sessionTurnPricing(ctx.RequestModel))
	}
	if ctx.RequestModel == "" {
		return
	}
	recordModelUsageTokens(ctx.RequestModel, usage)
	metrics.RecordModelWindowedRequest(
		ctx.RequestModel,
		completionLatency.Seconds(),
		int64(usage.promptTokens),
		int64(usage.completionTokens),
		false,
		false,
	)
	if usage.completionTokens > 0 && completionLatency > 0 {
		timePerToken := completionLatency.Seconds() / float64(usage.completionTokens)
		metrics.RecordModelTPOT(ctx.RequestModel, timePerToken)
		latency.UpdateTPOT(ctx.RequestModel, timePerToken)
	}
	replayUsage := r.recordResponseCost(ctx, completionLatency, usage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	r.observeRouterLearningUsageTelemetry(ctx, completionLatency, usage, replayUsage)
}
