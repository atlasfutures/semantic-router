package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type anthropicStreamDecoder struct {
	streamState
	framer       sseFramer
	stopSequence string
}
type anthropicStreamEncoder struct {
	streamState
	blocks         map[anthropicBlockKey]llmprotocol.ContentKind
	blockIndexes   map[anthropicBlockKey]int
	blockStarted   map[anthropicBlockKey]bool
	blockStopped   map[anthropicBlockKey]bool
	itemBlockKeys  map[int][]anthropicBlockKey
	blockResumes   map[streamContentKey]int
	activeBlock    anthropicBlockKey
	hasActiveBlock bool
	nextBlockIndex int
	// truncationUsage is what a turn the Router ended cost, set by the caller
	// before it ends the stream. Left nil for every other failure: a provider
	// failure is the provider's to describe.
	truncationUsage  *llmprotocol.Usage
	truncationSource string
}

func (encoder *anthropicStreamEncoder) SetTruncationUsage(usage *llmprotocol.Usage, source string) {
	encoder.truncationUsage = usage
	encoder.truncationSource = source
}

func (AnthropicMessagesCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &anthropicStreamDecoder{streamState: streamState{context: context, policy: policy}, framer: newSSEFramer(policy.Limits.SSEFrameBytes)}
}

func (AnthropicMessagesCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &anthropicStreamEncoder{
		streamState:   streamState{context: context, policy: policy},
		blocks:        make(map[anthropicBlockKey]llmprotocol.ContentKind),
		blockIndexes:  make(map[anthropicBlockKey]int),
		blockStarted:  make(map[anthropicBlockKey]bool),
		blockStopped:  make(map[anthropicBlockKey]bool),
		itemBlockKeys: make(map[int][]anthropicBlockKey),
		blockResumes:  make(map[streamContentKey]int),
	}
}

type anthropicEventWire struct {
	Type         string                          `json:"type"`
	Message      *anthropicResponseWire          `json:"message,omitempty"`
	Index        *int                            `json:"index,omitempty"`
	ContentBlock *anthropicContentWire           `json:"content_block,omitempty"`
	Delta        *anthropicDeltaWire             `json:"delta,omitempty"`
	Usage        *anthropicMessageDeltaUsageWire `json:"usage,omitempty"`
	Error        *anthropicErrorWire             `json:"error,omitempty"`
}

// anthropicTruncationFrameWire is the error frame for a turn the Router ended
// itself. It is the vendor error event plus what the turn cost, because
// nothing downstream can bill a cut turn otherwise.
//
// It is its own shape rather than two more members on anthropicEventWire: the
// event wire is the published Messages contract, checked field by field
// against a provider fixture, and these two members are the Router's, written
// and never read. The count sits beside the error object rather than inside
// it, so an SDK parsing the failure sees exactly the shape it always saw.
type anthropicTruncationFrameWire struct {
	Type        string                          `json:"type"`
	Error       *anthropicErrorWire             `json:"error,omitempty"`
	Usage       *anthropicMessageDeltaUsageWire `json:"usage,omitempty"`
	UsageSource string                          `json:"usage_source,omitempty"`
}

type anthropicMessageDeltaUsageWire struct {
	CacheCreationInputTokens int64                           `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64                           `json:"cache_read_input_tokens"`
	InputTokens              int64                           `json:"input_tokens"`
	OutputTokens             int64                           `json:"output_tokens"`
	OutputTokensDetails      anthropicOutputUsageDetailsWire `json:"output_tokens_details"`
	ServerToolUse            anthropicServerToolUsageWire    `json:"server_tool_use"`
}

type anthropicDeltaWire struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	Thinking     string          `json:"thinking,omitempty"`
	PartialJSON  string          `json:"partial_json,omitempty"`
	Signature    string          `json:"signature,omitempty"`
	StopReason   *string         `json:"stop_reason,omitempty"`
	StopSequence *string         `json:"stop_sequence,omitempty"`
	Container    json.RawMessage `json:"container,omitempty"`
	StopDetails  json.RawMessage `json:"stop_details,omitempty"`
	Citation     json.RawMessage `json:"citation,omitempty"`
}

func (wire anthropicDeltaWire) MarshalJSON() ([]byte, error) {
	if wire.Type == "message_delta" {
		return json.Marshal(struct {
			Container    json.RawMessage `json:"container,omitempty"`
			StopDetails  json.RawMessage `json:"stop_details,omitempty"`
			StopReason   *string         `json:"stop_reason"`
			StopSequence *string         `json:"stop_sequence"`
		}{
			Container: wire.Container, StopDetails: wire.StopDetails,
			StopReason: wire.StopReason, StopSequence: wire.StopSequence,
		})
	}
	type deltaAlias anthropicDeltaWire
	return json.Marshal(deltaAlias(wire))
}

func (decoder *anthropicStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if err := decoder.observeProviderStreamBytes(chunk); err != nil {
		return nil, nil, err
	}
	frames, err := decoder.framer.Push(chunk)
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := decoder.pushFrame(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, decoder.policy.Limits.Diagnostics)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func (decoder *anthropicStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := decoder.parseProviderSSEFrame(frame)
	if err != nil || !parsed.HasData {
		return nil, nil, err
	}
	if decoder.terminal {
		return nil, nil, invalidProviderResponse("stream_event_after_terminal", "Anthropic stream emitted data after message_stop")
	}
	eventType, err := decodeProviderEventType(parsed.Data, parsed.Event, decoder.policy)
	if err != nil {
		return nil, nil, err
	}
	if !isSupportedAnthropicEvent(eventType) {
		return decoder.decodeUnknownAnthropicEvent(frame)
	}
	return decoder.decodeAnthropicWireFrame(parsed.Data, eventType, frame)
}

func (decoder *anthropicStreamDecoder) decodeAnthropicWireFrame(
	data []byte,
	eventType string,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	var wire anthropicEventWire
	if err := decodeProviderWire(data, &wire, decoder.policy); err != nil {
		return nil, nil, err
	}
	if wire.Type == "" {
		wire.Type = eventType
	}
	if err := validateAnthropicStreamEvent(wire, data); err != nil {
		return nil, nil, err
	}
	if wire.Type == "content_block_start" && anthropicEventIndex(wire) != len(decoder.items) {
		return nil, nil, invalidProviderResponse("stream_output_index_order", "Anthropic content block indexes must be contiguous from zero")
	}
	if wire.Message != nil {
		if err := decoder.observeProviderIdentity(wire.Message.ID, wire.Message.Model); err != nil {
			return nil, nil, err
		}
	}
	return decoder.decodeEvent(wire, frame)
}

func isSupportedAnthropicEvent(eventType string) bool {
	switch eventType {
	case "message_start", "message_delta", "message_stop",
		"content_block_start", "content_block_delta", "content_block_stop",
		"error", "ping":
		return true
	default:
		return false
	}
}

func (decoder *anthropicStreamDecoder) decodeEvent(
	wire anthropicEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	switch wire.Type {
	case "message_start":
		return decoder.emitAnthropicEvent(decodeAnthropicMessageStart(wire))
	case "content_block_start":
		return decoder.emitDecodedAnthropicEvent(decodeAnthropicContentStart(wire))
	case "content_block_delta":
		return decoder.emitDecodedAnthropicEvent(decodeAnthropicContentDelta(wire))
	case "content_block_stop":
		return decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: anthropicEventIndex(wire)})
	case "message_delta":
		return decoder.decodeAnthropicMessageDelta(wire)
	case "message_stop":
		return decoder.decodeAnthropicMessageStop()
	case "error":
		return decoder.emitAnthropicEvent(decodeAnthropicStreamError(wire))
	case "ping":
		return nil, nil, nil
	default:
		return decoder.decodeUnknownAnthropicEvent(frame)
	}
}

func (decoder *anthropicStreamDecoder) emitDecodedAnthropicEvent(
	event llmprotocol.Event,
	err error,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if err != nil {
		return nil, nil, err
	}
	return decoder.emitAnthropicEvent(event)
}

func (decoder *anthropicStreamDecoder) decodeAnthropicMessageStop() ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if decoder.stop == "" {
		return nil, nil, invalidProviderResponse("stream_stop_reason_missing", "Anthropic message_stop requires a preceding terminal message_delta")
	}
	return decoder.emitAnthropicEvent(llmprotocol.Event{
		Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop,
		MatchedStopSequence: decoder.stopSequence, Usage: &decoder.usage,
	})
}

func decodeAnthropicMessageStart(wire anthropicEventWire) llmprotocol.Event {
	event := llmprotocol.Event{Type: llmprotocol.EventResponseStarted}
	if wire.Message == nil {
		return event
	}
	event.ResponseID, event.Model = wire.Message.ID, wire.Message.Model
	if wire.Message.Usage != nil {
		usage := decodeAnthropicStreamUsage(*wire.Message.Usage, true)
		event.Usage = &usage
	}
	return event
}

func (decoder *anthropicStreamDecoder) decodeAnthropicMessageDelta(
	wire anthropicEventWire,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if err := decoder.observeAnthropicStop(wire.Delta); err != nil {
		return nil, nil, err
	}
	diagnostics := decoder.anthropicMessageDeltaDiagnostics(wire.Delta)
	if wire.Usage == nil {
		return nil, diagnostics, nil
	}
	usage := decodeAnthropicMessageDeltaUsage(*wire.Usage)
	events, eventDiagnostics, err := decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &usage})
	return events, appendDiagnostics(diagnostics, eventDiagnostics, decoder.policy.Limits.Diagnostics), err
}

func (decoder *anthropicStreamDecoder) observeAnthropicStop(delta *anthropicDeltaWire) error {
	if delta == nil || delta.StopReason == nil {
		return nil
	}
	stop := decodeAnthropicStop(*delta.StopReason)
	if decoder.stop != "" && decoder.stop != stop {
		return invalidProviderResponse("stream_stop_reason_mismatch", "Anthropic stream changed its terminal reason")
	}
	decoder.stop = stop
	if stop != llmprotocol.StopSequence || delta.StopSequence == nil {
		return nil
	}
	if decoder.stopSequence != "" && decoder.stopSequence != *delta.StopSequence {
		return invalidProviderResponse("stream_stop_sequence_mismatch", "Anthropic stream changed its matched stop sequence")
	}
	decoder.stopSequence = *delta.StopSequence
	return nil
}

func (decoder *anthropicStreamDecoder) anthropicMessageDeltaDiagnostics(delta *anthropicDeltaWire) llmprotocol.Diagnostics {
	var diagnostics llmprotocol.Diagnostics
	if delta == nil {
		return diagnostics
	}
	if len(delta.Container) > 0 && !bytes.Equal(bytes.TrimSpace(delta.Container), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.delta.container", "container metadata has no protocol-neutral representation",
		)
	}
	if len(delta.StopDetails) > 0 && !bytes.Equal(bytes.TrimSpace(delta.StopDetails), []byte("null")) {
		appendProviderFieldOmission(
			&diagnostics, decoder.policy, llmprotocol.AnthropicMessagesV1,
			"stream.delta.stop_details", "refusal details have no protocol-neutral representation",
		)
	}
	return diagnostics
}

func decodeAnthropicStreamError(wire anthropicEventWire) llmprotocol.Event {
	protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
	if wire.Error != nil {
		protocolError.Category = decodeProviderErrorCategory(wire.Error.Type)
		protocolError.Code, protocolError.Message = wire.Error.Type, wire.Error.Message
	}
	return llmprotocol.Event{
		Type: llmprotocol.EventResponseFailed, Error: protocolError,
		StopReason: llmprotocol.StopError, Failure: llmprotocol.FailureTransport,
	}
}

func (decoder *anthropicStreamDecoder) decodeUnknownAnthropicEvent(
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if decoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || decoder.context.Source != decoder.context.Target {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Anthropic stream event is unsupported", nil)
	}
	return decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventProviderOpaque, Opaque: append([]byte(nil), frame...)})
}

func (decoder *anthropicStreamDecoder) emitAnthropicEvent(
	event llmprotocol.Event,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	normalized, err := decoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	return []llmprotocol.Event{normalized}, nil, nil
}

func decodeAnthropicContentStart(wire anthropicEventWire) (llmprotocol.Event, error) {
	event := llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: anthropicEventIndex(wire), Role: llmprotocol.RoleAssistant}
	if wire.ContentBlock == nil {
		return event, nil
	}
	event.ItemID = wire.ContentBlock.ID
	switch wire.ContentBlock.Type {
	case "tool_use":
		event.ToolCall = &llmprotocol.ToolCall{ID: wire.ContentBlock.ID, Name: wire.ContentBlock.Name, Arguments: string(wire.ContentBlock.Input)}
	case "thinking":
		event.Content = &llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Signature: wire.ContentBlock.Signature,
			Reasoning: llmprotocol.ReasoningScopeText,
		}
	case "text":
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText}
	default:
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic stream content block is unsupported", nil)
	}
	return event, nil
}

func decodeAnthropicContentDelta(wire anthropicEventWire) (llmprotocol.Event, error) {
	if wire.Delta == nil {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_delta", "Anthropic stream delta is missing", nil)
	}
	event := llmprotocol.Event{ItemIndex: anthropicEventIndex(wire)}
	switch wire.Delta.Type {
	case "text_delta":
		event.Type, event.Delta, event.Content = llmprotocol.EventOutputTextDelta, wire.Delta.Text, &llmprotocol.Content{Kind: llmprotocol.ContentText, Text: wire.Delta.Text}
	case "thinking_delta":
		event.Type, event.Delta, event.Content = llmprotocol.EventReasoningDelta, wire.Delta.Thinking, &llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Text: wire.Delta.Thinking,
			Reasoning: llmprotocol.ReasoningScopeText,
		}
	case "input_json_delta":
		event.Type, event.ToolCall = llmprotocol.EventToolCallDelta, &llmprotocol.ToolCall{Arguments: wire.Delta.PartialJSON}
	case "signature_delta":
		event.Type, event.Content = llmprotocol.EventReasoningDelta, &llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Signature: wire.Delta.Signature,
			Reasoning: llmprotocol.ReasoningScopeText,
		}
	default:
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_delta", "Anthropic stream delta is unsupported", nil)
	}
	return event, nil
}

func decodeAnthropicStreamUsage(wire anthropicUsageWire, initial bool) llmprotocol.Usage {
	usage := llmprotocol.Usage{State: llmprotocol.UsageAvailable}
	if initial || wire.InputTokens > 0 || wire.CacheReadInputTokens > 0 || wire.CacheCreationInputTokens > 0 {
		usage.InputUncached = authoritative(wire.InputTokens)
		usage.InputCacheRead = authoritative(wire.CacheReadInputTokens)
		usage.InputCacheWrite = authoritative(wire.CacheCreationInputTokens)
		usage.InputTotal = authoritative(wire.InputTokens + wire.CacheReadInputTokens + wire.CacheCreationInputTokens)
	}
	if wire.OutputTokens > 0 || !initial {
		usage.OutputReasoning, usage.OutputOther = anthropicOutputSplit(wire)
		usage.OutputTotal = authoritative(wire.OutputTokens)
	}
	return usage
}

func decodeAnthropicMessageDeltaUsage(wire anthropicMessageDeltaUsageWire) llmprotocol.Usage {
	return decodeAnthropicStreamUsage(anthropicUsageWire{
		InputTokens: wire.InputTokens, OutputTokens: wire.OutputTokens,
		CacheCreationInputTokens: wire.CacheCreationInputTokens,
		CacheReadInputTokens:     wire.CacheReadInputTokens,
		OutputTokensDetails:      wire.OutputTokensDetails,
		ServerToolUse:            wire.ServerToolUse,
	}, false)
}

func encodeAnthropicMessageDeltaUsage(usage llmprotocol.Usage) *anthropicMessageDeltaUsageWire {
	full := encodeAnthropicUsage(usage)
	return &anthropicMessageDeltaUsageWire{
		CacheCreationInputTokens: full.CacheCreationInputTokens,
		CacheReadInputTokens:     full.CacheReadInputTokens,
		InputTokens:              full.InputTokens,
		OutputTokens:             full.OutputTokens,
		OutputTokensDetails:      full.OutputTokensDetails,
		ServerToolUse:            full.ServerToolUse,
	}
}

func (decoder *anthropicStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *anthropicStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, pushErr := encoder.next(event)
	if pushErr != nil {
		return nil, nil, pushErr
	}
	event = normalized
	if event.Type == llmprotocol.EventResponseCompleted {
		return encoder.encodeAnthropicCompletion(event)
	}
	if event.Type == llmprotocol.EventProviderOpaque {
		return encoder.encodeAnthropicOpaque(event)
	}
	if isAnthropicContentEvent(event.Type) {
		return encoder.encodeAnthropicContentEvent(event)
	}
	return encoder.encodeAnthropicLifecycleEvent(event)
}

func isAnthropicContentEvent(eventType llmprotocol.EventType) bool {
	return eventType == llmprotocol.EventOutputItemStarted ||
		eventType == llmprotocol.EventOutputTextDelta ||
		eventType == llmprotocol.EventReasoningDelta ||
		eventType == llmprotocol.EventToolCallDelta ||
		eventType == llmprotocol.EventOutputItemCompleted
}

func (encoder *anthropicStreamEncoder) encodeAnthropicContentEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		return encoder.encodeAnthropicItemStartEvent(event)
	case llmprotocol.EventOutputTextDelta:
		return encoder.encodeAnthropicTextDelta(event)
	case llmprotocol.EventReasoningDelta:
		return encoder.encodeAnthropicReasoningDelta(event)
	case llmprotocol.EventToolCallDelta:
		return validateAnthropicToolDelta(event)
	case llmprotocol.EventOutputItemCompleted:
		return encoder.completeAnthropicItem(event)
	default:
		return nil, nil, nil
	}
}

func (encoder *anthropicStreamEncoder) encodeAnthropicItemStartEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.ToolCall == nil && event.Content == nil || event.ToolCall != nil {
		return nil, nil, nil
	}
	kind := llmprotocol.ContentText
	if event.Content.Kind != "" {
		kind = event.Content.Kind
	}
	frames, _, err := encoder.ensureAnthropicBlockStarted(event, kind)
	return frames, nil, err
}

func (encoder *anthropicStreamEncoder) encodeAnthropicReasoningDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	frames, key, err := encoder.ensureAnthropicBlockStarted(event, llmprotocol.ContentReasoning)
	if err != nil {
		return nil, nil, err
	}
	blockIndex := encoder.blockIndexes[key]
	frames, err = appendAnthropicReasoningText(frames, blockIndex, event.Delta)
	if err != nil {
		return nil, nil, err
	}
	frames, err = appendAnthropicReasoningSignature(frames, blockIndex, event.Content)
	return frames, nil, err
}

func appendAnthropicReasoningText(frames [][]byte, blockIndex int, text string) ([][]byte, error) {
	if text == "" {
		return frames, nil
	}
	wire := anthropicEventWire{
		Type: "content_block_delta", Index: anthropicIndex(blockIndex),
		Delta: &anthropicDeltaWire{Type: "thinking_delta", Thinking: text},
	}
	delta, _, err := encodeAnthropicWireFrame(wire)
	if err != nil {
		return frames, err
	}
	return append(frames, delta...), nil
}

func appendAnthropicReasoningSignature(
	frames [][]byte,
	blockIndex int,
	content *llmprotocol.Content,
) ([][]byte, error) {
	if content == nil || content.Signature == "" {
		return frames, nil
	}
	wire := anthropicEventWire{
		Type: "content_block_delta", Index: anthropicIndex(blockIndex),
		Delta: &anthropicDeltaWire{Type: "signature_delta", Signature: content.Signature},
	}
	delta, _, err := encodeAnthropicWireFrame(wire)
	if err != nil {
		return frames, err
	}
	return append(frames, delta...), nil
}

func validateAnthropicToolDelta(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.ToolCall == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
	}
	return nil, nil, nil
}

func (encoder *anthropicStreamEncoder) encodeAnthropicLifecycleEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var wire anthropicEventWire
	var diagnostics llmprotocol.Diagnostics
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		wire = encodeAnthropicMessageStart(event)
	case llmprotocol.EventUsageUpdated:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "usage event is invalid", nil)
		}
		// streamState carries the merged usage into the terminal message_delta.
		// Emitting here would duplicate usage for source streams that publish a
		// usage update immediately before their terminal event.
		return nil, nil, nil
	case llmprotocol.EventResponseFailed:
		// A cut turn's count is the one exception: the caller asked for it to
		// travel, because nothing downstream can bill the turn without it.
		// Any other usage on a failure has nowhere to go in Messages.
		if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable && encoder.truncationSource == "" {
			appendAccountingOmission(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "usage", "Messages error events cannot carry token usage")
		}
		failureError := event.Error
		if failureError == nil {
			failureError = llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete",
				"upstream stream ended before completion", nil,
			)
		}
		frames, err := encoder.terminateAnthropicStream(failureError)
		return frames, diagnostics, err
	default:
		return nil, nil, nil
	}
	frames, _, err := encodeAnthropicWireFrame(wire)
	return frames, diagnostics, err
}

func encodeAnthropicWireFrame(wire anthropicEventWire) ([][]byte, llmprotocol.Diagnostics, error) {
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}

func encodeAnthropicMessageStart(event llmprotocol.Event) anthropicEventWire {
	usage := newAnthropicUsageWire()
	if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
		usage = encodeAnthropicUsage(*event.Usage)
	}
	return anthropicEventWire{
		Type: "message_start",
		Message: &anthropicResponseWire{
			ID: event.ResponseID, Type: "message", Role: "assistant", Model: event.Model,
			Content: json.RawMessage(`[]`), Usage: usage,
		},
	}
}

func (encoder *anthropicStreamEncoder) encodeAnthropicCompletion(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.Usage == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
	}
	stop := encodeAnthropicStop(event.StopReason)
	deltaWire := &anthropicDeltaWire{Type: "message_delta", StopReason: &stop}
	if event.StopReason == llmprotocol.StopSequence {
		deltaWire.StopSequence = &event.MatchedStopSequence
	}
	delta := anthropicEventWire{Type: "message_delta", Delta: deltaWire, Usage: encodeAnthropicMessageDeltaUsage(*event.Usage)}
	first, err := encodeSSE(delta.Type, delta)
	if err != nil {
		return nil, nil, err
	}
	encoder.terminal = true
	stopEvent := anthropicEventWire{Type: "message_stop"}
	second, err := encodeSSE(stopEvent.Type, stopEvent)
	return [][]byte{first, second}, nil, err
}

func (encoder *anthropicStreamEncoder) encodeAnthropicFailure(event llmprotocol.Event) (anthropicEventWire, error) {
	if event.Error == nil {
		return anthropicEventWire{}, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	return anthropicEventWire{
		Type:  "error",
		Error: &anthropicErrorWire{Type: canonicalAnthropicErrorType(event.Error), Message: event.Error.Message},
	}, nil
}

func (encoder *anthropicStreamEncoder) encodeAnthropicOpaque(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
	}
	return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
}

func (encoder *anthropicStreamEncoder) encodeAnthropicTextDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	if event.Content != nil && len(event.Content.Citations) > 0 {
		if err := appendLossy(
			&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target,
			"content.citations", "Messages cannot represent URL citations",
		); err != nil {
			return nil, diagnostics, err
		}
	}
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
		if err := appendLossy(
			&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target,
			"content.refusal", "Messages represents refusal as ordinary text",
		); err != nil {
			return nil, diagnostics, err
		}
	}
	frames, key, err := encoder.ensureAnthropicBlockStarted(event, llmprotocol.ContentText)
	if err != nil {
		return nil, diagnostics, err
	}
	if event.Delta == "" {
		return frames, diagnostics, nil
	}
	wire := anthropicEventWire{
		Type: "content_block_delta", Index: anthropicIndex(encoder.blockIndexes[key]),
		Delta: &anthropicDeltaWire{Type: "text_delta", Text: event.Delta},
	}
	frame, err := encodeSSE(wire.Type, wire)
	if err != nil {
		return frames, diagnostics, err
	}
	return append(frames, frame), diagnostics, nil
}

func (encoder *anthropicStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	protocolError := streamFinalizationError(reason, "stream ended before completion")
	frames, err := encoder.terminateAnthropicStream(protocolError)
	return frames, nil, err
}

// terminateAnthropicStream ends a message the upstream never finished. The
// client is holding an open content block and a message with no stop reason,
// so the block is closed, the failure is named, and the message is ended. A
// stream that just stops instead is indistinguishable from a slow one: the
// turn that prompted this reached the client as 1,191 deltas and a clean
// close, and curl exited 0.
// A turn the Router cut carries what it cost, so the client's proxy can bill
// it. The count rides beside the error object rather than inside it, and
// usageSource says it is not a settlement.
func (encoder *anthropicStreamEncoder) terminateAnthropicStream(
	protocolError *llmprotocol.ProtocolError,
) ([][]byte, error) {
	frames, err := encoder.closeOpenAnthropicBlocks()
	if err != nil {
		return nil, err
	}
	failure := anthropicTruncationFrameWire{Type: "error", Error: &anthropicErrorWire{
		Type: canonicalAnthropicErrorType(protocolError), Message: protocolError.Message,
	}}
	if encoder.truncationUsage != nil && encoder.truncationSource != "" {
		failure.Usage = encodeAnthropicMessageDeltaUsage(*encoder.truncationUsage)
		failure.UsageSource = encoder.truncationSource
	}
	failureFrame, err := encodeSSE(failure.Type, failure)
	if err != nil {
		return nil, err
	}
	return append(frames, failureFrame), nil
}
