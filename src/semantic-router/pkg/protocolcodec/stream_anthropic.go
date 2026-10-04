package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type anthropicStreamDecoder struct {
	streamState
	framer       sseFramer
	stopSequence string
	// stopDetails is the terminal message_delta's stop_details, carried on
	// the completed event as telemetry (llmprotocol.StopDetails).
	stopDetails *llmprotocol.StopDetails
	// serverBlocks holds each open server web search block until it stops:
	// its start frame, and the input a server_tool_use streams as JSON deltas.
	serverBlocks map[int]*anthropicServerBlock
	// textRunes counts each text block's characters, and pendingCitations
	// holds its streamed web search citations until the block stops: a
	// citation may arrive before the text it supports, and it covers the
	// whole block, as a buffered response's does.
	textRunes        map[int]int64
	pendingCitations map[int][]json.RawMessage
	// data is the JSON of the event being decoded.
	data []byte
	// usageUnknown records a terminal message_delta whose usage this Router
	// marked as the schema's placeholder (usage_source "unknown").
	usageUnknown bool
	// pendingStartUsage is a message_start usage of zero counts and no
	// charge, held until the stream says what it was: a Router's placeholder
	// when a later message_delta is marked usage_source "unknown" (then it
	// is dropped), otherwise the provider's evidence, applied before the
	// next usage or terminal event so nothing it stated is lost.
	pendingStartUsage *llmprotocol.Usage
	// terminalDelta is the JSON of the last message_delta before
	// message_stop: a late message_delta that restates it adds nothing.
	terminalDelta json.RawMessage
	// pendingToolStop is the index of a tool_use block that has stopped but
	// whose completion is held until the next event says whether it was the
	// reply's last block under a max_tokens stop (flushPendingToolStop).
	pendingToolStop *int
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
	// refused records that refusal content reached the client, so the turn
	// ends as one: stop_reason "refusal" is how Messages marks refusal text.
	refused bool
	// reasoningDetailsDropped records that the turn's reasoning_details drop
	// was reported, so it is reported once rather than per fragment.
	reasoningDetailsDropped bool
	// reasoningDetails and reasoningText accumulate, per thinking block, the
	// reasoning_details fragments and the thinking text the block streamed,
	// so the block can be signed with Claude's signature when it stops
	// (chatClaudeThinkingSignature). blockSigned records a block that was
	// already sent a signature.
	reasoningDetails map[anthropicBlockKey]json.RawMessage
	reasoningText    map[anthropicBlockKey]string
	blockSigned      map[anthropicBlockKey]bool
}

func (encoder *anthropicStreamEncoder) SetTruncationUsage(usage *llmprotocol.Usage, source string) {
	encoder.truncationUsage = usage
	encoder.truncationSource = source
}

func (AnthropicMessagesCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return newStateDiagnosticDecoder(&anthropicStreamDecoder{streamState: streamState{context: context, policy: policy}, framer: newSSEFramer(policy.Limits.SSEFrameBytes)}, policy)
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
// anthropicUnknownUsageDeltaWire is a terminal message_delta whose usage the
// provider never stated, marked with the Router's usage_source.
type anthropicUnknownUsageDeltaWire struct {
	anthropicEventWire
	UsageSource string `json:"usage_source"`
}

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
	providerUsageCostWire
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
	if err != nil {
		return nil, nil, err
	}
	if parsed.commentOnly() {
		return decoder.keepalive()
	}
	if !parsed.HasData {
		return nil, nil, nil
	}
	if decoder.terminal {
		return decoder.afterTerminal(parsed)
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
	decoder.data = data
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

// decodeEvent releases a held tool_use completion before any event but a
// ping, then decodes the event.
func (decoder *anthropicStreamDecoder) decodeEvent(
	wire anthropicEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if decoder.pendingToolStop == nil || wire.Type == "ping" {
		return decoder.decodeAnthropicEvent(wire, frame)
	}
	cut := wire.Type == "message_delta" && wire.Delta != nil && wire.Delta.StopReason != nil &&
		decodeAnthropicStop(*wire.Delta.StopReason) == llmprotocol.StopMaxTokens
	flushed, diagnostics, err := decoder.flushPendingToolStop(cut)
	if err != nil {
		return flushed, diagnostics, err
	}
	events, eventDiagnostics, err := decoder.decodeAnthropicEvent(wire, frame)
	return append(flushed, events...), appendDiagnostics(diagnostics, eventDiagnostics, decoder.policy.Limits.Diagnostics), err
}

// flushPendingToolStop completes the held tool_use block. Anthropic documents
// a max_tokens reply whose last block is a tool_use as one holding an
// incomplete tool use, and a buffered reply of that shape is decoded so
// (markAnthropicCutToolCall). A stream says the same only once message_delta
// arrives after the block stops, so the completion waits for it: when the
// next event is that max_tokens message_delta, the call completes marked
// incomplete. That includes a block cut before any input_json_delta, whose
// arguments are the start's {} placeholder and would otherwise read whole.
// Any other next event completes the call as it stands.
func (decoder *anthropicStreamDecoder) flushPendingToolStop(cut bool) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	index := *decoder.pendingToolStop
	decoder.pendingToolStop = nil
	event := llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: index}
	if cut {
		event.ToolCall = &llmprotocol.ToolCall{Incomplete: true}
	}
	return decoder.emitAnthropicEvent(event)
}

func (decoder *anthropicStreamDecoder) decodeAnthropicEvent(
	wire anthropicEventWire,
	frame []byte,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if events, handled, err := decoder.decodeAnthropicWebSearchEvent(wire); handled {
		return events, nil, err
	}
	switch wire.Type {
	case "message_start":
		event := decodeAnthropicMessageStart(wire)
		if wire.Message != nil && wire.Message.Usage != nil && anthropicUsageUncommitted(*wire.Message.Usage) {
			decoder.pendingStartUsage, event.Usage = event.Usage, nil
		}
		return decoder.emitAnthropicEvent(event)
	case "content_block_start":
		return decoder.emitDecodedAnthropicEvent(decodeAnthropicContentStart(wire))
	case "content_block_delta":
		if wire.Delta != nil && wire.Delta.Type == "text_delta" {
			decoder.countText(anthropicEventIndex(wire), wire.Delta.Text)
		}
		return decoder.emitDecodedAnthropicEvent(decodeAnthropicContentDelta(wire))
	case "content_block_stop":
		index := anthropicEventIndex(wire)
		if decoder.items[index] && !decoder.completedItems[index] && decoder.itemKinds[index] == llmprotocol.ContentToolCall {
			decoder.pendingToolStop = &index
			return nil, nil, nil
		}
		return decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: index})
	case "message_delta":
		return decoder.decodeAnthropicMessageDelta(wire)
	case "message_stop":
		return decoder.decodeAnthropicMessageStop()
	case "error":
		return decoder.emitAnthropicEvent(decodeAnthropicStreamError(wire))
	case "ping":
		// The provider is alive and thinking; the client hears so. A ping
		// after message_stop never reaches here (afterTerminal).
		return decoder.keepalive()
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
		MatchedStopSequence: decoder.stopSequence, Usage: &decoder.usage, StopDetails: decoder.stopDetails,
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

// anthropicUsageUncommitted reports a message_start usage that may be a
// Router's placeholder: zero counts and no charge. A charge (cost, is_byok,
// cost_details) is evidence in its own right.
func anthropicUsageUncommitted(wire anthropicUsageWire) bool {
	return wire.InputTokens == 0 && wire.OutputTokens == 0 &&
		wire.CacheCreationInputTokens == 0 && wire.CacheReadInputTokens == 0 &&
		len(wire.Cost) == 0 && len(wire.IsBYOK) == 0 && len(wire.CostDetails) == 0
}

func (decoder *anthropicStreamDecoder) decodeAnthropicMessageDelta(
	wire anthropicEventWire,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	decoder.terminalDelta = append(json.RawMessage(nil), decoder.data...)
	if err := decoder.observeAnthropicStop(wire.Delta); err != nil {
		return nil, nil, err
	}
	diagnostics := decoder.anthropicMessageDeltaDiagnostics(wire.Delta)
	var marked struct {
		UsageSource string `json:"usage_source"`
	}
	if err := json.Unmarshal(decoder.data, &marked); err == nil && marked.UsageSource != "" {
		if marked.UsageSource != UsageSourceUnknown {
			return nil, diagnostics, invalidProviderResponse("usage_source_invalid", "usage_source must be "+UsageSourceUnknown)
		}
		// The delta's usage is a placeholder, and so was a held zero-count
		// message_start usage: the turn's usage is unknown.
		decoder.usageUnknown = true
		decoder.pendingStartUsage = nil
		return nil, diagnostics, nil
	}
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
	if details := decodeStopDetails(delta.StopDetails); details != nil {
		decoder.stopDetails = details
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
	var events []llmprotocol.Event
	if decoder.pendingStartUsage != nil && (event.Type == llmprotocol.EventUsageUpdated ||
		event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed) {
		flushed, err := decoder.flushPendingStartUsage()
		if err != nil {
			return nil, nil, err
		}
		events = append(events, flushed...)
	}
	normalized, err := decoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	return append(events, normalized), nil, nil
}

// flushPendingStartUsage applies a held message_start usage as the provider's
// evidence.
func (decoder *anthropicStreamDecoder) flushPendingStartUsage() ([]llmprotocol.Event, error) {
	pending := decoder.pendingStartUsage
	decoder.pendingStartUsage = nil
	normalized, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: pending})
	if err != nil {
		return nil, err
	}
	return []llmprotocol.Event{normalized}, nil
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
	usage := llmprotocol.Usage{State: llmprotocol.UsageAvailable, ProviderCost: wire.providerUsageCostWire.decode()}
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
		providerUsageCostWire:    wire.providerUsageCostWire,
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
	if decoder.pendingToolStop != nil && !decoder.terminal {
		// No max_tokens message_delta followed: the call completes as it
		// stands, and the stream, which has no terminal, fails below as
		// incomplete.
		flushed, _, err := decoder.flushPendingToolStop(false)
		events = append(events, flushed...)
		if err != nil {
			return events, diagnostics, err
		}
	}
	if decoder.pendingStartUsage != nil {
		flushed, err := decoder.flushPendingStartUsage()
		if err != nil {
			return events, diagnostics, err
		}
		events = append(events, flushed...)
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *anthropicStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	if event.Content != nil {
		// A web search the provider ran is shown only to a Responses client;
		// here it would open an empty block. The answer follows as its own
		// item.
		if _, webSearch := carriedWebSearchCall(*event.Content); webSearch {
			return nil, nil, nil
		}
	}
	normalized, pushErr := encoder.next(event)
	if pushErr != nil {
		return nil, nil, pushErr
	}
	event = normalized
	if event.Type == llmprotocol.EventKeepalive {
		return encoder.encodeAnthropicKeepalive()
	}
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

// encodeAnthropicKeepalive writes a ping, as Anthropic itself does, once the
// message has started. Anthropic sends its pings after message_start, and a
// client may expect message_start first, so a keepalive before it (a provider
// still queueing the request) goes out as an SSE comment, which every SSE
// parser skips.
func (encoder *anthropicStreamEncoder) encodeAnthropicKeepalive() ([][]byte, llmprotocol.Diagnostics, error) {
	if !encoder.started {
		return [][]byte{sseKeepaliveComment()}, nil, nil
	}
	return encodeAnthropicWireFrame(anthropicEventWire{Type: "ping"})
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
	if serverBlock := carriedAnthropicServerBlock(event.Content); serverBlock != nil {
		if event.Type == llmprotocol.EventOutputItemStarted {
			frames, err := encoder.startAnthropicServerBlock(event, serverBlock)
			return frames, nil, err
		}
		if event.Type == llmprotocol.EventOutputItemCompleted {
			frames, err := encoder.completeAnthropicServerBlock(event, serverBlock)
			return frames, nil, err
		}
	}
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
	var diagnostics llmprotocol.Diagnostics
	if event.Content != nil && !encoder.reasoningDetailsDropped {
		// Messages has no member for reasoning_details, as on the buffered
		// path (encodeAnthropicResponse).
		if details, _ := reasoningDetailsOf(*event.Content); details != nil {
			encoder.reasoningDetailsDropped = true
			appendUnmodeledDrop(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "content.reasoning_details")
		}
	}
	if event.Delta == "" && (event.Content == nil || event.Content.Signature == "") {
		// Nothing a thinking block can show: a reasoning_details fragment with
		// no text (an encrypted blob, an OpenRouter signature). Opening a block
		// for it would hand the client an empty thinking block to replay. A
		// fragment for the open thinking block is kept for its signature.
		key := encoder.liveBlockKey(contentKey(event))
		if encoder.blockStarted[key] && !encoder.blockStopped[key] {
			if err := encoder.noteReasoningDetails(key, event.Content); err != nil {
				return nil, diagnostics, err
			}
		}
		return nil, diagnostics, nil
	}
	frames, key, err := encoder.ensureAnthropicBlockStarted(event, llmprotocol.ContentReasoning)
	if err != nil {
		return nil, diagnostics, err
	}
	if err := encoder.noteReasoningDetails(key, event.Content); err != nil {
		return nil, diagnostics, err
	}
	encoder.noteReasoningText(key, event.Delta)
	blockIndex := encoder.blockIndexes[key]
	frames, err = appendAnthropicReasoningText(frames, blockIndex, event.Delta)
	if err != nil {
		return nil, diagnostics, err
	}
	if event.Content != nil && event.Content.Signature != "" {
		encoder.markBlockSigned(key)
	}
	frames, err = appendAnthropicReasoningSignature(frames, blockIndex, event.Content)
	return frames, diagnostics, err
}

// noteReasoningDetails folds a reasoning delta's reasoning_details fragment
// into what its thinking block has streamed so far.
func (encoder *anthropicStreamEncoder) noteReasoningDetails(key anthropicBlockKey, content *llmprotocol.Content) error {
	if content == nil {
		return nil
	}
	fragment, _ := reasoningDetailsOf(*content)
	if fragment == nil {
		return nil
	}
	if encoder.reasoningDetails == nil {
		encoder.reasoningDetails = map[anthropicBlockKey]json.RawMessage{}
	}
	merged, err := mergeReasoningDetailsFragment(encoder.reasoningDetails[key], fragment)
	if err != nil {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_reasoning_details",
			"upstream stream sent malformed reasoning_details", err)
	}
	encoder.reasoningDetails[key] = merged
	return nil
}

func (encoder *anthropicStreamEncoder) noteReasoningText(key anthropicBlockKey, text string) {
	if text == "" {
		return
	}
	if encoder.reasoningText == nil {
		encoder.reasoningText = map[anthropicBlockKey]string{}
	}
	encoder.reasoningText[key] += text
}

func (encoder *anthropicStreamEncoder) markBlockSigned(key anthropicBlockKey) {
	if encoder.blockSigned == nil {
		encoder.blockSigned = map[anthropicBlockKey]bool{}
	}
	encoder.blockSigned[key] = true
}

// claudeSignatureFrames is the signature_delta that signs a thinking block
// about to stop with Claude's signature from its reasoning_details, where
// Anthropic sends it: after the thinking, before content_block_stop.
func (encoder *anthropicStreamEncoder) claudeSignatureFrames(key anthropicBlockKey) ([][]byte, error) {
	if encoder.blocks[key] != llmprotocol.ContentReasoning || encoder.blockSigned[key] {
		return nil, nil
	}
	signature := chatClaudeThinkingSignature(encoder.reasoningDetails[key], encoder.reasoningText[key])
	if signature == "" {
		return nil, nil
	}
	encoder.markBlockSigned(key)
	return appendAnthropicReasoningSignature(nil, encoder.blockIndexes[key], &llmprotocol.Content{Signature: signature})
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
	refusal := event.StopReason == llmprotocol.StopContentFilter || encoder.refused
	stop := encodeAnthropicStop(event.StopReason)
	if refusal {
		stop = encodeAnthropicStop(llmprotocol.StopContentFilter)
	}
	deltaWire := &anthropicDeltaWire{Type: "message_delta", StopReason: &stop}
	if event.StopReason == llmprotocol.StopSequence && !refusal {
		deltaWire.StopSequence = &event.MatchedStopSequence
	}
	delta := anthropicEventWire{Type: "message_delta", Delta: deltaWire, Usage: encodeAnthropicMessageDeltaUsage(*event.Usage)}
	var first []byte
	var err error
	if refusal && usageUnavailable(*event.Usage) {
		// The delta's usage is the schema's placeholder; usage_source says so.
		first, err = encodeSSE(delta.Type, anthropicUnknownUsageDeltaWire{anthropicEventWire: delta, UsageSource: UsageSourceUnknown})
	} else {
		first, err = encodeSSE(delta.Type, delta)
	}
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
	if event.Content != nil && len(event.Content.Citations) > 0 && len(event.Content.CitationsRaw) == 0 {
		if err := appendLossy(
			&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target,
			"content.citations", "Messages cannot represent URL citations",
		); err != nil {
			return nil, diagnostics, err
		}
	}
	// Refusal text streams as text, and the terminal message_delta states
	// stop_reason "refusal", which is how Messages marks it (#150).
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
		encoder.refused = true
	}
	frames, key, err := encoder.ensureAnthropicBlockStarted(event, llmprotocol.ContentText)
	if err != nil {
		return nil, diagnostics, err
	}
	if event.Content != nil && len(event.Content.CitationsRaw) > 0 {
		citations, err := encoder.encodeAnthropicCitationDeltas(key, event.Content.CitationsRaw)
		if err != nil {
			return frames, diagnostics, err
		}
		frames = append(frames, citations...)
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

// afterTerminal handles a frame that arrives after message_stop. The message
// is complete by then, so a frame that can add nothing to it -- an
// OpenAI-style [DONE] sentinel, a ping, a repeated message_stop, a late
// message_delta (usage or a restated stop) -- is dropped with a diagnostic
// rather than cutting a stream whose answer the client already has. Anything
// else (a new content block, an error) still fails the stream, and the
// failure names the event type, so the frame a provider sent is on record.
func (decoder *anthropicStreamDecoder) afterTerminal(parsed sseFrame) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		return nil, afterTerminalDiagnostic("[DONE]"), nil
	}
	eventType, err := decodeProviderEventType(parsed.Data, parsed.Event, decoder.policy)
	if err != nil {
		return nil, nil, invalidProviderResponse("stream_event_after_terminal", "Anthropic stream emitted an undecodable frame after message_stop")
	}
	switch eventType {
	case "ping", "message_stop":
		// Only the bare event: a ping or stop carrying anything else (usage,
		// a delta) brings evidence the stream can no longer carry.
		if bareEvent(parsed.Data) {
			return nil, afterTerminalDiagnostic(eventType), nil
		}
	case "message_delta":
		// Usage and stop reason were published with the completion at
		// message_stop, and the client has it: a late delta is dropped only
		// when it restates them. New counts or a charge are accounting the
		// stream can no longer carry, so they fail it rather than vanish.
		if decoder.lateDeltaRestates(parsed.Data) {
			return nil, afterTerminalDiagnostic(eventType), nil
		}
		return nil, nil, invalidProviderResponse("stream_event_after_terminal",
			"Anthropic stream emitted a message_delta with new usage or stop after message_stop")
	}
	if eventType == "ping" || eventType == "message_stop" {
		return nil, nil, invalidProviderResponse("stream_event_after_terminal",
			"Anthropic stream emitted a "+eventType+" carrying data after message_stop")
	}
	return nil, nil, invalidProviderResponse("stream_event_after_terminal",
		"Anthropic stream emitted "+boundedEventName(eventType)+" after message_stop")
}

func afterTerminalDiagnostic(frame string) llmprotocol.Diagnostics {
	return llmprotocol.Diagnostics{{
		Source: llmprotocol.AnthropicMessagesV1, Field: "stream.after_message_stop", Action: llmprotocol.DiagnosticDropped,
		Reason: "the provider sent " + frame + " after message_stop; the message was already complete",
	}}
}

// boundedEventName keeps a provider-chosen event name fit for an error
// message: at most 48 characters of [a-z0-9_.], anything else replaced.
func boundedEventName(name string) string {
	if len(name) > 48 {
		name = name[:48]
	}
	out := make([]byte, 0, len(name))
	for index := 0; index < len(name); index++ {
		character := name[index]
		if character >= 'a' && character <= 'z' || character >= '0' && character <= '9' || character == '_' || character == '.' {
			out = append(out, character)
		} else {
			out = append(out, '?')
		}
	}
	if len(out) == 0 {
		return "an unnamed event"
	}
	return string(out)
}

// lateDeltaRestates reports whether a message_delta after message_stop is
// the terminal message_delta again, as parsed JSON: its stop reason, matched
// stop sequence, usage and usage_source all as already published. Any
// difference is evidence the stream can no longer carry.
func (decoder *anthropicStreamDecoder) lateDeltaRestates(data []byte) bool {
	late, lateOK := exactJSON(data)
	terminal, terminalOK := exactJSON(decoder.terminalDelta)
	return len(decoder.terminalDelta) > 0 && lateOK && terminalOK && reflect.DeepEqual(late, terminal)
}

// exactJSON parses data keeping every number as written (json.Number), so two
// counts that differ beyond float64 precision still compare unequal.
func exactJSON(data []byte) (any, bool) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	return value, decoder.Decode(&value) == nil
}

// bareEvent reports whether an event's JSON holds nothing but its type.
func bareEvent(data []byte) bool {
	value, ok := exactJSON(data)
	object, isObject := value.(map[string]any)
	return ok && isObject && len(object) == 1 && object["type"] != nil
}
