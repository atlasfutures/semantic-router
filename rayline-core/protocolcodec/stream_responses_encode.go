package protocolcodec

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func (encoder *responsesStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, err := encoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	event = normalized
	if frames, diagnostics, handled, directErr := encoder.encodeDirectResponsesEvent(event); handled {
		return frames, diagnostics, directErr
	}
	var flushed [][]byte
	if event.Type == llmprotocol.EventResponseCompleted {
		if flushed, err = encoder.flushPendingAnthropicSearches(event); err != nil {
			return nil, nil, err
		}
	}
	wire, diagnostics, err := encoder.responsesWireForEvent(event)
	if err != nil || wire.Type == "" {
		return flushed, diagnostics, err
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return append(flushed, frame), diagnostics, err
}

func (encoder *responsesStreamEncoder) encodeDirectResponsesEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, bool, error) {
	if event.Type == llmprotocol.EventResponseFailed {
		if event.Failure == llmprotocol.FailureResponse {
			return nil, nil, false, nil
		}
		// A turn the Router cut ends as the resource Responses defines for one
		// that stopped early, response.incomplete with incomplete_details, and
		// that resource states usage. The bare error event this would otherwise
		// write carries none, which is what left a cut turn unbilled
		// (stream_truncation_usage.go).
		if event.Failure == llmprotocol.FailureTruncated {
			frames, diagnostics, err := encoder.encodeResponsesTruncation(event)
			return frames, diagnostics, true, err
		}
		frames, err := encoder.encodeResponsesTransportFailure(event)
		return frames, nil, true, err
	}
	return encoder.encodeDirectResponsesNonFailureEvent(event)
}

func (encoder *responsesStreamEncoder) encodeDirectResponsesNonFailureEvent(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, bool, error) {
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		frames, err := encoder.encodeResponsesStart(event)
		return frames, nil, true, err
	case llmprotocol.EventOutputItemStarted:
		frames, err := encoder.encodeResponsesItemStart(event)
		return frames, nil, true, err
	case llmprotocol.EventOutputTextDelta:
		frames, diagnostics, err := encoder.encodeResponsesTextDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventReasoningDelta:
		frames, diagnostics, err := encoder.encodeResponsesReasoningDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventToolCallDelta:
		frames, diagnostics, err := encoder.encodeResponsesToolDelta(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventImageGenerationProgress:
		frames, err := encoder.encodeResponsesImageGenerationProgress(event)
		return frames, nil, true, err
	case llmprotocol.EventOutputItemCompleted:
		frames, diagnostics, err := encoder.encodeCompletedResponsesItem(event)
		return frames, diagnostics, true, err
	case llmprotocol.EventUsageUpdated:
		return nil, nil, true, nil
	case llmprotocol.EventProviderOpaque:
		frames, err := encoder.encodeResponsesOpaque(event)
		return frames, nil, true, err
	case llmprotocol.EventKeepalive:
		// A comment takes no sequence_number: the next event's stays
		// contiguous with the last one's.
		return [][]byte{sseKeepaliveComment()}, nil, true, nil
	default:
		return nil, nil, false, nil
	}
}

func (encoder *responsesStreamEncoder) responsesWireForEvent(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	switch event.Type {
	case llmprotocol.EventResponseCompleted:
		return encoder.responsesCompletionWire(event)
	case llmprotocol.EventResponseFailed:
		return encoder.responsesFailureWire(event)
	default:
		return responsesEventWire{}, nil, nil
	}
}

func (encoder *responsesStreamEncoder) encodeResponsesItemStart(event llmprotocol.Event) ([][]byte, error) {
	encoder.neutralItemIDs[event.ItemIndex] = event.ItemID
	if carriedAnthropicServerBlock(event.Content) != nil {
		return nil, nil
	}
	if event.ToolCall != nil {
		frames, key, err := encoder.ensureResponsesOutputStarted(event, responsesOutputTool)
		if err != nil {
			return nil, err
		}
		// The call is known from its start, so a cut before its first
		// argument delta can still close it as the call it was. The start's
		// "{}" is a placeholder the deltas replace, not an argument, and is
		// not kept (streamState keeps the same rule for the accumulated call).
		seed := *event.ToolCall
		if strings.TrimSpace(seed.Arguments) == "{}" {
			seed.Arguments = ""
		}
		encoder.recordOpenToolCall(key, &seed)
		return frames, nil
	}
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentGeneratedImage {
		frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputImage)
		return frames, err
	}
	if event.Content != nil {
		if _, webSearch := carriedWebSearchCall(*event.Content); webSearch {
			frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputWebSearch)
			return frames, err
		}
	}
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
		encoder.recordReasoningSignature(event)
		key := contentKey(event)
		encoder.encodedKinds[key] = llmprotocol.ContentReasoning
		// An item-start event may identify a reasoning item without identifying
		// whether its later content is summary or encrypted reasoning text. Bind
		// the scope only when the source actually supplies it; the first content
		// event otherwise establishes the scope for its neutral content index.
		if event.Content.Reasoning != "" {
			encoder.reasoningScopes[key] = normalizedReasoningScope(event.Content.Reasoning)
		}
		frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputReasoning)
		return frames, err
	}
	if event.Content != nil {
		encoder.encodedKinds[contentKey(event)] = event.Content.Kind
		frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputMessage)
		return frames, err
	}
	return nil, nil
}

func (encoder *responsesStreamEncoder) ensureResponsesOutputStarted(
	event llmprotocol.Event,
	kind responsesOutputKind,
) ([][]byte, responsesOutputKey, error) {
	key := responsesOutputKey{item: event.ItemIndex, kind: kind}
	if encoder.outputStarted[key] {
		return nil, key, nil
	}
	encoder.outputStarted[key] = true
	encoder.outputIndexes[key] = encoder.nextOutputIndex
	encoder.nextOutputIndex++
	encoder.itemOutputKeys[event.ItemIndex] = append(encoder.itemOutputKeys[event.ItemIndex], key)
	id := encoder.responsesOutputID(event, key)
	encoder.outputIDs[key] = id
	item := responsesItemWire{Type: string(kind), ID: id, Status: "in_progress"}
	switch kind {
	case responsesOutputMessage:
		item.Role = "assistant"
	case responsesOutputTool:
		item.Type = "function_call"
		if event.ToolCall != nil {
			item.CallID, item.Name, item.Arguments = event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
			item.Namespace = event.ToolCall.Namespace
		}
	case responsesOutputImage:
		item.Type = "image_generation_call"
	}
	wire := responsesEventWire{
		Type:        "response.output_item.added",
		Sequence:    encoder.nextWireSequence(),
		OutputIndex: responsesOutputIndex(encoder.outputIndexes[key]),
		Item:        marshalResponsesEventItem(item),
	}
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return nil, key, err
	}
	return [][]byte{frame}, key, nil
}

func (encoder *responsesStreamEncoder) responsesOutputID(event llmprotocol.Event, key responsesOutputKey) string {
	base := event.ItemID
	if base == "" {
		base = encoder.neutralItemIDs[event.ItemIndex]
	}
	if base == "" {
		base = llmprotocol.StableID("responses-output", event.ResponseID, string(key.kind), strconv.Itoa(event.ItemIndex))
	}
	if len(encoder.itemOutputKeys[event.ItemIndex]) == 1 {
		return base
	}
	return llmprotocol.StableID("responses-output-split", base, string(key.kind))
}

func (encoder *responsesStreamEncoder) encodeResponsesStart(event llmprotocol.Event) ([][]byte, error) {
	if encoder.responseStarted {
		return nil, nil
	}
	encoder.responseStarted = true
	response := newResponsesResponseWire(event.ResponseID, event.Model, "in_progress", 0, encoder.context.PreviousResponseID)
	frames := make([][]byte, 0, 2)
	for _, eventType := range []string{"response.created", "response.in_progress"} {
		wire := responsesEventWire{
			Type: eventType, Sequence: encoder.nextWireSequence(), Response: &response,
		}
		frame, err := encoder.encodeResponsesStreamFrame(wire)
		if err != nil {
			return nil, err
		}
		frames = append(frames, frame)
	}
	return frames, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesToolDelta(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.ToolCall == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
	}
	frames, key, err := encoder.ensureResponsesOutputStarted(event, responsesOutputTool)
	if err != nil {
		return nil, nil, err
	}
	encoder.recordOpenToolCall(key, event.ToolCall)
	wire := responsesEventWire{
		Type:        "response.function_call_arguments.delta",
		Sequence:    encoder.nextWireSequence(),
		ItemID:      encoder.outputIDs[key],
		OutputIndex: responsesOutputIndex(encoder.outputIndexes[key]),
		Delta:       event.ToolCall.Arguments,
	}
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return frames, nil, err
	}
	return append(frames, frame), nil, nil
}

func (encoder *responsesStreamEncoder) responsesCompletionWire(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	if event.Usage == nil {
		return responsesEventWire{}, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
	}
	encoder.terminal = true
	response := newResponsesResponseWire(event.ResponseID, event.Model, "completed", 0, encoder.context.PreviousResponseID)
	output, err := encoder.responsesCompletedOutput()
	if err != nil {
		return responsesEventWire{}, nil, err
	}
	response.Output = output
	response.Usage = encodeResponsesUsage(*event.Usage)
	wire := responsesEventWire{
		Type:     "response.completed",
		Response: &response,
	}
	switch event.StopReason {
	case llmprotocol.StopMaxTokens, llmprotocol.StopContextWindow, llmprotocol.StopContentFilter:
		wire.Type = "response.incomplete"
		wire.Response.Status = "incomplete"
		reason := "max_output_tokens"
		if event.StopReason == llmprotocol.StopContentFilter {
			reason = "content_filter"
		}
		wire.Response.IncompleteDetails = &struct {
			Reason string `json:"reason"`
		}{Reason: reason}
		if event.StopReason == llmprotocol.StopContextWindow {
			var diagnostics llmprotocol.Diagnostics
			appendContextWindowStop(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target)
			return wire, diagnostics, nil
		}
	case llmprotocol.StopPaused, llmprotocol.StopCanceled, llmprotocol.StopUnknown:
		var diagnostics llmprotocol.Diagnostics
		if err := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "response.stop_reason", "Responses cannot represent the source terminal reason"); err != nil {
			return responsesEventWire{}, diagnostics, err
		}
		return wire, diagnostics, nil
	}
	return wire, nil, nil
}

func (encoder *responsesStreamEncoder) responsesFailureWire(
	event llmprotocol.Event,
) (responsesEventWire, llmprotocol.Diagnostics, error) {
	if event.Error == nil {
		return responsesEventWire{}, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	response := newResponsesResponseWire(event.ResponseID, event.Model, "failed", 0, encoder.context.PreviousResponseID)
	output, err := encoder.responsesCompletedOutput()
	if err != nil {
		return responsesEventWire{}, nil, err
	}
	response.Output = output
	response.Error = &responsesErrorWire{Code: responsesErrorCode(event.Error), Message: event.Error.Message}
	if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
		response.Usage = encodeResponsesUsage(*event.Usage)
	}
	return responsesEventWire{
		Type:     "response.failed",
		Response: &response,
	}, nil, nil
}

// encodeResponsesTruncation ends a turn the Router cut as the Responses
// resource for a response that stopped early.
//
// response.incomplete with incomplete_details is what the protocol defines for
// exactly this, and unlike the error event it already states usage, so the
// count the Router handed over reaches the client without a member being
// invented for it. The reason is "max_output_tokens": one of the two values
// this repo's own provider validation accepts (validateResponsesIncompleteDetails)
// and one of the four the OpenAI SDKs narrow the field to, so an SDK reading it
// sees a value it knows. There is no enum value for a router deadline, and
// coining one would be a string the SDKs' Literal types do not contain.
//
// A turn cut before anything was generated has no count to state. It still
// terminates as a resource the client can close, rather than as a usage of
// zeros that would bill a turn that produced nothing.
func (encoder *responsesStreamEncoder) encodeResponsesTruncation(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	truncated := event
	if truncated.StopReason == llmprotocol.StopError || truncated.StopReason == "" {
		truncated.StopReason = llmprotocol.StopMaxTokens
	}
	if !truncationUsageAvailable(truncated.Usage) {
		// responsesCompletionWire requires a usage to render the resource's
		// own member; an unavailable one renders as stated-but-unknown, which
		// is what this turn's count is.
		truncated.Usage = &llmprotocol.Usage{State: llmprotocol.UsageUnavailable}
	}
	// A search the upstream asked for but never answered is still part of what
	// the turn did, and the completion path flushes those before its terminal.
	frames, err := encoder.flushPendingAnthropicSearches(truncated)
	if err != nil {
		return nil, nil, err
	}
	// The item the cut landed in is still open. This terminal follows the
	// completion lifecycle, where an active item is a lifecycle error the
	// decoder refuses (stream_item_incomplete), and the resource's output is
	// assembled only from items that finished -- so an item left open both
	// breaks the round trip and drops the text the client was already streamed.
	// It is closed here, as the Messages encoder closes its open blocks.
	closed, err := encoder.closeOpenResponsesItems(truncated)
	if err != nil {
		return frames, nil, err
	}
	frames = append(frames, closed...)
	wire, diagnostics, err := encoder.responsesCompletionWire(truncated)
	if err != nil || wire.Type == "" {
		return frames, diagnostics, err
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return frames, diagnostics, err
	}
	return append(frames, frame), diagnostics, nil
}

// recordOpenToolCall keeps a started function call as it streams. The deltas
// pass straight through, so without this the encoder could not state the call
// it had opened if the turn ended before the call did.
func (encoder *responsesStreamEncoder) recordOpenToolCall(
	key responsesOutputKey,
	call *llmprotocol.ToolCall,
) {
	if call == nil {
		return
	}
	open := encoder.openToolCalls[key]
	if open == nil {
		open = &llmprotocol.ToolCall{ID: call.ID, Name: call.Name, Namespace: call.Namespace}
		encoder.openToolCalls[key] = open
	}
	if call.ID != "" {
		open.ID = call.ID
	}
	if call.Name != "" {
		open.Name = call.Name
	}
	open.Arguments += call.Arguments
}

// closeOpenResponsesItems finishes every output item this stream started and
// has not completed, in the order they were started, so the terminal states a
// resource whose lifecycle is whole.
//
// Items are closed in the order their wire outputs were assigned, which is the
// order the client saw them start, rather than in neutral item order: the
// resource's output is assembled by wire index, and a client reading the
// closures reads them against the starts they answer.
func (encoder *responsesStreamEncoder) closeOpenResponsesItems(
	event llmprotocol.Event,
) ([][]byte, error) {
	encoder.closingAtCut = true
	defer func() { encoder.closingAtCut = false }()
	type openItem struct {
		itemIndex, outputIndex int
	}
	open := make([]openItem, 0, len(encoder.itemOutputKeys))
	for itemIndex, keys := range encoder.itemOutputKeys {
		for _, key := range keys {
			if _, done := encoder.completedOutput[encoder.outputIndexes[key]]; done {
				continue
			}
			// An image generation or a provider's web search closes only with
			// state the upstream sends on its completion -- the image, the
			// search's result -- and a cut turn has none to synthesize. Such
			// an item stays open rather than closing as something it is not,
			// and the terminal still states the count; a client reading the
			// resource sees the item absent from output, as it was before.
			if key.kind == responsesOutputImage || key.kind == responsesOutputWebSearch {
				continue
			}
			open = append(open, openItem{itemIndex: itemIndex, outputIndex: encoder.outputIndexes[key]})
			break
		}
	}
	sort.Slice(open, func(left, right int) bool {
		return open[left].outputIndex < open[right].outputIndex
	})
	var frames [][]byte
	for _, item := range open {
		closing := event
		closing.Type = llmprotocol.EventOutputItemCompleted
		closing.ItemIndex = item.itemIndex
		// A function call closes as the call it had streamed. The completion
		// event the engine synthesized carries no call of its own, and without
		// one the tool encoder refuses the item -- which left the turn with no
		// terminal at all, worse than the open item this exists to fix. The
		// call is marked incomplete, because the turn ended in the middle of
		// its arguments: a caller that read them as whole would parse a
		// fragment as an argument object.
		if call := encoder.openToolCallFor(item.itemIndex); call != nil {
			cut := *call
			cut.Incomplete = true
			closing.ToolCall = &cut
		}
		// The item is closed as the turn ended, not as a completed answer.
		closed, _, err := encoder.encodeCompletedResponsesItem(closing)
		if err != nil {
			return frames, err
		}
		frames = append(frames, closed...)
	}
	return frames, nil
}

// closedItemStatus is the status an item closes with: "incomplete" while the
// items a cut landed in are being closed, since the turn ended in the middle
// of each, and "completed" for an item the model finished.
func (encoder *responsesStreamEncoder) closedItemStatus() string {
	if encoder.closingAtCut {
		return "incomplete"
	}
	return "completed"
}

// openToolCallFor is the function call an item had streamed, if it is one.
func (encoder *responsesStreamEncoder) openToolCallFor(itemIndex int) *llmprotocol.ToolCall {
	for _, key := range encoder.itemOutputKeys[itemIndex] {
		if call := encoder.openToolCalls[key]; call != nil {
			return call
		}
	}
	return nil
}

func (encoder *responsesStreamEncoder) encodeResponsesTransportFailure(event llmprotocol.Event) ([][]byte, error) {
	if event.Error == nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
	}
	encoder.terminal = true
	frame, err := encoder.encodeTransportError(event.Error)
	return [][]byte{frame}, err
}

func (encoder *responsesStreamEncoder) encodeResponsesReasoningDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var diagnostics llmprotocol.Diagnostics
	// The Responses event model has no signature delta; the signature rides
	// the finished reasoning item (encodeCompletedResponsesContent).
	encoder.recordReasoningSignature(event)
	if frames, handled, err := encoder.stashResponsesReasoningDetails(event); handled || err != nil {
		return frames, diagnostics, err
	}
	key := contentKey(event)
	encoder.encodedKinds[key] = llmprotocol.ContentReasoning
	reasoningScope := eventReasoningScope(event)
	if existing := encoder.reasoningScopes[key]; existing != "" && existing != reasoningScope {
		return nil, diagnostics, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_reasoning_scope_mismatch",
			"upstream stream changed a reasoning content scope",
			nil,
		)
	}
	encoder.reasoningScopes[key] = reasoningScope
	builder := encoder.contentText[key]
	if builder == nil {
		builder = &strings.Builder{}
		encoder.contentText[key] = builder
	}
	builder.WriteString(event.Delta)
	frames, outputKey, err := encoder.ensureResponsesOutputStarted(event, responsesOutputReasoning)
	if err != nil {
		return nil, diagnostics, err
	}
	started, err := encoder.startResponsesContent(event, outputKey, llmprotocol.ContentReasoning)
	if err != nil {
		return nil, diagnostics, err
	}
	frames = append(frames, started...)
	if event.Delta == "" {
		return frames, diagnostics, nil
	}
	scope := responsesScopeForReasoning(reasoningScope)
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, scope)
	wire := responsesEventWire{
		Type: responsesReasoningDeltaType(reasoningScope), Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
		Delta: event.Delta,
	}
	setResponsesReasoningIndex(&wire, reasoningScope, contentIndex)
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return frames, diagnostics, err
	}
	return append(frames, frame), diagnostics, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesOpaque(event llmprotocol.Event) ([][]byte, error) {
	if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
	}
	return [][]byte{append([]byte(nil), event.Opaque...)}, nil
}

func (encoder *responsesStreamEncoder) encodeResponsesTextDelta(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	content, err := encoder.recordResponsesTextDelta(event)
	if err != nil {
		return nil, nil, err
	}
	frames, outputKey, err := encoder.ensureResponsesOutputStarted(event, responsesOutputMessage)
	if err != nil {
		return nil, nil, err
	}
	contentIndex := encoder.responsesContentWireIndex(event, outputKey, responsesContentMessage)
	started, err := encoder.startResponsesContent(event, outputKey, content.kind)
	if err != nil {
		return nil, nil, err
	}
	frames = append(frames, started...)
	frames, err = encoder.appendResponsesTextFrame(frames, outputKey, contentIndex, content.kind, event.Delta)
	if err != nil {
		return nil, nil, err
	}
	frames, err = encoder.appendResponsesAnnotations(frames, outputKey, contentIndex, content)
	if err != nil {
		return nil, nil, err
	}
	return frames, nil, nil
}

type responsesRecordedTextDelta struct {
	kind           llmprotocol.ContentKind
	newCitations   []llmprotocol.Citation
	annotationBase int
}

func (encoder *responsesStreamEncoder) recordResponsesTextDelta(
	event llmprotocol.Event,
) (responsesRecordedTextDelta, error) {
	kind := llmprotocol.ContentText
	if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
		kind = llmprotocol.ContentRefusal
	}
	key := contentKey(event)
	if existing := encoder.encodedKinds[key]; existing != "" && existing != kind {
		return responsesRecordedTextDelta{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable, "stream_content_kind_mismatch", "upstream stream changed a content block kind", nil,
		)
	}
	encoder.encodedKinds[key] = kind
	builder := encoder.contentText[key]
	if builder == nil {
		builder = &strings.Builder{}
		encoder.contentText[key] = builder
	}
	builder.WriteString(event.Delta)
	recorded := responsesRecordedTextDelta{kind: kind, annotationBase: len(encoder.contentCitations[key])}
	if event.Content != nil && len(event.Content.Citations) > 0 {
		recorded.newCitations = event.Content.Citations
		encoder.contentCitations[key] = append(encoder.contentCitations[key], recorded.newCitations...)
	}
	return recorded, nil
}

func (encoder *responsesStreamEncoder) appendResponsesTextFrame(
	frames [][]byte,
	outputKey responsesOutputKey,
	contentIndex int,
	kind llmprotocol.ContentKind,
	delta string,
) ([][]byte, error) {
	if delta == "" {
		return frames, nil
	}
	wire := responsesEventWire{
		Type: responsesTextDeltaType(kind), Sequence: encoder.nextWireSequence(),
		ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]),
		ContentIndex: responsesContentIndex(contentIndex), Delta: delta,
	}
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	if err != nil {
		return frames, err
	}
	return append(frames, frame), nil
}

func (encoder *responsesStreamEncoder) appendResponsesAnnotations(
	frames [][]byte,
	outputKey responsesOutputKey,
	contentIndex int,
	content responsesRecordedTextDelta,
) ([][]byte, error) {
	for offset, annotation := range encodeResponsesAnnotations(content.newCitations) {
		index := content.annotationBase + offset
		wire := responsesEventWire{
			Type: "response.output_text.annotation.added", Sequence: encoder.nextWireSequence(),
			ItemID: encoder.outputIDs[outputKey], OutputIndex: responsesOutputIndex(encoder.outputIndexes[outputKey]), ContentIndex: responsesContentIndex(contentIndex),
			AnnotationIndex: &index, Annotation: &annotation,
		}
		frame, err := encoder.encodeResponsesStreamFrame(wire)
		if err != nil {
			return nil, err
		}
		frames = append(frames, frame)
	}
	return frames, nil
}

func (encoder *responsesStreamEncoder) responsesContentWireIndex(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
	scope responsesContentScope,
) int {
	key := contentKey(event)
	if index, found := encoder.contentIndexes[key]; found {
		return index
	}
	sequence := responsesOutputContentKey{output: outputKey, scope: scope}
	index := encoder.nextContentIndex[sequence]
	encoder.nextContentIndex[sequence] = index + 1
	encoder.contentIndexes[key] = index
	return index
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesItem(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if carried := carriedAnthropicServerBlock(event.Content); carried != nil {
		if carried.Type == "redacted_thinking" {
			return encoder.encodeResponsesRedactedThinking(event)
		}
		return encoder.encodeResponsesAnthropicWebSearch(event, carried)
	}
	var frames [][]byte
	keys := append([]responsesOutputKey(nil), encoder.itemOutputKeys[event.ItemIndex]...)
	if len(keys) == 0 {
		kind := responsesCompletionOutputKind(event)
		started, key, err := encoder.ensureResponsesOutputStarted(event, kind)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, started...)
		keys = append(keys, key)
	}
	for _, key := range keys {
		completed, diagnostics, err := encoder.encodeCompletedResponsesOutput(event, key)
		if err != nil {
			return nil, diagnostics, err
		}
		frames = append(frames, completed...)
	}
	return frames, nil, nil
}

func responsesCompletionOutputKind(event llmprotocol.Event) responsesOutputKind {
	if event.ToolCall != nil {
		return responsesOutputTool
	}
	if event.Content == nil {
		return responsesOutputMessage
	}
	if _, webSearch := carriedWebSearchCall(*event.Content); webSearch {
		return responsesOutputWebSearch
	}
	switch event.Content.Kind {
	case llmprotocol.ContentReasoning:
		return responsesOutputReasoning
	case llmprotocol.ContentGeneratedImage:
		return responsesOutputImage
	default:
		return responsesOutputMessage
	}
}

// encodeCompletedResponsesWebSearch writes a carried web_search_call item's
// done event, as the provider completed it, under this stream's output index
// and item ID.
func (encoder *responsesStreamEncoder) encodeCompletedResponsesWebSearch(
	event llmprotocol.Event,
	key responsesOutputKey,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var item responsesItemWire
	if event.Content != nil {
		item, _ = carriedWebSearchCall(*event.Content)
	}
	index := encoder.outputIndexes[key]
	item.Type, item.ID = "web_search_call", encoder.outputIDs[key]
	if item.Status == "" {
		item.Status = "completed"
	}
	wire := responsesEventWire{
		Type: "response.output_item.done", Sequence: encoder.nextWireSequence(),
		OutputIndex: responsesOutputIndex(index), Item: marshalResponsesEventItem(item),
	}
	encoder.recordResponsesCompletedOutput(index, wire.Item)
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return [][]byte{frame}, nil, err
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesOutput(
	event llmprotocol.Event,
	key responsesOutputKey,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if key.kind == responsesOutputMessage || key.kind == responsesOutputReasoning {
		return encoder.encodeCompletedResponsesContent(event, key)
	}
	if key.kind == responsesOutputImage {
		return encoder.encodeCompletedResponsesImage(event, key)
	}
	if key.kind == responsesOutputWebSearch {
		return encoder.encodeCompletedResponsesWebSearch(event, key)
	}
	if event.ToolCall == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
	}
	index, id := encoder.outputIndexes[key], encoder.outputIDs[key]
	done := responsesEventWire{
		Type: "response.function_call_arguments.done", Sequence: encoder.nextWireSequence(),
		ItemID: id, OutputIndex: responsesOutputIndex(index),
		Name: event.ToolCall.Name, Arguments: event.ToolCall.Arguments,
	}
	doneFrame, err := encoder.encodeResponsesStreamFrame(done)
	if err != nil {
		return nil, nil, err
	}
	wire := responsesEventWire{
		Type: "response.output_item.done", Sequence: encoder.nextWireSequence(), OutputIndex: responsesOutputIndex(index),
	}
	item := responsesItemWire{
		Type: "function_call", ID: id, Status: responsesToolCallStatus(event.ToolCall),
		CallID: event.ToolCall.ID, Name: event.ToolCall.Name, Arguments: event.ToolCall.Arguments,
		Namespace: event.ToolCall.Namespace,
	}
	wire.Item = marshalResponsesEventItem(item)
	encoder.recordResponsesCompletedOutput(index, wire.Item)
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return [][]byte{doneFrame, frame}, nil, err
}

// responsesToolCallStatus is a function_call item's final status. A call the
// model was cut off writing at max_output_tokens ends "incomplete" with the
// arguments it wrote, which is how OpenAI ends an item the limit interrupted;
// the response around it ends response.incomplete with reason
// max_output_tokens. "completed" would tell the client the call is whole.
func responsesToolCallStatus(call *llmprotocol.ToolCall) string {
	if call != nil && call.Incomplete {
		return "incomplete"
	}
	return "completed"
}

func marshalResponsesEventItem(item responsesItemWire) json.RawMessage {
	body, _ := json.Marshal(item)
	return body
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesContent(
	event llmprotocol.Event,
	outputKey responsesOutputKey,
) ([][]byte, llmprotocol.Diagnostics, error) {
	parts, frames, err := encoder.collectCompletedResponsesContent(event, outputKey)
	if err != nil {
		return nil, nil, err
	}
	index, id := encoder.outputIndexes[outputKey], encoder.outputIDs[outputKey]
	item, err := marshalCompletedResponsesItem(outputKey, id, parts, encoder.closedItemStatus())
	if err != nil {
		return nil, nil, err
	}
	if outputKey.kind == responsesOutputReasoning && event.Content != nil {
		if encrypted, _ := encryptedReasoningOf(*event.Content); encrypted != nil {
			item.EncryptedContent = encrypted
		}
	}
	if outputKey.kind == responsesOutputReasoning && len(item.EncryptedContent) == 0 {
		if details := encoder.reasoningDetails[outputKey]; details != nil {
			item.EncryptedContent = mintReasoningDetails(details)
		}
	}
	if signature := encoder.reasoningSignatures[event.ItemIndex]; outputKey.kind == responsesOutputReasoning && signature != "" {
		item.Signature, item.ReasoningFormat = signature, responsesAnthropicReasoningFormat
	}
	wire := responsesEventWire{
		Type: "response.output_item.done", Sequence: encoder.nextWireSequence(),
		OutputIndex: responsesOutputIndex(index), Item: marshalResponsesEventItem(item),
	}
	encoder.recordResponsesCompletedOutput(index, wire.Item)
	done, err := encoder.encodeResponsesStreamFrame(wire)
	return append(frames, done), nil, err
}

// stashResponsesReasoningDetails keeps the reasoning_details fragments an
// OpenRouter Chat stream sends, merged per reasoning output, so the output's
// done event can give a Responses client the whole array as its
// encrypted_content. A fragment with no reasoning text beside it opens the
// reasoning output and nothing else: handled reports that the event needs no
// further encoding.
func (encoder *responsesStreamEncoder) stashResponsesReasoningDetails(
	event llmprotocol.Event,
) ([][]byte, bool, error) {
	if event.Content == nil {
		return nil, false, nil
	}
	fragment, _ := reasoningDetailsOf(*event.Content)
	if fragment == nil {
		return nil, false, nil
	}
	outputKey := responsesOutputKey{item: event.ItemIndex, kind: responsesOutputReasoning}
	if encoder.reasoningDetails == nil {
		encoder.reasoningDetails = make(map[responsesOutputKey]json.RawMessage)
	}
	merged, err := mergeReasoningDetailsFragment(encoder.reasoningDetails[outputKey], fragment)
	if err != nil {
		return nil, true, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable, "invalid_reasoning_details",
			"upstream stream sent malformed reasoning_details", err,
		)
	}
	encoder.reasoningDetails[outputKey] = merged
	if event.Delta != "" {
		return nil, false, nil
	}
	// The content position is claimed for reasoning, so completion closes it
	// as reasoning (as it does an encrypted-only Responses reasoning item),
	// but no part is opened until text arrives.
	key := contentKey(event)
	encoder.encodedKinds[key] = llmprotocol.ContentReasoning
	if encoder.reasoningScopes[key] == "" {
		encoder.reasoningScopes[key] = eventReasoningScope(event)
	}
	frames, _, err := encoder.ensureResponsesOutputStarted(event, responsesOutputReasoning)
	return frames, true, err
}
