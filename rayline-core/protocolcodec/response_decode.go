/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package protocolcodec

import (
	"fmt"
	"sort"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// DecodeResponse decodes one buffered wire response into the neutral response
// contract without encoding a second provider representation.
func (engine *Engine) DecodeResponse(
	format llmprotocol.WireFormat,
	body []byte,
) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, err
	}
	response, envelope, diagnostics, err := pair.buffered.DecodeResponse(body, engine.policy)
	if err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, diagnostics, err
	}
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return response, envelope, diagnostics, err
	}
	return response, envelope, diagnostics, nil
}

// DecodeResponseStream decodes a complete wire stream into one neutral
// terminal response. It is intended for buffered orchestration such as Looper;
// forwarding paths should keep using StreamEngine to preserve backpressure.
func (engine *Engine) DecodeResponseStream(
	format llmprotocol.WireFormat,
	body []byte,
	context llmprotocol.StreamContext,
) (llmprotocol.Response, llmprotocol.Diagnostics, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return llmprotocol.Response{}, nil, err
	}
	if !pair.buffered.Capabilities().Supports(llmprotocol.CapabilityStreaming) {
		return llmprotocol.Response{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"streaming_unsupported",
			"wire format does not support streaming",
			nil,
		)
	}
	context.Source = format
	context.Target = format
	decoder := pair.stream.NewDecoder(context, engine.providerStreamPolicy())
	accumulator := NewResponseAccumulator()
	events, diagnostics, err := decoder.Push(body)
	if applyErr := accumulator.Apply(events); applyErr != nil {
		return llmprotocol.Response{}, diagnostics, applyErr
	}
	if err != nil {
		return llmprotocol.Response{}, diagnostics, err
	}
	finalEvents, finalDiagnostics, err := decoder.Finalize(nil)
	diagnostics = appendDiagnostics(diagnostics, finalDiagnostics, engine.policy.Limits.Diagnostics)
	if applyErr := accumulator.Apply(finalEvents); applyErr != nil {
		return llmprotocol.Response{}, diagnostics, applyErr
	}
	if err != nil {
		return llmprotocol.Response{}, diagnostics, err
	}
	response, err := accumulator.Response()
	if err != nil {
		return llmprotocol.Response{}, diagnostics, err
	}
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return response, diagnostics, err
	}
	return response, diagnostics, nil
}

// ResponseAccumulator collects semantic stream events without retaining wire frames.
// Call Response only after the decoder and stream have completed successfully.
type ResponseAccumulator struct {
	result   llmprotocol.Response
	items    map[int]*responseAccumulatorItem
	terminal bool
}

type responseAccumulatorItem struct {
	id       string
	role     llmprotocol.Role
	contents map[int]*llmprotocol.Content
	call     *llmprotocol.ToolCall
}

// NewResponseAccumulator creates an empty response accumulator.
func NewResponseAccumulator() *ResponseAccumulator {
	return &ResponseAccumulator{
		result: llmprotocol.Response{
			Generation: 1,
			Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		},
		items: make(map[int]*responseAccumulatorItem),
	}
}

// Apply consumes decoded semantic events in order.
func (accumulator *ResponseAccumulator) Apply(events []llmprotocol.Event) error {
	for _, event := range events {
		if err := accumulator.applyEvent(event); err != nil {
			return err
		}
	}
	return nil
}

func (accumulator *ResponseAccumulator) applyEvent(event llmprotocol.Event) error {
	if accumulator.terminal {
		return llmprotocol.NewError(
			llmprotocol.ErrorConflict,
			"stream_terminal",
			"stream emitted an event after its terminal",
			nil,
		)
	}
	accumulator.applyEventMetadata(event)
	if responseAccumulatorItemEvent(event.Type) {
		return accumulator.applyItemEvent(event)
	}
	switch event.Type {
	case llmprotocol.EventResponseStarted, llmprotocol.EventUsageUpdated, llmprotocol.EventKeepalive:
		return nil
	case llmprotocol.EventResponseCompleted:
		return accumulator.completeResponse(event)
	case llmprotocol.EventResponseFailed:
		return accumulator.failResponse(event)
	case llmprotocol.EventProviderOpaque:
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"opaque_stream_event",
			"provider-opaque stream events cannot enter the neutral response contract",
			nil,
		)
	default:
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unknown_stream_event",
			"stream emitted an unknown semantic event",
			nil,
		)
	}
}

func responseAccumulatorItemEvent(eventType llmprotocol.EventType) bool {
	switch eventType {
	case llmprotocol.EventOutputItemStarted, llmprotocol.EventOutputTextDelta,
		llmprotocol.EventReasoningDelta, llmprotocol.EventToolCallDelta,
		llmprotocol.EventImageGenerationProgress, llmprotocol.EventOutputItemCompleted:
		return true
	default:
		return false
	}
}

func (accumulator *ResponseAccumulator) applyItemEvent(event llmprotocol.Event) error {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		return accumulator.startItem(event)
	case llmprotocol.EventOutputTextDelta:
		return accumulator.appendOutputText(event)
	case llmprotocol.EventReasoningDelta:
		return accumulator.appendReasoning(event)
	case llmprotocol.EventToolCallDelta:
		return accumulator.appendToolCall(event.ItemIndex, event.ToolCall)
	case llmprotocol.EventImageGenerationProgress:
		return nil
	case llmprotocol.EventOutputItemCompleted:
		return accumulator.completeItem(event)
	default:
		return llmprotocol.NewError(llmprotocol.ErrorInternal, "invalid_accumulator_event", "response accumulator event is invalid", nil)
	}
}

func (accumulator *ResponseAccumulator) completeResponse(event llmprotocol.Event) error {
	accumulator.result.StopReason = event.StopReason
	accumulator.result.MatchedStopSequence = event.MatchedStopSequence
	accumulator.terminal = true
	return nil
}

func (accumulator *ResponseAccumulator) failResponse(event llmprotocol.Event) error {
	accumulator.result.Output = nil
	accumulator.result.StopReason = llmprotocol.StopError
	accumulator.result.Error = event.Error
	accumulator.terminal = true
	return nil
}

func (accumulator *ResponseAccumulator) applyEventMetadata(event llmprotocol.Event) {
	if event.ResponseID != "" {
		accumulator.result.ID = event.ResponseID
	}
	if event.Model != "" {
		accumulator.result.Model = event.Model
	}
	if event.Usage != nil {
		accumulator.result.Usage = *event.Usage
	}
	if event.ProviderStopReason != "" {
		accumulator.result.ProviderStopReason = event.ProviderStopReason
	}
	if event.StopDetails != nil {
		accumulator.result.StopDetails = event.StopDetails
	}
}

func (accumulator *ResponseAccumulator) startItem(event llmprotocol.Event) error {
	if _, found := accumulator.items[event.ItemIndex]; found {
		return fmt.Errorf("neutral stream item %d started twice", event.ItemIndex)
	}
	item := &responseAccumulatorItem{id: event.ItemID, role: event.Role, contents: make(map[int]*llmprotocol.Content)}
	if item.role == "" {
		item.role = llmprotocol.RoleAssistant
	}
	if event.ToolCall != nil {
		call := *event.ToolCall
		item.call = &call
	} else if event.Content != nil {
		content := *event.Content
		item.contents[event.ContentIndex] = &content
	}
	accumulator.items[event.ItemIndex] = item
	return nil
}

func (accumulator *ResponseAccumulator) appendOutputText(event llmprotocol.Event) error {
	kind := llmprotocol.ContentText
	var citations []llmprotocol.Citation
	if event.Content != nil && event.Content.Kind != "" {
		kind = event.Content.Kind
		citations = event.Content.Citations
	}
	return accumulator.appendText(event.ItemIndex, event.ContentIndex, kind, event.Delta, "", "", citations)
}

func (accumulator *ResponseAccumulator) appendReasoning(event llmprotocol.Event) error {
	signature := ""
	reasoning := llmprotocol.ReasoningScope("")
	if event.Content != nil {
		signature = event.Content.Signature
		reasoning = event.Content.Reasoning
	}
	if err := accumulator.appendText(
		event.ItemIndex, event.ContentIndex, llmprotocol.ContentReasoning, event.Delta, signature, reasoning, nil,
	); err != nil {
		return err
	}
	if event.Content == nil {
		return nil
	}
	return AccumulateReasoningDetails(accumulator.items[event.ItemIndex].contents[event.ContentIndex], *event.Content)
}

func (accumulator *ResponseAccumulator) item(index int) (*responseAccumulatorItem, error) {
	item := accumulator.items[index]
	if item == nil {
		return nil, fmt.Errorf("neutral stream item %d is unavailable", index)
	}
	return item, nil
}

func (accumulator *ResponseAccumulator) appendText(
	index int,
	contentIndex int,
	kind llmprotocol.ContentKind,
	text string,
	signature string,
	reasoning llmprotocol.ReasoningScope,
	citations []llmprotocol.Citation,
) error {
	item, err := accumulator.item(index)
	if err != nil {
		return err
	}
	content := item.contents[contentIndex]
	if content == nil {
		content = &llmprotocol.Content{Kind: kind}
		item.contents[contentIndex] = content
	} else if content.Kind != kind {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_content_kind_mismatch",
			"upstream stream changed a content block kind",
			nil,
		)
	}
	content.Text += text
	if signature != "" {
		content.Signature = signature
	}
	if reasoning != "" {
		if content.Reasoning != "" && content.Reasoning != reasoning {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_reasoning_scope_mismatch",
				"upstream stream changed a reasoning content scope",
				nil,
			)
		}
		content.Reasoning = reasoning
	}
	content.Citations = append(content.Citations, citations...)
	return nil
}

func (accumulator *ResponseAccumulator) appendToolCall(index int, delta *llmprotocol.ToolCall) error {
	if delta == nil {
		return fmt.Errorf("neutral tool-call delta is unavailable")
	}
	item, err := accumulator.item(index)
	if err != nil {
		return err
	}
	if item.call == nil {
		item.call = &llmprotocol.ToolCall{}
	}
	if strings.TrimSpace(item.call.Arguments) == "{}" && delta.Arguments != "" {
		item.call.Arguments = ""
	}
	if delta.ID != "" {
		item.call.ID = delta.ID
	}
	if delta.Name != "" {
		item.call.Name = delta.Name
	}
	item.call.Arguments += delta.Arguments
	return nil
}

func (accumulator *ResponseAccumulator) completeItem(event llmprotocol.Event) error {
	item, err := accumulator.item(event.ItemIndex)
	if err != nil {
		return err
	}
	if event.ItemID != "" {
		item.id = event.ItemID
	}
	if event.Role != "" {
		item.role = event.Role
	}
	if event.ToolCall != nil {
		call := *event.ToolCall
		item.call = &call
	}
	if event.Content != nil && event.Content.Kind != "" &&
		(event.Content.Text != "" || event.Content.Kind == llmprotocol.ContentGeneratedImage) {
		content := *event.Content
		item.contents[event.ContentIndex] = &content
	}
	return nil
}

// Response returns the terminal response, or an error for incomplete input.
// A provider failure is carried by Response.Error and must not be committed.
func (accumulator *ResponseAccumulator) Response() (llmprotocol.Response, error) {
	if !accumulator.terminal {
		return llmprotocol.Response{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_incomplete",
			"upstream stream ended without a terminal response",
			nil,
		)
	}
	if accumulator.result.Error != nil {
		return accumulator.result, nil
	}
	result := accumulator.result
	result.Output = append([]llmprotocol.OutputItem(nil), result.Output...)
	indexes := make([]int, 0, len(accumulator.items))
	for index := range accumulator.items {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		item := accumulator.items[index]
		if len(item.contents) == 0 && item.call == nil {
			// Some wire formats emit a role-only framing item before the first
			// semantic content block. It is transport state, not an empty output.
			continue
		}
		contentIndexes := make([]int, 0, len(item.contents))
		for contentIndex := range item.contents {
			contentIndexes = append(contentIndexes, contentIndex)
		}
		sort.Ints(contentIndexes)
		contents := make([]llmprotocol.Content, 0, len(contentIndexes)+1)
		for _, contentIndex := range contentIndexes {
			contents = append(contents, *item.contents[contentIndex])
		}
		if item.call != nil {
			call := *item.call
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &call})
		}
		result.Output = append(result.Output, llmprotocol.OutputItem{
			ID: item.id, Role: item.role, Content: contents,
		})
	}
	return result, nil
}
