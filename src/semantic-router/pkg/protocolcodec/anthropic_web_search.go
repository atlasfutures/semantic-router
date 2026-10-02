package protocolcodec

import (
	"encoding/json"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Anthropic's web search runs at the provider: the model's server_tool_use
// block names the query, and the web_search_tool_result block that follows
// holds the results. Both are carried whole, so an Anthropic client such as
// Claude Code gets them back as the provider sent them; a Responses client
// gets them as one web_search_call item, and any other client drops them.

// anthropicServerToolBlock reports whether a provider block is part of a
// server-run web search.
func anthropicServerToolBlock(typeName string) bool {
	return typeName == "server_tool_use" || typeName == "web_search_tool_result"
}

// anthropicCarriedResponseBlock reports whether a provider block is carried
// whole rather than modelled: a server web search block, or a
// redacted_thinking block. OpenRouter's Messages surface returns the latter
// for a model whose reasoning is encrypted (gpt-5.x); refusing it failed the
// whole billed turn. An Anthropic client gets it back exactly as sent, so the
// conversation can resend it; any other client has nowhere to show opaque
// reasoning and drops it.
func anthropicCarriedResponseBlock(typeName string) bool {
	return anthropicServerToolBlock(typeName) || typeName == "redacted_thinking"
}

type anthropicWebSearchCitationWire struct {
	Type  string `json:"type"`
	URL   string `json:"url"`
	Title string `json:"title"`
}

// webSearchURLCitations turns a provider text block's citations into URL
// citations for a client of another format. Every citation kind is carried raw
// (CitationsRaw), so an Anthropic client gets them back unchanged; only a kind
// that names a URL -- a web search result location -- has a neutral shape, and
// Anthropic does not say which span of the answer it supports, so each cites
// the whole block. A kind without a URL (a document's char, page or block
// location) has no neutral source and is left to the Anthropic carry; the
// block's text still reaches every client. Refusing them instead failed
// billed provider turns (kimi-k3 on OpenRouter Messages, 2026-10-01).
func webSearchURLCitations(raw json.RawMessage, text string) []llmprotocol.Citation {
	return webSearchURLCitationsTo(raw, int64(utf8.RuneCountInString(text)))
}

// appendUncarriedCitationDiagnostic records, for a client of another format,
// that a provider text block held a citation without a URL: its source has no
// neutral shape, so it reaches only an Anthropic client, and the text is
// delivered without it.
func appendUncarriedCitationDiagnostic(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source, target llmprotocol.WireFormat,
	output []llmprotocol.OutputItem,
) {
	for _, item := range output {
		for index := range item.Content {
			if uncarriedCitation(&item.Content[index]) {
				*diagnostics = appendDiagnostics(*diagnostics, llmprotocol.Diagnostics{
					uncarriedCitationDiagnostic(source, target),
				}, policy.Limits.Diagnostics)
				return
			}
		}
	}
}

// uncarriedCitation reports whether a text part holds a citation without a URL
// form, one a client of another format cannot receive.
func uncarriedCitation(content *llmprotocol.Content) bool {
	if content == nil || content.Kind != llmprotocol.ContentText || len(content.CitationsRaw) == 0 {
		return false
	}
	var raw []json.RawMessage
	return json.Unmarshal(content.CitationsRaw, &raw) == nil && len(raw) > len(content.Citations)
}

func uncarriedCitationDiagnostic(source, target llmprotocol.WireFormat) llmprotocol.Diagnostic {
	return llmprotocol.Diagnostic{
		Source: source, Target: target, Field: fieldContentCitations, Action: llmprotocol.DiagnosticDropped,
		Reason: "a citation without a URL has no neutral source; the text is delivered without it",
	}
}

// validProviderCitations reports whether a provider text block's citations
// member is null or a list of objects that each state a type. Any type name is
// carried, including ones not named yet; anything else is malformed provider
// output, refused as a streamed citation is.
func validProviderCitations(raw json.RawMessage) bool {
	var list []json.RawMessage
	if json.Unmarshal(raw, &list) != nil {
		return false
	}
	for _, citation := range list {
		if !typedCitationObject(citation) {
			return false
		}
	}
	return true
}

func typedCitationObject(citation json.RawMessage) bool {
	var object map[string]json.RawMessage
	var kind struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(citation, &object) == nil && object != nil &&
		json.Unmarshal(citation, &kind) == nil && kind.Type != ""
}

// webSearchURLCitationsTo is webSearchURLCitations for a span ending at end.
func webSearchURLCitationsTo(raw json.RawMessage, end int64) []llmprotocol.Citation {
	var wire []anthropicWebSearchCitationWire
	if json.Unmarshal(raw, &wire) != nil {
		return nil
	}
	citations := make([]llmprotocol.Citation, 0, len(wire))
	for _, citation := range wire {
		// Only a web search result location with an http(s) URL has the
		// neutral citation's shape; any other kind, a future one with a url
		// member included, stays raw, so it can never fail the response.
		if citation.Type != "web_search_result_location" || !httpURL(citation.URL) {
			continue
		}
		citations = append(citations, llmprotocol.Citation{
			URL: citation.URL, Title: citation.Title, StartIndex: 0, EndIndex: end,
		})
	}
	return citations
}

func httpURL(raw string) bool {
	parsed, err := url.Parse(raw)
	// The neutral citation contract takes an absolute http(s) URL without
	// credentials; anything else would fail validation of the whole response.
	return err == nil && (parsed.Scheme == "http" || parsed.Scheme == "https") && parsed.Host != "" &&
		parsed.User == nil
}

// withoutForeignServerToolOutput removes the carried Anthropic web search
// blocks, for a Chat client. An output that held nothing else -- a turn Anthropic
// paused mid-search -- keeps an empty text part, since a Chat choice must say
// something.
func withoutForeignServerToolOutput(output []llmprotocol.OutputItem) []llmprotocol.OutputItem {
	trimmed := make([]llmprotocol.OutputItem, 0, len(output))
	for _, item := range output {
		contents := make([]llmprotocol.Content, 0, len(item.Content))
		dropped := false
		for _, content := range item.Content {
			if block := content.Unmodeled; content.Kind == llmprotocol.ContentUnmodeled && block != nil &&
				block.Format == llmprotocol.AnthropicMessagesV1 && anthropicCarriedResponseBlock(block.Type) {
				dropped = true
				continue
			}
			contents = append(contents, content)
		}
		if dropped && len(contents) == 0 {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentText})
		}
		item.Content = contents
		trimmed = append(trimmed, item)
	}
	return trimmed
}

type anthropicServerToolUseWire struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input struct {
		Query string `json:"query"`
	} `json:"input"`
}

type anthropicWebSearchResultWire struct {
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
}

type anthropicWebSearchOutcome struct {
	sources []map[string]string
	failed  bool
	pending bool
}

// outcome reads a web search result: a list of results, or an error object
// (web_search_tool_result_error, such as max_uses_exceeded), which is a
// failed search.
func (result anthropicWebSearchResultWire) outcome() anthropicWebSearchOutcome {
	var items []struct {
		Type string `json:"type"`
		URL  string `json:"url"`
	}
	if json.Unmarshal(result.Content, &items) != nil {
		return anthropicWebSearchOutcome{failed: true}
	}
	var outcome anthropicWebSearchOutcome
	for _, item := range items {
		if item.Type == "web_search_result" && item.URL != "" {
			outcome.sources = append(outcome.sources, map[string]string{"type": "url", "url": item.URL})
		}
	}
	return outcome
}

// webSearchCallFor is the Responses web_search_call for one Anthropic search
// and its outcome.
func webSearchCallFor(use anthropicServerToolUseWire, outcome anthropicWebSearchOutcome) llmprotocol.Content {
	fields := map[string]any{"type": "search", "query": use.Input.Query}
	if len(outcome.sources) > 0 {
		fields["sources"] = outcome.sources
	}
	action, _ := json.Marshal(fields)
	status := "completed"
	switch {
	case outcome.failed:
		status = "failed"
	case outcome.pending:
		status = "in_progress"
	}
	return webSearchCallContent(responsesItemWire{Type: "web_search_call", ID: use.ID, Status: status, Action: action})
}

// anthropicWebSearchAsResponses rewrites the carried Anthropic web search
// blocks of one output into Responses web_search_call contents: each
// server_tool_use naming web_search becomes one call carrying its query and
// the sources of the result that answers it, failed if the result is an
// error. A search with no result yet -- a turn Anthropic paused mid-search --
// is reported in progress, as the stream reports it at completion. The result
// blocks are consumed.
func anthropicWebSearchAsResponses(contents []llmprotocol.Content) []llmprotocol.Content {
	return rewriteAnthropicWebSearch(contents, anthropicWebSearchOutcomes(contents))
}

// anthropicWebSearchOutputAsResponses is anthropicWebSearchAsResponses over a
// whole response: a search and its result may sit in different output items
// -- a response rebuilt from a stream keeps one item per content block -- so
// results are matched across every item before any is rewritten.
func anthropicWebSearchOutputAsResponses(output []llmprotocol.OutputItem) []llmprotocol.OutputItem {
	var all []llmprotocol.Content
	for _, item := range output {
		all = append(all, item.Content...)
	}
	outcomes := anthropicWebSearchOutcomes(all)
	rewritten := make([]llmprotocol.OutputItem, 0, len(output))
	for _, item := range output {
		item.Content = rewriteAnthropicWebSearch(item.Content, outcomes)
		if len(item.Content) == 0 {
			continue
		}
		rewritten = append(rewritten, item)
	}
	return rewritten
}

func anthropicWebSearchOutcomes(contents []llmprotocol.Content) map[string]anthropicWebSearchOutcome {
	outcomes := map[string]anthropicWebSearchOutcome{}
	for _, content := range contents {
		block := content.Unmodeled
		if content.Kind != llmprotocol.ContentUnmodeled || block == nil ||
			block.Format != llmprotocol.AnthropicMessagesV1 || block.Type != "web_search_tool_result" {
			continue
		}
		var result anthropicWebSearchResultWire
		if json.Unmarshal(block.Raw, &result) != nil {
			continue
		}
		outcomes[result.ToolUseID] = result.outcome()
	}
	return outcomes
}

func rewriteAnthropicWebSearch(contents []llmprotocol.Content, outcomes map[string]anthropicWebSearchOutcome) []llmprotocol.Content {
	rewritten := make([]llmprotocol.Content, 0, len(contents))
	for _, content := range contents {
		block := content.Unmodeled
		if content.Kind != llmprotocol.ContentUnmodeled || block == nil || block.Format != llmprotocol.AnthropicMessagesV1 {
			rewritten = append(rewritten, content)
			continue
		}
		switch block.Type {
		case "web_search_tool_result":
			continue
		case "server_tool_use":
			var use anthropicServerToolUseWire
			if json.Unmarshal(block.Raw, &use) != nil || use.Name != "web_search" {
				continue
			}
			outcome, answered := outcomes[use.ID]
			if !answered {
				outcome.pending = true
			}
			rewritten = append(rewritten, webSearchCallFor(use, outcome))
		default:
			rewritten = append(rewritten, content)
		}
	}
	return rewritten
}

type anthropicServerBlock struct {
	block map[string]json.RawMessage
	input strings.Builder
}

// decodeAnthropicWebSearchEvent handles the stream events of a server web
// search: a server_tool_use or web_search_tool_result block starts an item
// holding the block, a server_tool_use's input_json_delta fragments are
// collected, and the block's stop completes the item with the whole block. A
// text block's web search citation is a text delta that carries it. Every
// other event is left to the caller.
func (decoder *anthropicStreamDecoder) decodeAnthropicWebSearchEvent(
	wire anthropicEventWire,
) ([]llmprotocol.Event, bool, error) {
	index := anthropicEventIndex(wire)
	switch wire.Type {
	case "content_block_start":
		if wire.ContentBlock == nil || !anthropicCarriedResponseBlock(wire.ContentBlock.Type) {
			return nil, false, nil
		}
		var frame struct {
			ContentBlock map[string]json.RawMessage `json:"content_block"`
		}
		if err := json.Unmarshal(decoder.data, &frame); err != nil || frame.ContentBlock == nil {
			return nil, true, invalidProviderResponse("invalid_stream_block", "Anthropic web search block is invalid")
		}
		if decoder.serverBlocks == nil {
			decoder.serverBlocks = map[int]*anthropicServerBlock{}
		}
		pending := &anthropicServerBlock{block: frame.ContentBlock}
		decoder.serverBlocks[index] = pending
		started := llmprotocol.Event{
			Type: llmprotocol.EventOutputItemStarted, ItemIndex: index, Role: llmprotocol.RoleAssistant,
			ItemID: wire.ContentBlock.ID, Content: pending.content(wire.ContentBlock.Type),
		}
		events, _, err := decoder.emitAnthropicEvent(started)
		return events, true, err
	case "content_block_delta":
		if pending := decoder.serverBlocks[index]; pending != nil {
			if wire.Delta == nil || wire.Delta.Type != "input_json_delta" {
				return nil, true, invalidProviderResponse("invalid_stream_delta", "Anthropic web search block streamed an unexpected delta")
			}
			pending.input.WriteString(wire.Delta.PartialJSON)
			return nil, true, nil
		}
		if wire.Delta != nil && wire.Delta.Type == "citations_delta" {
			return decoder.holdAnthropicCitation(index, wire.Delta.Citation)
		}
		return nil, false, nil
	case "content_block_stop":
		pending := decoder.serverBlocks[index]
		if pending == nil {
			return decoder.releaseAnthropicCitations(index)
		}
		delete(decoder.serverBlocks, index)
		if input := strings.TrimSpace(pending.input.String()); input != "" {
			if !json.Valid([]byte(input)) {
				return nil, true, invalidProviderResponse("invalid_stream_tool_arguments", "Anthropic web search input is not JSON")
			}
			pending.block["input"] = json.RawMessage(input)
		}
		var blockType string
		_ = json.Unmarshal(pending.block["type"], &blockType)
		completed := llmprotocol.Event{
			Type: llmprotocol.EventOutputItemCompleted, ItemIndex: index, Content: pending.content(blockType),
		}
		events, _, err := decoder.emitAnthropicEvent(completed)
		return events, true, err
	}
	return nil, false, nil
}

func (pending *anthropicServerBlock) content(blockType string) *llmprotocol.Content {
	raw, _ := json.Marshal(pending.block)
	content := carriedAnthropicBlock(blockType, raw)
	return &content
}

func (decoder *anthropicStreamDecoder) countText(index int, text string) {
	if decoder.textRunes == nil {
		decoder.textRunes = map[int]int64{}
	}
	decoder.textRunes[index] += int64(utf8.RuneCountInString(text))
}

// holdAnthropicCitation keeps one streamed citation, of any kind, until its
// text block stops (see webSearchURLCitations).
func (decoder *anthropicStreamDecoder) holdAnthropicCitation(
	index int,
	citation json.RawMessage,
) ([]llmprotocol.Event, bool, error) {
	// Any citation kind is carried, including ones not named yet, but it must
	// be an object stating its type: anything else would reach an Anthropic
	// client as a malformed citation.
	if !typedCitationObject(citation) {
		return nil, true, invalidProviderResponse("invalid_stream_delta", "Anthropic citation delta is not a typed citation object")
	}
	if decoder.pendingCitations == nil {
		decoder.pendingCitations = map[int][]json.RawMessage{}
	}
	decoder.pendingCitations[index] = append(decoder.pendingCitations[index], citation)
	return nil, true, nil
}

// releaseAnthropicCitations completes a text block that held web search
// citations: one text delta carries them -- raw, for an Anthropic client,
// and as URL citations over the whole block, for any other -- then the block
// completes. A block that held none is left to the caller.
func (decoder *anthropicStreamDecoder) releaseAnthropicCitations(index int) ([]llmprotocol.Event, bool, error) {
	held := decoder.pendingCitations[index]
	if len(held) == 0 {
		return nil, false, nil
	}
	delete(decoder.pendingCitations, index)
	raw, _ := json.Marshal(held)
	events, _, err := decoder.emitAnthropicEvent(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: index,
		Content: &llmprotocol.Content{
			Kind: llmprotocol.ContentText, CitationsRaw: raw,
			Citations: webSearchURLCitationsTo(raw, decoder.textRunes[index]),
		},
	})
	if err != nil {
		return nil, true, err
	}
	completed, _, err := decoder.emitAnthropicEvent(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: index})
	return append(events, completed...), true, err
}

// carriedAnthropicServerBlock returns the carried whole Anthropic block a
// content holds (a web search block or redacted_thinking), or nil.
func carriedAnthropicServerBlock(content *llmprotocol.Content) *llmprotocol.UnmodeledBlock {
	if content == nil || content.Kind != llmprotocol.ContentUnmodeled || content.Unmodeled == nil ||
		content.Unmodeled.Format != llmprotocol.AnthropicMessagesV1 || !anthropicCarriedResponseBlock(content.Unmodeled.Type) {
		return nil
	}
	return content.Unmodeled
}

// startAnthropicServerBlock opens a carried web search block as the provider
// did: a server_tool_use starts with an empty input that its completion
// streams, and a web_search_tool_result starts whole.
func (encoder *anthropicStreamEncoder) startAnthropicServerBlock(
	event llmprotocol.Event,
	carried *llmprotocol.UnmodeledBlock,
) ([][]byte, error) {
	key := encoder.liveBlockKey(contentKey(event))
	var frames [][]byte
	if encoder.hasActiveBlock && encoder.activeBlock != key {
		stopped, err := encoder.stopAnthropicBlock(encoder.activeBlock)
		if err != nil {
			return nil, err
		}
		frames = append(frames, stopped...)
	}
	encoder.blockStarted[key] = true
	encoder.blocks[key] = llmprotocol.ContentUnmodeled
	encoder.blockIndexes[key] = encoder.nextBlockIndex
	encoder.nextBlockIndex++
	encoder.itemBlockKeys[event.ItemIndex] = append(encoder.itemBlockKeys[event.ItemIndex], key)
	var block map[string]json.RawMessage
	if err := json.Unmarshal(carried.Raw, &block); err != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "encode_wire", "carried web search block is invalid", err)
	}
	if carried.Type == "server_tool_use" {
		block["input"] = json.RawMessage(`{}`)
	}
	frame, err := encodeSSE("content_block_start", map[string]any{
		"type": "content_block_start", "index": encoder.blockIndexes[key], "content_block": block,
	})
	if err != nil {
		return nil, err
	}
	encoder.activeBlock, encoder.hasActiveBlock = key, true
	return append(frames, frame), nil
}

// completeAnthropicServerBlock streams a server_tool_use's input and stops the
// block.
func (encoder *anthropicStreamEncoder) completeAnthropicServerBlock(
	event llmprotocol.Event,
	carried *llmprotocol.UnmodeledBlock,
) ([][]byte, error) {
	keys := encoder.itemBlockKeys[event.ItemIndex]
	if len(keys) == 0 {
		started, err := encoder.startAnthropicServerBlock(event, carried)
		if err != nil {
			return nil, err
		}
		completed, err := encoder.completeAnthropicServerBlock(event, carried)
		return append(started, completed...), err
	}
	key := keys[len(keys)-1]
	var frames [][]byte
	if carried.Type == "server_tool_use" {
		var block struct {
			Input json.RawMessage `json:"input"`
		}
		if json.Unmarshal(carried.Raw, &block) == nil && len(block.Input) > 0 && string(block.Input) != "{}" {
			frame, err := encodeSSE("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": encoder.blockIndexes[key],
				"delta": map[string]string{"type": "input_json_delta", "partial_json": string(block.Input)},
			})
			if err != nil {
				return nil, err
			}
			frames = append(frames, frame)
		}
	}
	stopped, err := encoder.stopAnthropicBlock(key)
	return append(frames, stopped...), err
}

// encodeAnthropicCitationDeltas writes one citations_delta per carried
// citation, as Anthropic streamed them.
func (encoder *anthropicStreamEncoder) encodeAnthropicCitationDeltas(
	key anthropicBlockKey,
	raw json.RawMessage,
) ([][]byte, error) {
	var citations []json.RawMessage
	if err := json.Unmarshal(raw, &citations); err != nil {
		return nil, nil
	}
	frames := make([][]byte, 0, len(citations))
	for _, citation := range citations {
		frame, err := encodeSSE("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": encoder.blockIndexes[key],
			"delta": map[string]any{"type": "citations_delta", "citation": citation},
		})
		if err != nil {
			return nil, err
		}
		frames = append(frames, frame)
	}
	return frames, nil
}

// encodeResponsesAnthropicWebSearch writes a carried Anthropic web search as a
// Responses web_search_call. The server_tool_use names the query and the
// web_search_tool_result that answers it holds the sources, so the call is
// written once, when the result completes.
func (encoder *responsesStreamEncoder) encodeResponsesAnthropicWebSearch(
	event llmprotocol.Event,
	carried *llmprotocol.UnmodeledBlock,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.Type != llmprotocol.EventOutputItemCompleted {
		return nil, nil, nil
	}
	if carried.Type == "server_tool_use" {
		var use anthropicServerToolUseWire
		if json.Unmarshal(carried.Raw, &use) == nil && use.Name == "web_search" {
			if encoder.anthropicSearchQueries == nil {
				encoder.anthropicSearchQueries = map[string]anthropicPendingSearch{}
			}
			encoder.anthropicSearchQueries[use.ID] = anthropicPendingSearch{query: use.Input.Query, item: event.ItemIndex}
			encoder.anthropicSearchOrder = append(encoder.anthropicSearchOrder, use.ID)
		}
		return nil, nil, nil
	}
	var result anthropicWebSearchResultWire
	if json.Unmarshal(carried.Raw, &result) != nil {
		return nil, nil, nil
	}
	pending, known := encoder.anthropicSearchQueries[result.ToolUseID]
	if !known {
		return nil, nil, nil
	}
	delete(encoder.anthropicSearchQueries, result.ToolUseID)
	return encoder.encodeAnthropicSearchCall(event, result.ToolUseID, pending.query, result.outcome())
}

type anthropicPendingSearch struct {
	query string
	item  int
}

// flushPendingAnthropicSearches reports, before the response completes, each
// search whose result never arrived -- a turn Anthropic paused mid-search --
// as a web_search_call in progress, as a buffered response reports it.
func (encoder *responsesStreamEncoder) flushPendingAnthropicSearches(event llmprotocol.Event) ([][]byte, error) {
	var frames [][]byte
	for _, id := range encoder.anthropicSearchOrder {
		pending, open := encoder.anthropicSearchQueries[id]
		if !open {
			continue
		}
		delete(encoder.anthropicSearchQueries, id)
		call := event
		call.Type, call.ItemIndex = llmprotocol.EventOutputItemCompleted, pending.item
		flushed, _, err := encoder.encodeAnthropicSearchCall(call, id, pending.query, anthropicWebSearchOutcome{pending: true})
		if err != nil {
			return nil, err
		}
		frames = append(frames, flushed...)
	}
	return frames, nil
}

func (encoder *responsesStreamEncoder) encodeAnthropicSearchCall(
	event llmprotocol.Event,
	id, query string,
	outcome anthropicWebSearchOutcome,
) ([][]byte, llmprotocol.Diagnostics, error) {
	var use anthropicServerToolUseWire
	use.ID, use.Name, use.Input.Query = id, "web_search", query
	content := webSearchCallFor(use, outcome)
	call := event
	call.ItemID, call.Content = id, &content
	frames, key, err := encoder.ensureResponsesOutputStarted(call, responsesOutputWebSearch)
	if err != nil {
		return nil, nil, err
	}
	completed, diagnostics, err := encoder.encodeCompletedResponsesWebSearch(call, key)
	return append(frames, completed...), diagnostics, err
}
