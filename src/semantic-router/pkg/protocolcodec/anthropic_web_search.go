package protocolcodec

import (
	"encoding/json"
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

type anthropicWebSearchCitationWire struct {
	Type  string `json:"type"`
	URL   string `json:"url"`
	Title string `json:"title"`
}

// webSearchResultCitations reports whether every citation on a provider text
// block is a web search result location: the one kind whose source, a URL,
// has a neutral shape. Any other kind still fails closed.
func webSearchResultCitations(raw json.RawMessage) bool {
	var citations []anthropicWebSearchCitationWire
	if json.Unmarshal(raw, &citations) != nil || len(citations) == 0 {
		return false
	}
	for _, citation := range citations {
		if citation.Type != "web_search_result_location" || citation.URL == "" {
			return false
		}
	}
	return true
}

// webSearchURLCitations turns web search result locations into URL citations.
// Anthropic does not say which span of the answer a result supports, so each
// cites the whole block: the span is derived, which is why only this kind is
// admitted, and an Anthropic client gets the raw citations back instead.
func webSearchURLCitations(raw json.RawMessage, text string) []llmprotocol.Citation {
	return webSearchURLCitationsTo(raw, int64(utf8.RuneCountInString(text)))
}

// webSearchURLCitationsTo is webSearchURLCitations for a span ending at end.
func webSearchURLCitationsTo(raw json.RawMessage, end int64) []llmprotocol.Citation {
	var wire []anthropicWebSearchCitationWire
	if json.Unmarshal(raw, &wire) != nil {
		return nil
	}
	citations := make([]llmprotocol.Citation, 0, len(wire))
	for _, citation := range wire {
		citations = append(citations, llmprotocol.Citation{
			URL: citation.URL, Title: citation.Title, StartIndex: 0, EndIndex: end,
		})
	}
	return citations
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
				block.Format == llmprotocol.AnthropicMessagesV1 && anthropicServerToolBlock(block.Type) {
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
	ToolUseID string `json:"tool_use_id"`
	Content   []struct {
		Type string `json:"type"`
		URL  string `json:"url"`
	} `json:"content"`
}

// anthropicWebSearchAsResponses rewrites the carried Anthropic web search
// blocks of one output into Responses web_search_call contents: each
// server_tool_use naming web_search becomes one call carrying its query, and
// the sources of the result that answers it. The result blocks are consumed.
func anthropicWebSearchAsResponses(contents []llmprotocol.Content) []llmprotocol.Content {
	sources := map[string][]map[string]string{}
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
		for _, item := range result.Content {
			if item.Type == "web_search_result" && item.URL != "" {
				sources[result.ToolUseID] = append(sources[result.ToolUseID], map[string]string{"type": "url", "url": item.URL})
			}
		}
	}
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
			fields := map[string]any{"type": "search", "query": use.Input.Query}
			if found := sources[use.ID]; len(found) > 0 {
				fields["sources"] = found
			}
			action, _ := json.Marshal(fields)
			rewritten = append(rewritten, webSearchCallContent(responsesItemWire{
				Type: "web_search_call", ID: use.ID, Status: "completed", Action: action,
			}))
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
		if wire.ContentBlock == nil || !anthropicServerToolBlock(wire.ContentBlock.Type) {
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
			return decoder.decodeAnthropicCitationDelta(index, wire.Delta.Citation)
		}
		return nil, false, nil
	case "content_block_stop":
		pending := decoder.serverBlocks[index]
		if pending == nil {
			return nil, false, nil
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

// decodeAnthropicCitationDelta turns one streamed web search citation into a
// text delta that carries it: raw, for an Anthropic client, and as a URL
// citation over the block's text so far, for any other. Any other citation
// kind still fails closed.
func (decoder *anthropicStreamDecoder) decodeAnthropicCitationDelta(
	index int,
	citation json.RawMessage,
) ([]llmprotocol.Event, bool, error) {
	raw := append(append(json.RawMessage("["), citation...), ']')
	if !webSearchResultCitations(raw) {
		return nil, true, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_citations",
			"Anthropic citations are not supported by the neutral contract", nil)
	}
	delta := llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: index,
		Content: &llmprotocol.Content{
			Kind: llmprotocol.ContentText, CitationsRaw: raw,
			Citations: webSearchURLCitationsTo(raw, decoder.textRunes[index]),
		},
	}
	events, _, err := decoder.emitAnthropicEvent(delta)
	return events, true, err
}

// carriedAnthropicServerBlock returns the carried Anthropic web search block a
// content holds, or nil.
func carriedAnthropicServerBlock(content *llmprotocol.Content) *llmprotocol.UnmodeledBlock {
	if content == nil || content.Kind != llmprotocol.ContentUnmodeled || content.Unmodeled == nil ||
		content.Unmodeled.Format != llmprotocol.AnthropicMessagesV1 || !anthropicServerToolBlock(content.Unmodeled.Type) {
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
				encoder.anthropicSearchQueries = map[string]string{}
			}
			encoder.anthropicSearchQueries[use.ID] = use.Input.Query
		}
		return nil, nil, nil
	}
	var result anthropicWebSearchResultWire
	if json.Unmarshal(carried.Raw, &result) != nil {
		return nil, nil, nil
	}
	query, known := encoder.anthropicSearchQueries[result.ToolUseID]
	if !known {
		return nil, nil, nil
	}
	contents := anthropicWebSearchAsResponses([]llmprotocol.Content{
		carriedAnthropicBlock("server_tool_use", anthropicJSON(map[string]any{
			"type": "server_tool_use", "id": result.ToolUseID, "name": "web_search", "input": map[string]string{"query": query},
		})),
		{Kind: llmprotocol.ContentUnmodeled, Unmodeled: carried},
	})
	if len(contents) != 1 {
		return nil, nil, nil
	}
	call := event
	call.ItemID, call.Content = result.ToolUseID, &contents[0]
	frames, key, err := encoder.ensureResponsesOutputStarted(call, responsesOutputWebSearch)
	if err != nil {
		return nil, nil, err
	}
	completed, diagnostics, err := encoder.encodeCompletedResponsesWebSearch(call, key)
	return append(frames, completed...), diagnostics, err
}

func anthropicJSON(value any) json.RawMessage {
	raw, _ := json.Marshal(value)
	return raw
}
