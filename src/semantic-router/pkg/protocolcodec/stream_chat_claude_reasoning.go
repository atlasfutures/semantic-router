package protocolcodec

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The Chat stream encoder's half of chat_claude_reasoning.go: Claude's
// thinking, streamed from a Messages arm, reaches a Chat client as the
// reasoning_details items it can replay, in the shape OpenRouter streams them.
//
// When the Router has said the worker is Claude (StreamContext.WorkerFamily)
// and the stream comes in the Messages format, the one format whose thinking
// blocks Anthropic signs, each thinking_delta goes out as reasoning_content
// and, beside it, a reasoning.text fragment tagged anthropic-claude-v1 under
// the block's item index; the block's signature_delta goes out as the item's
// last fragment, the signature alone. That is OpenRouter's wire shape for
// Claude over Chat (testdata/golden/capability/056), so a client that already
// merges OpenRouter's fragments by index reads this stream unchanged. A
// redacted_thinking block is one reasoning.encrypted item when it completes.
//
// Any other stream is left as it was. A Chat-format Claude arm's stream
// already carries OpenRouter's own items, which pass through untouched, and
// its text-only chunks must not be tagged beside them: a fragment minted here
// would take an index OpenRouter is using. A Responses-format arm's signature
// arrives with the item, so its block is minted whole when the signature
// does. And a stream whose worker the encoder was not told about (one encoded
// outside the Router's dispatch) also waits for the signature, so nothing is
// tagged as Claude's on the strength of the wire format alone.

// claudeWorkerFamily is the family the Router names for Claude.
const claudeWorkerFamily = "anthropic"

// streamsClaudeFragments reports whether the encoder knows the worker is
// Claude and the stream is one Anthropic signs, so its thinking streams in
// OpenRouter's fragment shape.
func (encoder *chatStreamEncoder) streamsClaudeFragments() bool {
	return encoder.context.WorkerFamily == claudeWorkerFamily &&
		encoder.context.Source == llmprotocol.AnthropicMessagesV1
}

// mayStillSign reports whether this stream's source can still send a
// signature for a thinking block: Messages and Responses decoders put one on
// the content, a Chat decoder never does (its reasoning_details ride as an
// extension), so a Chat-sourced block's text is not kept for one.
func (encoder *chatStreamEncoder) mayStillSign() bool {
	return encoder.context.Source != llmprotocol.OpenAIChatV1
}

// claudeItemIndex is the reasoning_details index a thinking block's fragments
// share, allotted when the block is first seen.
func (encoder *chatStreamEncoder) claudeItemIndex(key streamContentKey) int {
	if index, found := encoder.claudeItemIndexes[key]; found {
		return index
	}
	if encoder.claudeItemIndexes == nil {
		encoder.claudeItemIndexes = map[streamContentKey]int{}
	}
	index := encoder.claudeItems
	encoder.claudeItems++
	encoder.claudeItemIndexes[key] = index
	return index
}

// applyClaudeThinkingText handles a thinking block's text delta, which has
// already been written as reasoning_content: for a Claude worker it goes out
// as a tagged fragment too; otherwise it is kept for the item the signature
// will complete. A summary is never an item.
func (encoder *chatStreamEncoder) applyClaudeThinkingText(event llmprotocol.Event, choice *chatChunkChoiceWire) {
	if event.Content.Reasoning == llmprotocol.ReasoningScopeSummary || event.Delta == "" {
		return
	}
	key := contentKey(event)
	if !encoder.streamsClaudeFragments() {
		if encoder.mayStillSign() {
			encoder.noteReasoningText(key, event.Delta)
		}
		return
	}
	if carrier := claudeReasoningDetails(claudeReasoningTextFragment(event.Delta, encoder.claudeItemIndex(key))); carrier != nil {
		choice.Delta.ReasoningDetails = carrier.Fields[chatReasoningDetailsMember]
	}
}

// noteReasoningText keeps a thinking block's text so far, for the item its
// signature will complete.
func (encoder *chatStreamEncoder) noteReasoningText(key streamContentKey, text string) {
	if encoder.reasoningText == nil {
		encoder.reasoningText = map[streamContentKey]*strings.Builder{}
	}
	builder := encoder.reasoningText[key]
	if builder == nil {
		builder = &strings.Builder{}
		encoder.reasoningText[key] = builder
	}
	builder.WriteString(text)
}

// takeReasoningText returns a block's text so far and forgets it: the
// signature that asks for it ends the block.
func (encoder *chatStreamEncoder) takeReasoningText(key streamContentKey) string {
	builder := encoder.reasoningText[key]
	if builder == nil {
		return ""
	}
	delete(encoder.reasoningText, key)
	return builder.String()
}

// applyClaudeThinkingSignature writes a thinking block's signature: for a
// Claude worker as the item's last fragment, otherwise as the whole item with
// the block's text. A signature in the Router's namespace, or one on a
// summary, is never carried: it is dropped and reported, as every signature
// was before #278.
func (encoder *chatStreamEncoder) applyClaudeThinkingSignature(
	event llmprotocol.Event,
	choice *chatChunkChoiceWire,
) (llmprotocol.Diagnostics, bool, error) {
	key := contentKey(event)
	if event.Delta != "" {
		choice.Delta.Reasoning = &event.Delta
		encoder.applyClaudeThinkingText(event, choice)
	}
	if _, reserved := routerSignature(event.Content.Signature); reserved ||
		event.Content.Reasoning == llmprotocol.ReasoningScopeSummary {
		encoder.takeReasoningText(key)
		return appendDiagnostics(nil, llmprotocol.Diagnostics{{
			Source: encoder.context.Source, Target: encoder.context.Target, Field: "reasoning.signature",
			Action: llmprotocol.DiagnosticDropped, Reason: responseReasoningSignatureDropped,
		}}, encoder.policy.Limits.Diagnostics), event.Delta != "", nil
	}
	var item map[string]any
	if encoder.streamsClaudeFragments() {
		item = claudeReasoningSignatureFragment(event.Content.Signature, encoder.claudeItemIndex(key))
	} else {
		item = claudeReasoningTextItem(encoder.takeReasoningText(key), event.Content.Signature, encoder.claudeItems)
		encoder.claudeItems++
	}
	carrier := claudeReasoningDetails(item)
	if carrier == nil {
		return nil, false, llmprotocol.NewError(llmprotocol.ErrorInternal, "invalid_reasoning_details", "reasoning_details cannot be encoded", nil)
	}
	choice.Delta.ReasoningDetails = carrier.Fields[chatReasoningDetailsMember]
	return appendDiagnostics(nil, llmprotocol.Diagnostics{{
		Source: encoder.context.Source, Target: encoder.context.Target, Field: "reasoning.signature",
		Action: llmprotocol.DiagnosticApproximated, Reason: chatClaudeReasoningCarried,
	}}, encoder.policy.Limits.Diagnostics), true, nil
}

// encodeClaudeRedactedThinking writes a completed redacted_thinking block as
// the reasoning.encrypted item a Chat client can replay. The block arrives
// whole, so it is one chunk, and it opens the choice if nothing has yet. The
// event has been through next, so its id and model are the stream's.
func (encoder *chatStreamEncoder) encodeClaudeRedactedThinking(
	event llmprotocol.Event,
	data string,
) ([][]byte, llmprotocol.Diagnostics, error) {
	carrier := claudeReasoningDetails(claudeReasoningEncryptedItem(data, encoder.claudeItems))
	if carrier == nil {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "invalid_reasoning_details", "reasoning_details cannot be encoded", nil)
	}
	encoder.claudeItems++
	chunk := chatChunkWire{ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model}
	choice := chatChunkChoiceWire{Index: 0}
	encoder.startChatItem(&choice)
	choice.Delta.ReasoningDetails = carrier.Fields[chatReasoningDetailsMember]
	chunk.Choices = []chatChunkChoiceWire{choice}
	diagnostics := appendDiagnostics(nil, llmprotocol.Diagnostics{{
		Source: encoder.context.Source, Target: encoder.context.Target, Field: fieldRedactedThinking,
		Action: llmprotocol.DiagnosticApproximated, Reason: chatRedactedThinkingCarried,
	}}, encoder.policy.Limits.Diagnostics)
	frame, err := encodeChatStreamFrame(encoder.context, chunk)
	if err != nil {
		return nil, diagnostics, err
	}
	return [][]byte{frame}, diagnostics, nil
}
