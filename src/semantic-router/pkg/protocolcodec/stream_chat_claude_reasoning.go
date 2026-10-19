package protocolcodec

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The Chat stream encoder's half of chat_claude_reasoning.go: Claude's
// thinking, streamed from a Messages arm, reaches a Chat client as the
// reasoning_details items it can replay. A thinking block's text streams on
// as reasoning_content; its signature, which Anthropic sends after the text,
// becomes one complete reasoning.text item holding the whole text; a
// redacted_thinking block becomes one reasoning.encrypted item when it
// completes. One complete item per block means no text is tagged with
// Anthropic provenance before the signature proves it, and a client that
// merges fragments by index ends with the array a buffered answer carries.

// noteReasoningText keeps a thinking block's text so far, for the item its
// signature will complete.
func (encoder *chatStreamEncoder) noteReasoningText(key streamContentKey, text string) {
	if text == "" {
		return
	}
	if encoder.reasoningText == nil {
		encoder.reasoningText = map[streamContentKey]string{}
	}
	encoder.reasoningText[key] += text
}

// applyClaudeThinkingSignature writes a thinking block's signature as the
// reasoning.text item carrying the block's text. A signature in the Router's
// namespace, or one on a summary, is never carried: it is dropped and
// reported, as every signature was before #278.
func (encoder *chatStreamEncoder) applyClaudeThinkingSignature(
	event llmprotocol.Event,
	choice *chatChunkChoiceWire,
) (llmprotocol.Diagnostics, bool, error) {
	key := contentKey(event)
	if event.Content.Reasoning != llmprotocol.ReasoningScopeSummary {
		encoder.noteReasoningText(key, event.Delta)
	}
	if event.Delta != "" {
		choice.Delta.Reasoning = &event.Delta
	}
	if _, reserved := routerSignature(event.Content.Signature); reserved ||
		event.Content.Reasoning == llmprotocol.ReasoningScopeSummary {
		return appendDiagnostics(nil, llmprotocol.Diagnostics{{
			Source: encoder.context.Source, Target: encoder.context.Target, Field: "reasoning.signature",
			Action: llmprotocol.DiagnosticDropped, Reason: responseReasoningSignatureDropped,
		}}, encoder.policy.Limits.Diagnostics), event.Delta != "", nil
	}
	carrier := claudeReasoningDetails(claudeReasoningTextItem(encoder.reasoningText[key], event.Content.Signature, encoder.claudeItems))
	if carrier == nil {
		return nil, false, llmprotocol.NewError(llmprotocol.ErrorInternal, "invalid_reasoning_details", "reasoning_details cannot be encoded", nil)
	}
	encoder.claudeItems++
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
