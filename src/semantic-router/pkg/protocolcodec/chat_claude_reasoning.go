package protocolcodec

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude's thinking round-trips through a Chat Completions client in the
// carrier OpenRouter itself uses for Claude over Chat: reasoning_details items
// tagged anthropic-claude-v1 (atlasfutures/semantic-router#278).
//
// A Chat client has no member for a thinking signature or a redacted_thinking
// block. Until #278 the response leg cleared the signature and dropped the
// redacted block, and the request leg then dropped the unsigned thinking the
// client replayed (content.thinking.unsigned): Claude lost its own reasoning
// on every tool turn of a loop that stayed on a Messages Claude arm. Anthropic
// accepts that replay, with every block removed, and answers without the
// reasoning; it rejects a replay with a block missing or a signature empty.
//
// So the two legs mirror each other and are all-or-nothing:
//
//   - Response leg (withClaudeReasoningCarriers, and the Chat stream encoder):
//     each signed thinking block becomes a reasoning.text item with its text
//     and signature, each redacted_thinking block a reasoning.encrypted item
//     with its data, numbered in order across the choice, because the Chat
//     encoder merges a choice's output items into one message and joins their
//     arrays. The Chat client sees reasoning_content as before, and beside it
//     the array OpenRouter would have sent.
//   - Request leg (ClaudeThinkingFromReasoningDetails, at dispatch for a
//     Claude worker on a Messages target): an assistant message's array is
//     read back into blocks only when every item is Claude's and complete and
//     the item texts are the message's reasoning text; otherwise the array is
//     left alone and the unsigned row drops the reasoning whole. A replay is
//     never partial. A Chat target is untouched: it gets the array back byte
//     for byte, as before.
//
// The provenance assumption is the one the dispatch filters already make: a
// signature on a Messages-format thinking block is Anthropic's. A Router
// marker (routerSignature) is never minted into an item and never read out
// of one.

// chatClaudeReasoningCarried is the diagnostic reason for a thinking
// signature a Chat client is given inside reasoning_details.
const chatClaudeReasoningCarried = "the client contract has no member for a reasoning signature; it is carried as an anthropic-claude-v1 reasoning_details item"

// chatRedactedThinkingCarried is the diagnostic reason for a redacted_thinking
// block a Chat client is given as a reasoning.encrypted item.
const chatRedactedThinkingCarried = "redacted_thinking is carried as an anthropic-claude-v1 reasoning.encrypted item"

// claudeReasoningTextItem is the reasoning_details item OpenRouter returns for
// one signed Claude thinking block and asks for back.
func claudeReasoningTextItem(text, signature string, index int) map[string]any {
	return map[string]any{
		"type": "reasoning.text", "text": text, "signature": signature,
		"format": responsesAnthropicReasoningFormat, "index": index,
	}
}

// claudeReasoningEncryptedItem is the reasoning_details item OpenRouter returns
// for one Claude redacted_thinking block.
func claudeReasoningEncryptedItem(data string, index int) map[string]any {
	return map[string]any{
		"type": "reasoning.encrypted", "data": data,
		"format": responsesAnthropicReasoningFormat, "index": index,
	}
}

// claudeReasoningDetails wraps items as the carrier a reasoning content holds.
func claudeReasoningDetails(items ...map[string]any) *llmprotocol.UnmodeledFields {
	details, err := json.Marshal(items)
	if err != nil {
		return nil
	}
	return reasoningDetailsFields(details)
}

// claudeReasoningCarry counts what withClaudeReasoningCarriers changed.
type claudeReasoningCarry struct {
	// Signed is the number of signed thinking blocks carried as items.
	Signed int
	// Redacted is the number of redacted_thinking blocks carried as items.
	Redacted int
	// Refused is the number of signatures in the Router's namespace, which
	// are never carried: they are cleared, as they were before #278.
	Refused int
}

// withClaudeReasoningCarriers returns, on fresh slices where it changes
// anything, one choice's output with every signed thinking block and every
// redacted_thinking block rewritten as the reasoning_details item a Chat
// client can replay. Items are numbered in order across the choice. A
// reasoning content that already carries reasoning_details is left as it is:
// it came from a Chat upstream and holds that upstream's items.
func withClaudeReasoningCarriers(items []llmprotocol.OutputItem) ([]llmprotocol.OutputItem, claudeReasoningCarry) {
	var carry claudeReasoningCarry
	var out []llmprotocol.OutputItem
	next := 0
	for index, item := range items {
		var contents []llmprotocol.Content
		for contentIndex, content := range item.Content {
			rewritten, ok := claudeReasoningCarrier(content, &next, &carry)
			if !ok {
				continue
			}
			if contents == nil {
				contents = append([]llmprotocol.Content(nil), item.Content...)
			}
			contents[contentIndex] = rewritten
		}
		if contents == nil {
			continue
		}
		if out == nil {
			out = append([]llmprotocol.OutputItem(nil), items...)
		}
		out[index].Content = contents
	}
	if out == nil {
		return items, carry
	}
	return out, carry
}

// claudeReasoningCarrier returns the carrier content for one block, and false
// for a content that is not a signed thinking or redacted_thinking block.
func claudeReasoningCarrier(content llmprotocol.Content, next *int, carry *claudeReasoningCarry) (llmprotocol.Content, bool) {
	if data, ok := OpaqueReasoningData(content); ok {
		carry.Redacted++
		item := claudeReasoningEncryptedItem(data, *next)
		*next++
		return llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeText,
			Extensions: claudeReasoningDetails(item),
		}, true
	}
	if content.Kind != llmprotocol.ContentReasoning || content.Signature == "" || content.Extensions != nil ||
		content.Reasoning == llmprotocol.ReasoningScopeSummary {
		return content, false
	}
	if _, reserved := routerSignature(content.Signature); reserved {
		carry.Refused++
		content.Signature = ""
		return content, true
	}
	carry.Signed++
	item := claudeReasoningTextItem(content.Text, content.Signature, *next)
	*next++
	content.Signature = ""
	content.Extensions = claudeReasoningDetails(item)
	return content, true
}

// responseWithClaudeReasoningCarriers is responseForTarget for a Chat target:
// the response a Chat encoder is given, with Claude's thinking carried as
// reasoning_details items in every choice, and the diagnostics that say so.
func responseWithClaudeReasoningCarriers(
	response llmprotocol.Response,
	target llmprotocol.WireFormat,
	limit int,
) (llmprotocol.Response, llmprotocol.Diagnostics) {
	output, carry := withClaudeReasoningCarriers(response.Output)
	var alternatives [][]llmprotocol.OutputItem
	for index, alternative := range response.Alternatives {
		kept, carryHere := withClaudeReasoningCarriers(alternative)
		if carryHere == (claudeReasoningCarry{}) {
			continue
		}
		if alternatives == nil {
			alternatives = append([][]llmprotocol.OutputItem(nil), response.Alternatives...)
		}
		alternatives[index] = kept
		carry.Signed += carryHere.Signed
		carry.Redacted += carryHere.Redacted
		carry.Refused += carryHere.Refused
	}
	if carry == (claudeReasoningCarry{}) {
		return response, nil
	}
	response.Output = output
	if alternatives != nil {
		response.Alternatives = alternatives
	}
	var diagnostics llmprotocol.Diagnostics
	if carry.Signed > 0 {
		diagnostics = append(diagnostics, llmprotocol.Diagnostic{
			Target: target, Field: "reasoning.signature",
			Action: llmprotocol.DiagnosticApproximated, Reason: chatClaudeReasoningCarried,
		})
	}
	if carry.Redacted > 0 {
		diagnostics = append(diagnostics, llmprotocol.Diagnostic{
			Target: target, Field: fieldRedactedThinking,
			Action: llmprotocol.DiagnosticApproximated, Reason: chatRedactedThinkingCarried,
		})
	}
	if carry.Refused > 0 {
		diagnostics = append(diagnostics, llmprotocol.Diagnostic{
			Target: target, Field: "reasoning.signature",
			Action: llmprotocol.DiagnosticDropped, Reason: responseReasoningSignatureDropped,
		})
	}
	return response, appendDiagnostics(nil, diagnostics, limit)
}

// claudeReasoningItemWire is the shape of one reasoning_details item as
// ClaudeThinkingFromReasoningDetails reads it.
type claudeReasoningItemWire struct {
	Type      string `json:"type"`
	Text      string `json:"text"`
	Signature string `json:"signature"`
	Data      string `json:"data"`
	Format    string `json:"format"`
}

// claudeThinkingBlocks reads a reasoning_details array back into the blocks
// it carries for Claude, in item order, and false when the array is not a
// complete Claude replay: an item of another format or type, a reasoning.text
// item with no signature or one in the Router's namespace, a
// reasoning.encrypted item with no data, or item texts that are not the
// message's reasoning text. text is the reasoning text beside the array; a
// carrier with no text of its own passes "".
func claudeThinkingBlocks(details json.RawMessage, text string) ([]llmprotocol.Content, bool) {
	var items []claudeReasoningItemWire
	if json.Unmarshal(details, &items) != nil || len(items) == 0 {
		return nil, false
	}
	blocks := make([]llmprotocol.Content, 0, len(items))
	var joined strings.Builder
	for _, item := range items {
		if item.Format != responsesAnthropicReasoningFormat {
			return nil, false
		}
		switch item.Type {
		case "reasoning.text":
			if item.Signature == "" {
				return nil, false
			}
			if _, reserved := routerSignature(item.Signature); reserved {
				return nil, false
			}
			joined.WriteString(item.Text)
			blocks = append(blocks, llmprotocol.Content{
				Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeText,
				Text: item.Text, Signature: item.Signature,
			})
		case "reasoning.encrypted":
			if item.Data == "" {
				return nil, false
			}
			blocks = append(blocks, redactedThinkingContent(item.Data))
		default:
			return nil, false
		}
	}
	if joined.String() != text {
		return nil, false
	}
	return blocks, true
}

// ClaudeThinkingFromReasoningDetails rebuilds, on fresh slices, the thinking
// blocks a Chat client replays as anthropic-claude-v1 reasoning_details: in
// each assistant message, a reasoning content whose array is a complete
// Claude replay (claudeThinkingBlocks) becomes that array's blocks, signed
// thinking and redacted_thinking in item order, so a Messages target is sent
// what Anthropic returned. It is the request-leg mirror of
// withClaudeReasoningCarriers, for a Claude worker reached over Messages,
// and the counterpart of SignedThinkingAsReasoningDetails for the other
// target. The caller has already kept only reasoning with Anthropic
// provenance (DropReasoningNotFromAnthropic).
//
// A message is rebuilt whole or not at all: if any of its reasoning_details
// arrays is not a complete replay, the message is left as it is, and the
// unsigned row of the disposition table drops its reasoning, which Anthropic
// accepts. A replayed turn with some blocks missing it rejects. It returns
// how many blocks it rebuilt.
func ClaudeThinkingFromReasoningDetails(request *llmprotocol.Request) int {
	if request == nil {
		return 0
	}
	rebuilt := 0
	messages := make([]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		messages[index] = message
		if message.Role != llmprotocol.RoleAssistant || !holdsReasoningDetails(message) {
			continue
		}
		contents, blocks, ok := claudeThinkingFromMessage(message)
		if !ok {
			continue
		}
		messages[index].Content = contents
		rebuilt += blocks
	}
	if rebuilt > 0 {
		request.Messages = messages
	}
	return rebuilt
}

// claudeThinkingFromMessage returns one message's contents with every
// reasoning_details carrier replaced by its blocks, how many blocks, and
// false when any carrier is not a complete Claude replay.
func claudeThinkingFromMessage(message llmprotocol.Message) ([]llmprotocol.Content, int, bool) {
	contents := make([]llmprotocol.Content, 0, len(message.Content))
	rebuilt := 0
	for _, content := range message.Content {
		details, _ := reasoningDetailsOf(content)
		if details == nil {
			contents = append(contents, content)
			continue
		}
		if content.Signature != "" || content.Reasoning == llmprotocol.ReasoningScopeSummary {
			// Already signed, or a summary: not a carrier this reads.
			return nil, 0, false
		}
		blocks, ok := claudeThinkingBlocks(details, content.Text)
		if !ok {
			return nil, 0, false
		}
		contents = append(contents, blocks...)
		rebuilt += len(blocks)
	}
	return contents, rebuilt, true
}
