package protocolcodec

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// MessageThinkingSignatures lists the provider signatures of a message's
// signed thinking, in order.
func MessageThinkingSignatures(message llmprotocol.Message) []string {
	var signatures []string
	for _, content := range message.Content {
		if signature, ok := providerThinkingSignature(content); ok {
			signatures = append(signatures, signature)
		}
	}
	return signatures
}

// MessageThinkingKeys lists what identifies each of a message's thinking
// blocks on the wire: a signed block's signature and a redacted_thinking
// block's data.
func MessageThinkingKeys(message llmprotocol.Message) []string {
	keys := MessageThinkingSignatures(message)
	for _, content := range message.Content {
		if data, ok := OpaqueReasoningData(content); ok {
			keys = append(keys, data)
		}
	}
	return keys
}

// MessageHoldsThinking reports whether a message holds reasoning of any kind
// or a redacted_thinking block.
func MessageHoldsThinking(message llmprotocol.Message) bool {
	for _, content := range message.Content {
		if content.Kind == llmprotocol.ContentReasoning || redactedThinkingBlock(content) {
			return true
		}
	}
	return false
}

// DropMessageThinking removes, on fresh slices, every reasoning content and
// redacted_thinking block of each message drop selects, whole, and any
// message it leaves empty: Anthropic accepts an assistant turn with all of
// its thinking removed and rejects one with part of it removed. Visible text,
// tool calls and tool results are never touched. drop sees each message with
// its index in the request as given. It returns how many blocks it removed.
func DropMessageThinking(request *llmprotocol.Request, drop func(index int, message llmprotocol.Message) bool) int {
	if request == nil {
		return 0
	}
	dropped := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for index, message := range request.Messages {
		if !MessageHoldsThinking(message) || !drop(index, message) {
			messages = append(messages, message)
			continue
		}
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentReasoning || redactedThinkingBlock(content) {
				dropped++
				continue
			}
			contents = append(contents, content)
		}
		if len(contents) == 0 {
			continue
		}
		message.Content = contents
		messages = append(messages, message)
	}
	if dropped > 0 {
		request.Messages = messages
	}
	return dropped
}

func providerThinkingSignature(content llmprotocol.Content) (string, bool) {
	// A Router marker never reaches here: the request decoders strip it
	// (stripRouterSignatures) and a decoded response is never minted.
	if content.Kind != llmprotocol.ContentReasoning || content.Signature == "" {
		return "", false
	}
	return content.Signature, true
}

// The keys a thinking run is anchored by. A served assistant message is found
// again in a later history by what the client resends of it unchanged: a
// thinking signature, a redacted_thinking block's data, a tool call's id, or
// failing those its visible text. The first key of a served reply is the
// run's anchor, so it must be one the client keeps: a streamed Chat reply's
// reconstruction keeps no reasoning_details, but it keeps its tool calls and
// text.
const (
	anchorKeySignature = "s\x00"
	anchorKeyRedacted  = "r\x00"
	anchorKeyToolCall  = "t\x00"
	anchorKeyText      = "x\x00"
)

// SignatureAnchorKey and RedactedAnchorKey are the anchor keys of a thinking
// block's signature and of a redacted_thinking block's data.
func SignatureAnchorKey(signature string) string { return anchorKeySignature + signature }

func RedactedAnchorKey(data string) string { return anchorKeyRedacted + data }

// ToolCallAnchorKey is the anchor key of a tool call's id.
func ToolCallAnchorKey(id string) string { return anchorKeyToolCall + id }

// MessageAnchorKeys lists the anchor keys a message at index holds,
// signatures and redacted data first, then tool call ids, then its visible
// text with its index: a reply's text alone may repeat an earlier turn's
// ("Done."), its text at its place in the history does not.
func MessageAnchorKeys(message llmprotocol.Message, index int) []string {
	var keys, calls []string
	var text strings.Builder
	for _, content := range message.Content {
		switch {
		case content.Kind == llmprotocol.ContentReasoning && content.Signature != "":
			keys = append(keys, anchorKeySignature+content.Signature)
		case content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil && content.ToolCall.ID != "":
			calls = append(calls, anchorKeyToolCall+content.ToolCall.ID)
		case content.Kind == llmprotocol.ContentText:
			text.WriteString(content.Text)
		default:
			if data, ok := OpaqueReasoningData(content); ok {
				keys = append(keys, anchorKeyRedacted+data)
			}
		}
	}
	keys = append(keys, calls...)
	if text.Len() > 0 {
		digest := sha256.Sum256([]byte(text.String()))
		keys = append(keys, anchorKeyText+strconv.Itoa(index)+"\x00"+hex.EncodeToString(digest[:16]))
	}
	return keys
}

// ResponseAnchorKey is the anchor key of a served reply that will sit at
// index in the next history: the first key of its output read as one
// assistant message, or "" when it holds none.
func ResponseAnchorKey(response *llmprotocol.Response, index int) string {
	if response == nil {
		return ""
	}
	message := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
	for _, item := range response.Output {
		message.Content = append(message.Content, item.Content...)
	}
	if keys := MessageAnchorKeys(message, index); len(keys) > 0 {
		return keys[0]
	}
	return ""
}

// DropForeignOpaqueReasoning removes, on fresh slices, what of a message's
// reasoning only its issuer can read, from each message drop selects, for a
// worker that is not Claude: redacted_thinking blocks, thinking signatures,
// and the reasoning_details OpenRouter carries (Anthropic signatures, OpenAI
// encrypted reasoning, Gemini thought signatures). The reasoning's text is
// kept, as the reasoning carry rules keep it for a Chat or Responses target:
// DeepSeek and MiMo expect reasoning_content on every tool-calling turn of a
// history. A Messages target accepts thinking only signed, so there a
// selected message loses its reasoning whole. A content or message left
// empty goes. It returns how many contents it changed or removed.
func DropForeignOpaqueReasoning(
	request *llmprotocol.Request,
	messagesTarget bool,
	drop func(index int, message llmprotocol.Message) bool,
) int {
	if request == nil {
		return 0
	}
	changed := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for index, message := range request.Messages {
		if !MessageHoldsThinking(message) || !drop(index, message) {
			messages = append(messages, message)
			continue
		}
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			if redactedThinkingBlock(content) {
				changed++
				continue
			}
			if content.Kind != llmprotocol.ContentReasoning {
				contents = append(contents, content)
				continue
			}
			if messagesTarget {
				changed++
				continue
			}
			details, _ := reasoningDetailsOf(content)
			if content.Signature == "" && details == nil {
				contents = append(contents, content)
				continue
			}
			changed++
			content.Signature = ""
			if details != nil {
				if content.Text == "" {
					// The carrier was the reasoning's only text: its
					// plaintext stays.
					content.Text = ReasoningDetailsPlaintext(details)
				}
				content.Extensions = withoutReasoningDetailsMember(content.Extensions)
			}
			if content.Text == "" && content.Extensions == nil {
				continue
			}
			contents = append(contents, content)
		}
		if len(contents) == 0 {
			continue
		}
		message.Content = contents
		messages = append(messages, message)
	}
	if changed > 0 {
		request.Messages = messages
	}
	return changed
}

func withoutReasoningDetailsMember(fields *llmprotocol.UnmodeledFields) *llmprotocol.UnmodeledFields {
	if fields == nil {
		return nil
	}
	kept := make(map[string]json.RawMessage, len(fields.Fields))
	for name, value := range fields.Fields {
		if name != chatReasoningDetailsMember {
			kept[name] = value
		}
	}
	if len(kept) == 0 && len(fields.Children) == 0 {
		return nil
	}
	copied := *fields
	copied.Fields = kept
	return &copied
}

// ReasoningDetailsPlaintext is the readable text a reasoning_details array
// holds: its items' text and summaries, in order.
func ReasoningDetailsPlaintext(details json.RawMessage) string {
	var items []struct {
		Text    string `json:"text"`
		Summary string `json:"summary"`
	}
	if json.Unmarshal(details, &items) != nil {
		return ""
	}
	var text strings.Builder
	for _, item := range items {
		text.WriteString(item.Text)
		text.WriteString(item.Summary)
	}
	return text.String()
}
