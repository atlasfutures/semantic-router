package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ReasoningCarry counts, by kind, what CarryReasoningTo changed in a
// request's history. Each kind is a row of the request disposition table.
type ReasoningCarry struct {
	// SignaturesStripped is Anthropic thinking whose signature was removed
	// for a Chat or Responses target, its text kept (content.thinking.signed).
	SignaturesStripped int
	// RedactedDropped is Anthropic redacted_thinking, dropped for a Chat or
	// Responses target (content.redacted_thinking).
	RedactedDropped int
	// UnsignedDropped is reasoning with no signature, dropped for a Messages
	// target, which accepts thinking only with the signature that proves its
	// issuer (content.thinking.unsigned): another provider's thinking (kimi
	// on OpenRouter sends it empty) or a Chat or Responses reasoning echoed
	// back.
	UnsignedDropped int
	// ForeignDropped is reasoning not provably Claude's, dropped for a Claude
	// worker on any wire format (DropReasoningNotFromAnthropic). It runs
	// first, so a Claude worker on Messages counts its drops here rather than
	// as UnsignedDropped.
	ForeignDropped int
}

// Changed reports whether the request was changed.
func (carry ReasoningCarry) Changed() bool {
	return carry.Dropped() > 0 || carry.SignaturesStripped > 0
}

// Dropped is the number of reasoning blocks dropped, of every kind.
func (carry ReasoningCarry) Dropped() int {
	return carry.RedactedDropped + carry.UnsignedDropped + carry.ForeignDropped
}

func (carry *ReasoningCarry) count(path string) {
	switch path {
	case fieldRedactedThinking:
		carry.RedactedDropped++
	case fieldReasoningUnsigned:
		carry.UnsignedDropped++
	}
}

// CarryReasoningTo applies the reasoning rows of the request disposition
// table to a request's history, in place, and counts what it changed. It runs
// at dispatch, before the capability gate: reasoning a target cannot carry
// was once refused there (Chat and Responses: "does not support:
// reasoning_signature") or sent untranslatable (Messages: thinking without a
// signature, which the provider rejects), so a conversation that switched
// models failed its next turn. Neither is the client's error.
//
// What a row drops is removed. What it transforms keeps its text and loses its
// signature. What it carries is left exactly as it is:
// unsigned reasoning stays reasoning for a Chat target, which encodes it as
// reasoning_content. Nothing here ever turns reasoning into visible text. A
// message left with no content is removed. Visible text, tool calls and tool
// results are never touched.
func CarryReasoningTo(request *llmprotocol.Request, target llmprotocol.WireFormat) ReasoningCarry {
	var carry ReasoningCarry
	if request == nil {
		return carry
	}
	// Fresh slices: the request a dispatch prepares may share its backing
	// arrays with one prepared for another target.
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		if len(message.Content) == 0 {
			messages = append(messages, message)
			continue
		}
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			if path := reasoningProvenance(content); path != "" {
				switch dispositionFor(path, target).Action {
				case dispositionDrop:
					carry.count(path)
					continue
				case dispositionTransform:
					// A reasoning row's transform strips the signature the
					// target cannot verify and keeps the text.
					content.Signature = ""
					carry.SignaturesStripped++
				}
			}
			contents = append(contents, content)
		}
		if len(contents) == 0 {
			continue
		}
		message.Content = contents
		messages = append(messages, message)
	}
	request.Messages = messages
	return carry
}

// OpaqueReasoningData returns the data of a carried Anthropic redacted_thinking
// block: reasoning only its issuing model can read.
func OpaqueReasoningData(content llmprotocol.Content) (string, bool) {
	block := content.Unmodeled
	if content.Kind != llmprotocol.ContentUnmodeled || block == nil ||
		block.Format != llmprotocol.AnthropicMessagesV1 || block.Type != "redacted_thinking" {
		return "", false
	}
	var wire struct {
		Data string `json:"data"`
	}
	if json.Unmarshal(block.Raw, &wire) != nil || wire.Data == "" {
		return "", false
	}
	return wire.Data, true
}

// ResponseOpaqueReasoning lists the opaque reasoning blocks a response holds.
func ResponseOpaqueReasoning(response *llmprotocol.Response) []string {
	if response == nil {
		return nil
	}
	var blocks []string
	for _, item := range response.Output {
		for _, content := range item.Content {
			if data, ok := OpaqueReasoningData(content); ok {
				blocks = append(blocks, data)
			}
		}
	}
	return blocks
}

// DropOpaqueReasoning removes, on fresh slices, every opaque reasoning block
// drop selects from a request's history, and any message it leaves empty. It
// returns how many blocks it removed.
func DropOpaqueReasoning(request *llmprotocol.Request, drop func(data string) bool) int {
	if request == nil {
		return 0
	}
	dropped := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			if data, ok := OpaqueReasoningData(content); ok && drop(data) {
				dropped++
				continue
			}
			contents = append(contents, content)
		}
		if len(contents) == 0 && len(message.Content) > 0 {
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

// reasoningDetailsFormatAnthropic is the format OpenRouter tags Claude's
// reasoning_details items with.
const reasoningDetailsFormatAnthropic = "anthropic-claude-v1"

// DropReasoningNotFromAnthropic removes, on fresh slices, the reasoning in a
// request's history that is not provably Claude's, and any message it leaves
// empty. It returns how many reasoning contents it removed or pruned. It is for a
// target whose model is Claude, whatever the wire format.
//
// Reasoning is provably Claude's only by what Claude alone writes: a thinking
// signature, or reasoning_details items OpenRouter tagged
// anthropic-claude-v1. Everything else (unsigned thinking, reasoning_content,
// reasoning_details of any other format or of none) was written by another
// model, or by one the request cannot name. Handed to Claude it is not
// continuity: Claude cannot use thinking it cannot verify, and Anthropic
// answers another model's reasoning in an assistant turn with a
// content_filter refusal. A reasoning_details array that mixes formats keeps
// only Claude's items, and the content goes when none are left.
func DropReasoningNotFromAnthropic(request *llmprotocol.Request) int {
	if request == nil {
		return 0
	}
	dropped := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			if content.Kind != llmprotocol.ContentReasoning || content.Signature != "" {
				contents = append(contents, content)
				continue
			}
			kept, ok := anthropicReasoningDetails(content)
			if !ok {
				dropped++
				continue
			}
			if kept.Extensions != content.Extensions {
				dropped++
			}
			contents = append(contents, kept)
		}
		if len(contents) == 0 && len(message.Content) > 0 {
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

// anthropicReasoningDetails returns an unsigned reasoning content with only
// the reasoning_details items tagged as Claude's, and false when it has none.
func anthropicReasoningDetails(content llmprotocol.Content) (llmprotocol.Content, bool) {
	details, _ := reasoningDetailsOf(content)
	if details == nil {
		return content, false
	}
	var items []json.RawMessage
	if json.Unmarshal(details, &items) != nil {
		return content, false
	}
	kept := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		var tag struct {
			Format string `json:"format"`
		}
		if json.Unmarshal(item, &tag) == nil && tag.Format == reasoningDetailsFormatAnthropic {
			kept = append(kept, item)
		}
	}
	if len(kept) == 0 {
		return content, false
	}
	if len(kept) == len(items) {
		return content, true
	}
	filtered, err := json.Marshal(kept)
	if err != nil {
		return content, false
	}
	carrier := *content.Extensions
	carrier.Fields = make(map[string]json.RawMessage, len(content.Extensions.Fields))
	for name, value := range content.Extensions.Fields {
		carrier.Fields[name] = value
	}
	carrier.Fields[chatReasoningDetailsMember] = filtered
	content.Extensions = &carrier
	return content, true
}
