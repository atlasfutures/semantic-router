package protocolcodec

import (
	"encoding/json"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
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
	// ClaudeThinking is Claude's thinking put in the shape its worker reads
	// it signed in (CarryReasoningToClaude): blocks rebuilt from
	// anthropic-claude-v1 reasoning_details for a Messages or Responses
	// worker, or signed blocks carried as such items for a Chat worker.
	ClaudeThinking int
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
	return carryReasoningTo(request, target, dispositionFor)
}

// CarryReasoningToClaude is CarryReasoningTo for a Claude worker, whose host
// verifies Anthropic's thinking signature whatever the wire format. The
// caller has already kept only reasoning with Anthropic provenance
// (DropReasoningNotFromAnthropic). Claude's thinking first takes the shape
// the worker reads it signed in: for a Messages or Responses worker the
// anthropic-claude-v1 reasoning_details a Chat client replays, or a Responses
// client resends as the Router's minted encrypted_content, become the signed
// thinking and redacted_thinking blocks again
// (ClaudeThinkingFromReasoningDetails); for a Chat worker each signed block
// becomes the reasoning_details item OpenRouter asks for back
// (SignedThinkingAsReasoningDetails). Then the table applies, except that a
// Responses worker is sent signed thinking and redacted_thinking as they are:
// OpenRouter's /responses takes both back on a reasoning item
// (responses_anthropic_reasoning.go), where the table's rows strip and drop
// them for a Responses host that cannot verify a signature.
func CarryReasoningToClaude(request *llmprotocol.Request, target llmprotocol.WireFormat) ReasoningCarry {
	if request == nil {
		return ReasoningCarry{}
	}
	var shaped int
	if target == llmprotocol.OpenAIChatV1 {
		shaped = SignedThinkingAsReasoningDetails(request)
	} else {
		shaped = ClaudeThinkingFromReasoningDetails(request)
	}
	carry := carryReasoningTo(request, target, claudeHostDispositionFor)
	carry.ClaudeThinking = shaped
	return carry
}

// claudeHostDispositionFor is dispositionFor with the Responses rows a Claude
// host verifies for itself carried.
func claudeHostDispositionFor(path string, target llmprotocol.WireFormat) targetDisposition {
	if target == llmprotocol.OpenAIResponsesV1 && (path == fieldReasoningSigned || path == fieldRedactedThinking) {
		return targetDisposition{Action: dispositionCarry}
	}
	return dispositionFor(path, target)
}

func carryReasoningTo(
	request *llmprotocol.Request,
	target llmprotocol.WireFormat,
	disposition func(path string, target llmprotocol.WireFormat) targetDisposition,
) ReasoningCarry {
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
				switch disposition(path, target).Action {
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
// drop selects from a request's history, and any message it leaves empty. A
// turn that loses an opaque block loses its other reasoning too, thinking and
// redacted alike: Anthropic accepts an assistant turn with every thinking
// block removed and rejects one with some removed, so a turn this cannot
// forward whole is not forwarded in part. A turn runs from an assistant
// message to the next user message, across the tool results between: to
// Anthropic a tool-use loop is one assistant turn, whose thinking must come
// back whole with every tool result in it, and a Responses history holds a
// turn's reasoning items, message and function calls as a message each
// (regroupAnthropicMessages). Visible text and tool calls are never touched.
// It returns how many blocks it removed.
func DropOpaqueReasoning(request *llmprotocol.Request, drop func(data string) bool) int {
	if request == nil {
		return 0
	}
	dropped := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for start := 0; start < len(request.Messages); {
		end := assistantTurnEnd(request.Messages, start)
		turn := request.Messages[start:end]
		start = end
		if !holdsOpaqueReasoning(turn, drop) {
			messages = append(messages, turn...)
			continue
		}
		for _, message := range turn {
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
	}
	if dropped > 0 {
		request.Messages = messages
	}
	return dropped
}

// assistantTurnEnd returns the index after the turn that starts at start: the
// run of assistant and tool-result messages from it, or the one message of
// another role. Every decoder gives a tool result a message of its own, with
// RoleTool, so a tool-use loop is one run.
func assistantTurnEnd(messages []llmprotocol.Message, start int) int {
	end := start + 1
	if messages[start].Role != llmprotocol.RoleAssistant {
		return end
	}
	for end < len(messages) && (messages[end].Role == llmprotocol.RoleAssistant || messages[end].Role == llmprotocol.RoleTool) {
		end++
	}
	return end
}

// holdsOpaqueReasoning reports whether a turn holds an opaque reasoning block
// that drop selects.
func holdsOpaqueReasoning(turn []llmprotocol.Message, drop func(data string) bool) bool {
	for _, message := range turn {
		for _, content := range message.Content {
			if data, ok := OpaqueReasoningData(content); ok && drop(data) {
				return true
			}
		}
	}
	return false
}

// reasoningDetailsFormatAnthropic is the format OpenRouter tags Claude's
// reasoning_details items with.
const reasoningDetailsFormatAnthropic = "anthropic-claude-v1"

// DropReasoningNotFromAnthropic removes, on fresh slices, the reasoning in a
// request's history that is not provably Claude's, and any message it leaves
// empty. It returns how many reasoning contents it removed or pruned. It is
// for a worker whose model is Claude, whatever the wire format. source is the
// format the client sent the request in.
//
// Reasoning is provably Claude's only by what Claude alone writes, with its
// provenance:
//   - a thinking signature a Messages client sent on a thinking block, or one
//     a Responses client sent on a reasoning item tagged anthropic-claude-v1
//     (the only item the Responses decoder gives a signature); a signature on
//     a request from any other source is not proof;
//   - reasoning_details items OpenRouter tagged anthropic-claude-v1;
//   - an encrypted Responses blob, which the issuer record already sends only
//     back to the target that issued it.
//
// Everything else (unsigned thinking, reasoning_content, reasoning_details of
// any other format or of none, a Responses reasoning item signed under
// another format or none) was written by another model, or by one the request
// cannot name. Handed to Claude it is not continuity: Claude cannot use
// thinking it cannot verify, and Anthropic answers another model's reasoning
// in an assistant turn with a content_filter refusal. A reasoning_details
// array that mixes formats keeps only Claude's items, and its text is rebuilt
// from them, so no other model's text rides along as reasoning_content; the
// content goes when no Claude item is left.
//
// A block's fate depends only on the block and source, so a history gives the
// same result on every turn. A message is removed only when nothing is left
// in it, and an assistant turn that called a tool always keeps its call, so a
// tool result never loses the call it answers.
func DropReasoningNotFromAnthropic(request *llmprotocol.Request, source llmprotocol.WireFormat) int {
	if request == nil {
		return 0
	}
	signatureIsProof := source == llmprotocol.AnthropicMessagesV1 || source == llmprotocol.OpenAIResponsesV1
	dropped := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			if foreignResponsesReasoningItem(content) {
				dropped++
				continue
			}
			if content.Kind != llmprotocol.ContentReasoning || (content.Signature != "" && signatureIsProof) {
				contents = append(contents, content)
				continue
			}
			kept, ok := anthropicReasoningDetails(content)
			if !ok {
				dropped++
				continue
			}
			if kept.Extensions != content.Extensions || kept.Text != content.Text {
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

// foreignResponsesReasoningItem reports whether a content is a Responses
// reasoning item the decoder carried whole, because it is not tagged
// anthropic-claude-v1, and that holds no encrypted_content. An encrypted
// item is left to the issuer record, which forwards a blob only to the target
// that issued it (applyRaylineARCReasoningIssuer) and has every other
// encoder drop it: a Claude worker on a direct Responses backend reads its
// own blobs back.
func foreignResponsesReasoningItem(content llmprotocol.Content) bool {
	block := content.Unmodeled
	return content.Kind == llmprotocol.ContentUnmodeled && block != nil &&
		block.Format == llmprotocol.OpenAIResponsesV1 && block.Type == "reasoning" &&
		!hasResponsesEncryptedReasoning(block.Raw)
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
	var text strings.Builder
	for _, item := range items {
		var tag struct {
			Format  string `json:"format"`
			Text    string `json:"text"`
			Summary string `json:"summary"`
		}
		if json.Unmarshal(item, &tag) == nil && tag.Format == reasoningDetailsFormatAnthropic {
			kept = append(kept, item)
			text.WriteString(tag.Text)
			text.WriteString(tag.Summary)
		}
	}
	if len(kept) == 0 {
		return content, false
	}
	if len(kept) == len(items) {
		// Every item is Claude's, but the text beside them is the client's:
		// only the text those items hold is proven. Text beyond it -- such as
		// the reasoning of an item the decoder stripped for a Router marker
		// (clientReasoningDetails) -- is not Claude's.
		if content.Text != "" && content.Text != text.String() {
			content.Text = text.String()
		}
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
	// The text was merged from every item; only Claude's may remain.
	content.Text = text.String()
	return content, true
}
