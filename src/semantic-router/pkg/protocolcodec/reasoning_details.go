package protocolcodec

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// This file carries OpenRouter's reasoning_details through the Router.
//
// OpenRouter returns a model's reasoning twice on a Chat Completions message:
// as plain text (reasoning) and as reasoning_details, an ordered array whose
// items hold what the text cannot -- an Anthropic signature, an encrypted
// blob (OpenAI's encrypted reasoning, a Gemini thought signature, an
// Anthropic redacted block), a summary. OpenRouter asks for the array back,
// unmodified, on the next turn: that is how a tool-calling conversation keeps
// its reasoning on the models that need it.
//
// The Router used to prune the member as one the contract did not name, so a
// Chat client never saw it and could not resend it. It is now carried as an
// opaque member of the reasoning content it belongs to (Content.Extensions,
// format Chat), never as Content.Signature: a signature there is one the
// Messages encoder emits and CarryReasoningTo keeps for a Messages target,
// which would hand an OpenRouter blob of any model family to Anthropic. What
// the array may reach is unchanged: a Chat target gets it back, every other
// target drops and counts it (cross-family replay is router-infra#95).
//
// A Responses client has no reasoning_details member. It is given the array
// as a reasoning item's encrypted_content, minted here (mintReasoningDetails),
// and its resend is read back into the same carrier, so the array again
// reaches only a Chat target. A minted blob is never sent to a Responses
// target: it is not that target's blob, and only the issuer can read one.

const chatReasoningDetailsMember = "reasoning_details"

// mintedReasoningDetailsPrefix marks an encrypted_content the Router minted
// from reasoning_details. No provider blob starts with it: OpenAI's are
// base64 of a versioned binary envelope ("gAAAA..."), and the dot is outside
// the base64 alphabet.
const mintedReasoningDetailsPrefix = "vsr.reasoning_details.v1."

// reasoningDetailsFields wraps a reasoning_details array as the carrier a
// reasoning content holds.
func reasoningDetailsFields(details json.RawMessage) *llmprotocol.UnmodeledFields {
	if !hasJSONValue(details) {
		return nil
	}
	return &llmprotocol.UnmodeledFields{
		Format: llmprotocol.OpenAIChatV1,
		Fields: map[string]json.RawMessage{chatReasoningDetailsMember: append(json.RawMessage(nil), details...)},
	}
}

// reasoningDetailsOf returns the reasoning_details a reasoning content
// carries, and whether the content is only that carrier, with no text of its
// own to encode.
func reasoningDetailsOf(content llmprotocol.Content) (json.RawMessage, bool) {
	if content.Kind != llmprotocol.ContentReasoning || content.Extensions == nil ||
		content.Extensions.Format != llmprotocol.OpenAIChatV1 {
		return nil, false
	}
	details := content.Extensions.Fields[chatReasoningDetailsMember]
	if !hasJSONValue(details) {
		return nil, false
	}
	return details, content.Text == ""
}

// holdsReasoningDetails reports whether any content of a message carries
// reasoning_details.
func holdsReasoningDetails(message llmprotocol.Message) bool {
	for _, content := range message.Content {
		if details, _ := reasoningDetailsOf(content); details != nil {
			return true
		}
	}
	return false
}

// decodeReasoningDetailsArray accepts a reasoning_details value that is an
// array of objects, which is every shape OpenRouter documents. It returns nil
// for an absent, null or empty value. ok is false for any other shape.
func decodeReasoningDetailsArray(raw json.RawMessage) (json.RawMessage, bool) {
	if !hasJSONValue(raw) {
		return nil, true
	}
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, false
	}
	for _, item := range items {
		if item == nil {
			return nil, false
		}
	}
	if len(items) == 0 {
		return nil, true
	}
	return append(json.RawMessage(nil), bytes.TrimSpace(raw)...), true
}

// attachReasoningDetails puts a decoded reasoning_details array on the
// message's reasoning content: on the first one when the message has
// reasoning text, otherwise on a reasoning content that holds only the
// carrier.
func attachReasoningDetails(contents []llmprotocol.Content, details json.RawMessage) []llmprotocol.Content {
	carrier := reasoningDetailsFields(details)
	if carrier == nil {
		return contents
	}
	for index := range contents {
		if contents[index].Kind == llmprotocol.ContentReasoning && contents[index].Extensions == nil {
			contents[index].Extensions = carrier
			return contents
		}
	}
	return append(contents, llmprotocol.Content{
		Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeText, Extensions: carrier,
	})
}

// concatReasoningDetails joins two reasoning_details arrays in order. Either
// may be empty.
func concatReasoningDetails(left, right json.RawMessage) (json.RawMessage, error) {
	if !hasJSONValue(left) {
		return append(json.RawMessage(nil), right...), nil
	}
	if !hasJSONValue(right) {
		return append(json.RawMessage(nil), left...), nil
	}
	var leftItems, rightItems []json.RawMessage
	if err := json.Unmarshal(left, &leftItems); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(right, &rightItems); err != nil {
		return nil, err
	}
	return json.Marshal(append(leftItems, rightItems...))
}

// reasoningDetailsStringMembers are the members a streamed reasoning_details
// item spreads over several chunks. Each fragment holds the next piece.
var reasoningDetailsStringMembers = map[string]bool{
	"text": true, "summary": true, "data": true, "signature": true,
}

// mergeReasoningDetailsFragment folds one chunk's reasoning_details into the
// array accumulated so far. Fragments name the item they extend by index; a
// fragment for an index already seen appends its string members to that
// item's and fills members the item does not have yet. A fragment with no
// index, or a new one, is a new item.
func mergeReasoningDetailsFragment(accumulated, fragment json.RawMessage) (json.RawMessage, error) {
	if !hasJSONValue(fragment) {
		return accumulated, nil
	}
	var items []map[string]json.RawMessage
	if hasJSONValue(accumulated) {
		if err := json.Unmarshal(accumulated, &items); err != nil {
			return nil, err
		}
	}
	var pieces []map[string]json.RawMessage
	if err := json.Unmarshal(fragment, &pieces); err != nil {
		return nil, err
	}
	for _, piece := range pieces {
		target := reasoningDetailsItemAt(items, piece["index"])
		if target == nil {
			items = append(items, piece)
			continue
		}
		if err := mergeReasoningDetailsItem(target, piece); err != nil {
			return nil, err
		}
	}
	return json.Marshal(items)
}

func reasoningDetailsItemAt(items []map[string]json.RawMessage, index json.RawMessage) map[string]json.RawMessage {
	if !hasJSONValue(index) {
		return nil
	}
	for _, item := range items {
		if bytes.Equal(bytes.TrimSpace(item["index"]), bytes.TrimSpace(index)) {
			return item
		}
	}
	return nil
}

func mergeReasoningDetailsItem(target, piece map[string]json.RawMessage) error {
	for name, value := range piece {
		existing, present := target[name]
		if !present || !hasJSONValue(existing) {
			target[name] = value
			continue
		}
		if !reasoningDetailsStringMembers[name] || !hasJSONValue(value) {
			continue
		}
		var head, tail string
		if json.Unmarshal(existing, &head) != nil || json.Unmarshal(value, &tail) != nil {
			// Not a string after all: the later value is the provider's last word.
			target[name] = value
			continue
		}
		joined, err := json.Marshal(head + tail)
		if err != nil {
			return err
		}
		target[name] = joined
	}
	return nil
}

// mintReasoningDetails encodes a reasoning_details array as the value of a
// Responses reasoning item's encrypted_content.
func mintReasoningDetails(details json.RawMessage) json.RawMessage {
	compact := bytes.Buffer{}
	if err := json.Compact(&compact, details); err != nil {
		return nil
	}
	blob, err := json.Marshal(mintedReasoningDetailsPrefix + base64.RawURLEncoding.EncodeToString(compact.Bytes()))
	if err != nil {
		return nil
	}
	return blob
}

// mintedReasoningDetails reads an encrypted_content the Router minted back
// into the reasoning_details array it holds. ok is false for any other
// value, a provider's blob included.
func mintedReasoningDetails(encrypted json.RawMessage) (json.RawMessage, bool) {
	var text string
	if !hasJSONValue(encrypted) || json.Unmarshal(encrypted, &text) != nil ||
		!strings.HasPrefix(text, mintedReasoningDetailsPrefix) {
		return nil, false
	}
	decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(text, mintedReasoningDetailsPrefix))
	if err != nil {
		return nil, false
	}
	details, ok := decodeReasoningDetailsArray(decoded)
	if !ok || details == nil {
		return nil, false
	}
	return details, true
}

// isMintedReasoningItem reports whether a Responses input item is a
// reasoning item whose encrypted_content the Router minted.
func isMintedReasoningItem(body json.RawMessage) bool {
	var item struct {
		Type             string          `json:"type"`
		EncryptedContent json.RawMessage `json:"encrypted_content"`
	}
	if json.Unmarshal(body, &item) != nil || item.Type != "reasoning" {
		return false
	}
	_, minted := mintedReasoningDetails(item.EncryptedContent)
	return minted
}

// outputHoldsReasoningDetails reports whether any reasoning in output
// carries reasoning_details.
func outputHoldsReasoningDetails(output []llmprotocol.OutputItem) bool {
	for _, item := range output {
		if holdsReasoningDetails(llmprotocol.Message{Content: item.Content}) {
			return true
		}
	}
	return false
}

// withoutReasoningDetailsCarriers removes reasoning contents that hold only
// reasoning_details, for a client format with nowhere to put them
// (Anthropic Messages). Reasoning that also has text keeps its text.
func withoutReasoningDetailsCarriers(output []llmprotocol.OutputItem) []llmprotocol.OutputItem {
	trimmed := make([]llmprotocol.OutputItem, 0, len(output))
	for _, item := range output {
		contents := make([]llmprotocol.Content, 0, len(item.Content))
		for _, content := range item.Content {
			if _, carrierOnly := reasoningDetailsOf(content); carrierOnly {
				continue
			}
			contents = append(contents, content)
		}
		item.Content = contents
		trimmed = append(trimmed, item)
	}
	return trimmed
}

// appendReasoningDetailsDrops counts each reasoning block whose
// reasoning_details a Messages target is not sent. The encoder keeps the
// reasoning text and drops a block that held only the details.
func appendReasoningDetailsDrops(
	diagnostics *llmprotocol.Diagnostics,
	messages []llmprotocol.Message,
	target llmprotocol.WireFormat,
	policy llmprotocol.Policy,
) {
	for _, message := range messages {
		for _, content := range message.Content {
			if details, _ := reasoningDetailsOf(content); details != nil {
				appendUnmodeledDrop(diagnostics, policy, content.Extensions.Format, target, "content.reasoning_details")
			}
		}
	}
}
