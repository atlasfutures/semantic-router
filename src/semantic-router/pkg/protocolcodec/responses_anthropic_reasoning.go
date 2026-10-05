package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude's thinking travels over Responses in the shape OpenRouter's own
// /api/v1/responses gives it (measured live with anthropic models,
// atlasfutures/semantic-router#164): a reasoning item whose reasoning_text is
// the thinking, with two members the Responses API does not name,
//
//	{"type":"reasoning","content":[{"type":"reasoning_text","text":"..."}],
//	 "summary":[],"signature":"<sig>","format":"anthropic-claude-v1"}
//
// and a redacted_thinking block's data in encrypted_content under the same
// format. The signature is what lets Claude keep its thinking on the next
// turn: replayed verbatim the thinking is kept, without it the provider
// silently ignores the thinking, and a corrupted one is a 400. So such an item
// decodes to signed thinking (and opaque reasoning to the carried Anthropic
// redacted_thinking block), which a Messages worker is sent as
// thinking{thinking, signature} and redacted_thinking{data}. Whether a worker
// may see the signature at all is decided at dispatch, by CarryReasoningTo
// and the opaque-reasoning provenance, exactly as for a Messages client's
// Claude thinking.
const responsesAnthropicReasoningFormat = "anthropic-claude-v1"

// responsesAnthropicReasoning holds the members a format-tagged reasoning item
// carries beyond the Responses contract.
type responsesAnthropicReasoning struct {
	signature string
	redacted  string
}

// splitResponsesAnthropicReasoning separates a reasoning item tagged with the
// Anthropic format from its signature, format and encrypted_content, and
// returns the rest of the item for the ordinary decoder. ok is false for any
// other item, which is left as it was: an untagged reasoning item never gains
// a signature.
func splitResponsesAnthropicReasoning(body json.RawMessage) (json.RawMessage, responsesAnthropicReasoning, bool) {
	var object map[string]json.RawMessage
	if json.Unmarshal(body, &object) != nil {
		return body, responsesAnthropicReasoning{}, false
	}
	var itemType, format string
	if json.Unmarshal(object["type"], &itemType) != nil || itemType != "reasoning" ||
		json.Unmarshal(object["format"], &format) != nil || format != responsesAnthropicReasoningFormat {
		return body, responsesAnthropicReasoning{}, false
	}
	var members responsesAnthropicReasoning
	if raw := object["signature"]; hasJSONValue(raw) && json.Unmarshal(raw, &members.signature) != nil {
		return body, responsesAnthropicReasoning{}, false
	}
	if raw := object["encrypted_content"]; hasJSONValue(raw) && json.Unmarshal(raw, &members.redacted) != nil {
		return body, responsesAnthropicReasoning{}, false
	}
	for _, name := range []string{"format", "signature", "encrypted_content"} {
		delete(object, name)
	}
	rest, err := json.Marshal(object)
	if err != nil {
		return body, responsesAnthropicReasoning{}, false
	}
	return rest, members, true
}

// applyTo puts the signature and the opaque reasoning onto an item's decoded
// contents. A signature proves one thinking block, so the item's reasoning
// text becomes that one block; an item with none is thinking whose text was
// omitted, which Anthropic also signs. Summaries are left as they are.
func (members responsesAnthropicReasoning) applyTo(contents []llmprotocol.Content) []llmprotocol.Content {
	if signature := clientThinkingSignature(members.signature, llmprotocol.OpenAIResponsesV1); signature != "" {
		contents = withThinkingSignature(contents, signature)
	}
	if members.redacted != "" {
		contents = append(contents, redactedThinkingContent(members.redacted))
	}
	return contents
}

func withThinkingSignature(contents []llmprotocol.Content, signature string) []llmprotocol.Content {
	signed := make([]llmprotocol.Content, 0, len(contents)+1)
	thinking := -1
	for _, content := range contents {
		if content.Kind == llmprotocol.ContentReasoning && content.Reasoning != llmprotocol.ReasoningScopeSummary {
			if thinking >= 0 {
				signed[thinking].Text += content.Text
				continue
			}
			thinking = len(signed)
			content.Signature = signature
		}
		signed = append(signed, content)
	}
	if thinking < 0 {
		signed = append(signed, llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeText, Signature: signature,
		})
	}
	return signed
}

// redactedThinkingContent is the carried Anthropic block a Messages client
// would have sent for the same opaque reasoning, so the provenance and drop
// rules for it (OpaqueReasoningData) apply unchanged.
func redactedThinkingContent(data string) llmprotocol.Content {
	raw, _ := json.Marshal(struct {
		Type string `json:"type"`
		Data string `json:"data"`
	}{Type: "redacted_thinking", Data: data})
	return llmprotocol.Content{Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
		Format: llmprotocol.AnthropicMessagesV1, Type: "redacted_thinking", Raw: raw,
	}}
}
