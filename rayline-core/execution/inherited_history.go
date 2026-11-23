package execution

import (
	"encoding/json"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
)

// A branch the core has no state for can arrive with assistant or tool
// history: a harness resumed or forked a session this process never served.
// That history is accepted as unattributed and a new ledger starts with the
// turn. Its model-bound reasoning (thinking signatures, redacted thinking,
// encrypted Responses reasoning, Chat reasoning_details) was issued by a
// model the core cannot name, so it is never replayed as signed reasoning:
// the core records a digest of each such block and neutralises the message
// holding it on this and every later turn of the branch.

// inheritedReasoningKeys lists the model-bound reasoning tokens a content
// carries, each prefixed by its kind.
func inheritedReasoningKeys(content llmprotocol.Content) []string {
	var keys []string
	switch content.Kind {
	case llmprotocol.ContentReasoning:
		if content.Signature != "" {
			keys = append(keys, "signature\x00"+content.Signature)
		}
		if details, ok := chatReasoningDetails(content); ok {
			keys = append(keys, "details\x00"+string(details))
		}
	case llmprotocol.ContentUnmodeled:
		if data, ok := protocolcodec.OpaqueReasoningData(content); ok {
			keys = append(keys, "redacted\x00"+data)
		}
		if blob, ok := encryptedResponsesReasoning(content); ok {
			keys = append(keys, "encrypted\x00"+blob)
		}
	}
	return keys
}

func chatReasoningDetails(content llmprotocol.Content) (json.RawMessage, bool) {
	extension := content.Extensions
	if extension == nil || extension.Format != llmprotocol.OpenAIChatV1 {
		return nil, false
	}
	details, ok := extension.Fields["reasoning_details"]
	return details, ok
}

func encryptedResponsesReasoning(content llmprotocol.Content) (string, bool) {
	block := content.Unmodeled
	if block == nil || block.Format != llmprotocol.OpenAIResponsesV1 || block.Type != "reasoning" {
		return "", false
	}
	var item struct {
		EncryptedContent json.RawMessage `json:"encrypted_content"`
	}
	if json.Unmarshal(block.Raw, &item) != nil || len(item.EncryptedContent) == 0 || string(item.EncryptedContent) == "null" {
		return "", false
	}
	return string(item.EncryptedContent), true
}

// inheritHistory records the model-bound reasoning of a history the branch
// received without a ledger.
func (state *branchState) inheritHistory(request llmprotocol.Request) {
	for _, message := range request.Messages {
		for _, content := range message.Content {
			for _, key := range inheritedReasoningKeys(content) {
				if state.inherited == nil {
					state.inherited = make(map[string]struct{})
				}
				state.inherited[reasoning.ReasoningBlockDigest(key)] = struct{}{}
			}
		}
	}
}

// neutraliseInheritedReasoning strips, on fresh slices, the model-bound
// reasoning of every message holding a block recorded by inheritHistory:
// signatures and reasoning_details are removed with their readable text kept,
// and redacted or encrypted blocks are dropped. The whole message is treated
// alike, as Anthropic rejects an assistant turn that keeps only part of its
// thinking. Visible text, tool calls and tool results are never touched. A
// message left with no content is removed. It returns how many blocks it
// changed.
func neutraliseInheritedReasoning(request *llmprotocol.Request, inherited map[string]struct{}) int {
	if len(inherited) == 0 {
		return 0
	}
	changed := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		if !holdsInheritedReasoning(message, inherited) {
			messages = append(messages, message)
			continue
		}
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			if _, ok := protocolcodec.OpaqueReasoningData(content); ok {
				changed++
				continue
			}
			if _, ok := encryptedResponsesReasoning(content); ok {
				changed++
				continue
			}
			if content.Kind == llmprotocol.ContentReasoning {
				var stripped int
				content, stripped = withoutReasoningProof(content)
				changed += stripped
				if content.Text == "" && content.Extensions == nil {
					continue
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
	return changed
}

func holdsInheritedReasoning(message llmprotocol.Message, inherited map[string]struct{}) bool {
	for _, content := range message.Content {
		for _, key := range inheritedReasoningKeys(content) {
			if _, ok := inherited[reasoning.ReasoningBlockDigest(key)]; ok {
				return true
			}
		}
	}
	return false
}

// withoutReasoningProof removes what binds a reasoning content to the model
// that issued it, its signature and Chat reasoning_details, keeping the text.
// It returns how many of the two it removed.
func withoutReasoningProof(content llmprotocol.Content) (llmprotocol.Content, int) {
	stripped := 0
	if content.Signature != "" {
		content.Signature = ""
		stripped++
	}
	if _, ok := chatReasoningDetails(content); ok {
		fields := make(map[string]json.RawMessage, len(content.Extensions.Fields))
		for key, value := range content.Extensions.Fields {
			if key != "reasoning_details" {
				fields[key] = value
			}
		}
		format := content.Extensions.Format
		content.Extensions = nil
		if len(fields) > 0 {
			content.Extensions = &llmprotocol.UnmodeledFields{Format: format, Fields: fields}
		}
		stripped++
	}
	return content, stripped
}
