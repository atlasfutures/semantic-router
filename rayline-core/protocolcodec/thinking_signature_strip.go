package protocolcodec

import "github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

// ResponseThinkingSignature returns the first thinking signature a response
// holds, or "" when it holds none.
func ResponseThinkingSignature(response *llmprotocol.Response) string {
	if response == nil {
		return ""
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			if signature, ok := providerThinkingSignature(content); ok {
				return signature
			}
		}
	}
	return ""
}

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
