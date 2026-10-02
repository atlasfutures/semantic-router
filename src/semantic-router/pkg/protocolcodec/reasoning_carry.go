package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// ReasoningCarry counts what CarryReasoningTo changed in a request's history.
type ReasoningCarry struct {
	// UnsignedDropped is reasoning dropped because a Messages target accepts
	// thinking only with the signature that proves its issuer, and this
	// reasoning had none: another provider's thinking (kimi on OpenRouter
	// sends it empty) or a Chat or Responses reasoning echoed back.
	UnsignedDropped int
	// SignaturesStripped is Anthropic signatures removed from reasoning sent
	// to a Chat or Responses target, which has nowhere to put one. The
	// reasoning text itself is kept.
	SignaturesStripped int
}

// Changed reports whether the request was changed.
func (carry ReasoningCarry) Changed() bool {
	return carry.UnsignedDropped > 0 || carry.SignaturesStripped > 0
}

// CarryReasoningTo makes the reasoning in a request's history carriable by
// the target wire format, in place. Reasoning a target cannot carry as it is
// was refused (Chat and Responses: "does not support: reasoning_signature")
// or sent untranslatable (Messages: thinking without a signature, which the
// provider rejects), so a conversation that switched models failed its next
// turn. Neither is the client's error, and the reasoning is the previous
// model's private working, so it is neutralised: a signature a target cannot
// hold is stripped, and reasoning a Messages target cannot verify is dropped.
// A message left with no content is removed. Visible text, tool calls and
// tool results are never touched.
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
			if content.Kind == llmprotocol.ContentReasoning {
				switch {
				case target == llmprotocol.AnthropicMessagesV1 && content.Signature == "":
					carry.UnsignedDropped++
					continue
				case target != llmprotocol.AnthropicMessagesV1 && content.Signature != "":
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
