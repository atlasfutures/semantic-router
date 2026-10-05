package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The request encoders refuse a Router signature as a second guard behind
// the decoder strip: a marker reaching one means some path carried a
// signature around the decoders, and it must still never reach a provider.
// Each refusal is logged as a warning, with the target and nothing else.
func reportRouterSignatureRefused(target llmprotocol.WireFormat, family string) {
	logging.ComponentWarnEvent("protocolcodec", "thinking_marker_refused", map[string]interface{}{
		"family": thinkingMarkerFamilyLabel(family),
		"target": string(target),
	})
}

// withoutRouterSignedThinking drops, on fresh slices, each thinking block an
// Anthropic Messages request would carry under a Router signature, and any
// message left with no content. The block goes whole: Anthropic refuses
// thinking it did not sign, and a Messages target is never sent unsigned
// thinking (CarryReasoningTo). messages is returned as it was when nothing is
// dropped.
func withoutRouterSignedThinking(messages []llmprotocol.Message) []llmprotocol.Message {
	var kept []llmprotocol.Message
	for index, message := range messages {
		contents := withoutRouterSignedContents(message.Content)
		if kept == nil && len(contents) == len(message.Content) {
			continue
		}
		if kept == nil {
			kept = append(make([]llmprotocol.Message, 0, len(messages)), messages[:index]...)
		}
		if len(contents) == 0 && len(message.Content) > 0 {
			continue
		}
		message.Content = contents
		kept = append(kept, message)
	}
	if kept == nil {
		return messages
	}
	return kept
}

func withoutRouterSignedContents(contents []llmprotocol.Content) []llmprotocol.Content {
	var kept []llmprotocol.Content
	for index, content := range contents {
		family, reserved := routerSignature(content.Signature)
		if content.Kind != llmprotocol.ContentReasoning || !reserved {
			if kept != nil {
				kept = append(kept, content)
			}
			continue
		}
		reportRouterSignatureRefused(llmprotocol.AnthropicMessagesV1, family)
		if kept == nil {
			kept = append(make([]llmprotocol.Content, 0, len(contents)), contents[:index]...)
		}
	}
	if kept == nil {
		return contents
	}
	return kept
}

// providerReasoningSignature is the signature a Responses request may carry
// for a reasoning content: its own, or none for a Router signature. The
// thinking text stays, as the unsigned reasoning a Responses target is sent.
func providerReasoningSignature(signature string) string {
	family, reserved := routerSignature(signature)
	if !reserved {
		return signature
	}
	reportRouterSignatureRefused(llmprotocol.OpenAIResponsesV1, family)
	return ""
}

// withoutRouterSignedDetails removes each reasoning_details item whose
// signature is a Router signature, so a Chat request never carries one. The
// array is returned byte for byte when no item holds one, and nil when every
// item did.
func withoutRouterSignedDetails(details json.RawMessage) (json.RawMessage, error) {
	var items []json.RawMessage
	if err := json.Unmarshal(details, &items); err != nil {
		return nil, err
	}
	kept := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		var signed struct {
			Signature any `json:"signature"`
		}
		if json.Unmarshal(item, &signed) == nil {
			if signature, ok := signed.Signature.(string); ok {
				if family, reserved := routerSignature(signature); reserved {
					reportRouterSignatureRefused(llmprotocol.OpenAIChatV1, family)
					continue
				}
			}
		}
		kept = append(kept, item)
	}
	if len(kept) == len(items) {
		return details, nil
	}
	if len(kept) == 0 {
		return nil, nil
	}
	return json.Marshal(kept)
}
