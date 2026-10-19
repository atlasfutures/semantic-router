package protocolcodec

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A reasoning signature in a response is provider state: Anthropic signs a
// thinking block so a client can resend it and the model keeps its thinking.
// A target contract with no member for one -- Chat Completions -- is given
// the signature where OpenRouter puts Claude's: an anthropic-claude-v1
// reasoning_details item beside the reasoning_content text
// (chat_claude_reasoning.go, atlasfutures/semantic-router#278). Before #278
// the signature was cleared and the text kept, which Anthropic accepts on
// replay but answers without Claude's reasoning. The response is not refused
// either way: a Chat client on a Messages-only Claude arm would otherwise get
// a 502 for every signed answer (atlasfutures/semantic-router#212).
//
// Any other target without the capability is sent the text without the
// signature, as the request leg's CarryReasoningTo does (content.thinking.signed),
// and the drop is reported as a diagnostic. A signature in the Router's
// namespace is never carried: it is cleared the same way for every target.
//
// Only what is encoded for the client changes; the neutral response a caller
// keeps for settlement is left as it was.
const responseReasoningSignatureDropped = "the client contract has no member for a reasoning signature; the thinking text is kept"

// responseForTarget returns the response a target's encoder is given: the
// response itself when the target can carry reasoning signatures or none is
// present, a Chat target's copy with Claude's thinking carried as
// reasoning_details, and otherwise a copy, on fresh slices, without the
// signatures.
func responseForTarget(
	response llmprotocol.Response,
	target llmprotocol.WireFormat,
	capabilities llmprotocol.CapabilitySet,
	limit int,
) (llmprotocol.Response, llmprotocol.Diagnostics) {
	if capabilities.Supports(llmprotocol.CapabilityReasoningSignature) {
		return response, nil
	}
	if target == llmprotocol.OpenAIChatV1 {
		return responseWithClaudeReasoningCarriers(response, target, limit)
	}
	output, stripped := withoutReasoningSignatures(response.Output)
	var alternatives [][]llmprotocol.OutputItem
	for index, alternative := range response.Alternatives {
		kept, strippedHere := withoutReasoningSignatures(alternative)
		if strippedHere == 0 {
			continue
		}
		if alternatives == nil {
			alternatives = append([][]llmprotocol.OutputItem(nil), response.Alternatives...)
		}
		alternatives[index] = kept
		stripped += strippedHere
	}
	if stripped == 0 {
		return response, nil
	}
	response.Output = output
	if alternatives != nil {
		response.Alternatives = alternatives
	}
	return response, appendDiagnostics(nil, llmprotocol.Diagnostics{{
		Target: target, Field: "reasoning.signature",
		Action: llmprotocol.DiagnosticDropped, Reason: responseReasoningSignatureDropped,
	}}, limit)
}

// withoutReasoningSignatures returns the items with every reasoning
// signature removed, copying only what changes, and how many it removed.
func withoutReasoningSignatures(items []llmprotocol.OutputItem) ([]llmprotocol.OutputItem, int) {
	stripped := 0
	var out []llmprotocol.OutputItem
	for index, item := range items {
		var contents []llmprotocol.Content
		for contentIndex, content := range item.Content {
			if content.Kind != llmprotocol.ContentReasoning || content.Signature == "" {
				continue
			}
			if contents == nil {
				contents = append([]llmprotocol.Content(nil), item.Content...)
			}
			contents[contentIndex].Signature = ""
			stripped++
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
		return items, 0
	}
	return out, stripped
}
