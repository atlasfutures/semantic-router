package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// acceptedAnthropicProjection is the diagnostic an Anthropic control the
// selected Chat or Responses backend cannot honour leaves behind when it is
// dropped or approximated instead of refused (decision 36, US-003e/g/h).
func acceptedAnthropicProjection(
	target llmprotocol.WireFormat,
	field string,
	action llmprotocol.DiagnosticAction,
	reason string,
) llmprotocol.Diagnostic {
	return llmprotocol.Diagnostic{
		Source: llmprotocol.AnthropicMessagesV1,
		Target: target,
		Field:  field,
		Action: action,
		Reason: reason,
	}
}
