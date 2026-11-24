package extproc

import (
	"os"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
)

// A Chat Completions provider that counts reasoning tokens beside
// completion_tokens is settled on what it stated: the turn's counts are
// valid, and completion is the visible output plus the reasoning.
func TestReasoningOutsideCompletionIsSettled(t *testing.T) {
	body, err := os.ReadFile("../../../../rayline-core/protocolcodec/testdata/provider/vertex/chat-response-grok.json")
	if err != nil {
		t.Fatal(err)
	}
	result, err := protocolcodec.NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body,
		func(response *llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	usage := responseUsageFromSemanticUsage(result.Response.Usage)
	if usage.invalid {
		t.Fatalf("settlement refused the usage: %s", usage.invalidReason)
	}
	if usage.promptTokens != 182 || usage.completionTokens != 8+320 || usage.totalTokens != 510 ||
		usage.cachedPromptTokens != 128 {
		t.Fatalf("settled usage = %+v", usage)
	}
}
