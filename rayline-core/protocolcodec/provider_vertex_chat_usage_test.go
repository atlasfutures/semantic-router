package protocolcodec

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func loadVertexFixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "provider", "vertex", name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// xAI counts reasoning tokens beside completion_tokens, not inside it:
// total_tokens = prompt + completion + reasoning. Read as OpenAI's shape, the
// totals disagree and the turn failed (usage_total_mismatch) although the
// provider answered. Read by its total, every count is the provider's own and
// the client is billed the completion plus the reasoning.
func TestReasoningCountedOutsideCompletionIsReconciledByItsTotal(t *testing.T) {
	t.Run("non-streamed", func(t *testing.T) {
		body := loadVertexFixture(t, "chat-response-grok.json")
		billed, response := vertexAnthropicUsageFromResponse(t, body)
		usage := response.Usage
		for name, check := range map[string]struct {
			count llmprotocol.TokenCount
			value int64
			from  llmprotocol.UsageProvenance
		}{
			"output total":     {usage.OutputTotal, 16 + 224, llmprotocol.UsageAuthoritative},
			"output reasoning": {usage.OutputReasoning, 224, llmprotocol.UsageAuthoritative},
			"output other":     {usage.OutputOther, 16, llmprotocol.UsageDerived},
			"total":            {usage.Total, 460, llmprotocol.UsageAuthoritative},
			"input total":      {usage.InputTotal, 220, llmprotocol.UsageAuthoritative},
		} {
			if check.count.Value == nil || *check.count.Value != check.value || check.count.Provenance != check.from {
				t.Errorf("%s = %+v, want %d (%s)", name, check.count, check.value, check.from)
			}
		}
		// The client is billed the reasoning as output: 16 + 224.
		assertArmUsage(t, billed, armUsage(220, 240, 0, 0, 224))
	})
	t.Run("streamed", func(t *testing.T) {
		stream := mustNewMatrixStream(t, NewBuiltinEngine(), llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1)
		frames, _, _, err := stream.Push(loadVertexFixture(t, "chat-stream-grok.sse"))
		if err != nil {
			t.Fatalf("push: %v", err)
		}
		finalFrames, _, _, err := stream.Finalize(nil)
		if err != nil {
			t.Fatalf("finalize: %v", err)
		}
		var encoded bytes.Buffer
		writeFrames(&encoded, frames)
		writeFrames(&encoded, finalFrames)
		// 220 prompt tokens, 128 of them cached; output 16 + 172.
		assertArmUsage(t, messageDeltaUsage(t, "chat-stream-grok.sse", encoded.Bytes()), armUsage(92, 188, 128, 0, 172))
	})
}

// A total that matches neither reading is still refused: the reconciliation
// is decided by the total, never assumed.
func TestReasoningOutsideCompletionNeedsItsTotal(t *testing.T) {
	for name, usage := range map[string]chatUsageWire{
		"total matches neither": {PromptTokens: 10, CompletionTokens: 5, TotalTokens: 99,
			CompletionTokensDetails: &chatCompletionTokensDetailsWire{ReasoningTokens: 20}},
		"OpenAI's inclusive shape": {PromptTokens: 10, CompletionTokens: 25, TotalTokens: 35,
			CompletionTokensDetails: &chatCompletionTokensDetailsWire{ReasoningTokens: 20}},
		"no reasoning": {PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15,
			CompletionTokensDetails: &chatCompletionTokensDetailsWire{ReasoningTokens: 0}},
	} {
		if chatReasoningOutsideCompletion(usage) {
			t.Errorf("%s: read as reasoning outside completion", name)
		}
	}
	inclusive := decodeChatUsage(chatUsageWire{PromptTokens: 10, CompletionTokens: 25, TotalTokens: 35,
		CompletionTokensDetails: &chatCompletionTokensDetailsWire{ReasoningTokens: 20}})
	if *inclusive.OutputTotal.Value != 25 || *inclusive.OutputReasoning.Value != 20 || *inclusive.OutputOther.Value != 5 {
		t.Fatalf("inclusive shape changed: %+v", inclusive)
	}
}

// Some serving stacks put non-string values in the response's metadata
// (weight_versions as an array). Metadata is request metadata, read only for
// its presence, so the response decodes instead of failing as invalid JSON.
func TestChatResponseMetadataOfAnyTypeDecodes(t *testing.T) {
	body := loadVertexFixture(t, "chat-response-deepseek-flash.json")
	// The engine validates the decoded response, usage totals included.
	billed, _ := vertexAnthropicUsageFromResponse(t, body)
	assertArmUsage(t, billed, armUsage(18, 17, 0, 0, 0))

	// Streamed, the chunks carry no metadata; the capture is pinned all the same.
	stream := mustNewMatrixStream(t, NewBuiltinEngine(), llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1)
	frames, _, _, err := stream.Push(loadVertexFixture(t, "chat-stream-deepseek-flash.sse"))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	var encoded bytes.Buffer
	writeFrames(&encoded, frames)
	writeFrames(&encoded, finalFrames)
	assertArmUsage(t, messageDeltaUsage(t, "chat-stream-deepseek-flash.sse", encoded.Bytes()), armUsage(18, 14, 0, 0, 0))
}

// vertexAnthropicUsageFromResponse translates a provider body to Messages as
// the cell does (decode, validate, encode) and returns the usage the client
// is billed and the neutral response behind it.
func vertexAnthropicUsageFromResponse(t *testing.T, body []byte) (armUsageWire, llmprotocol.Response) {
	t.Helper()
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body,
		func(response *llmprotocol.Response) error {
			response.Model = "public-model"
			return nil
		},
	)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var translated struct {
		Usage armUsageWire `json:"usage"`
	}
	if err := json.Unmarshal(result.Body, &translated); err != nil {
		t.Fatalf("decode translated body: %v\n%s", err, result.Body)
	}
	return translated.Usage, result.Response
}

// A stream can report usage more than once. An early update with no
// reasoning reads OpenAI's way, a later one beside completion_tokens: the
// output total only grows and stays authoritative, so the merge accepts it.
func TestReasoningOutsideCompletionAfterAnEarlierUsageUpdate(t *testing.T) {
	stream := mustNewMatrixStream(t, NewBuiltinEngine(), llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1)
	chunk := func(content, usage string) string {
		return `data: {"id":"fixture-grok-stream","object":"chat.completion.chunk","created":1795540101,"model":"xai/grok-4.7",` +
			`"choices":[{"index":0,"delta":{"content":"` + content + `"},"finish_reason":null}],"usage":` + usage + "}\n\n"
	}
	// Inclusive (no reasoning), then beside completion, then a restatement
	// with no reasoning again: every output total grows.
	body := chunk("Hi", `{"prompt_tokens":10,"completion_tokens":0,"total_tokens":10,"completion_tokens_details":{"reasoning_tokens":0}}`) +
		chunk(" there", `{"prompt_tokens":10,"completion_tokens":5,"total_tokens":35,"completion_tokens_details":{"reasoning_tokens":20}}`) +
		chunk(".", `{"prompt_tokens":10,"completion_tokens":26,"total_tokens":36,"completion_tokens_details":{"reasoning_tokens":0}}`) +
		`data: {"id":"fixture-grok-stream","object":"chat.completion.chunk","created":1795540101,"model":"xai/grok-4.7","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` +
		"\n\ndata: [DONE]\n\n"
	frames, _, _, err := stream.Push([]byte(body))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	var encoded bytes.Buffer
	writeFrames(&encoded, frames)
	writeFrames(&encoded, finalFrames)
	assertArmUsage(t, messageDeltaUsage(t, "three usage updates", encoded.Bytes()), armUsage(10, 26, 0, 0, 0))
}
