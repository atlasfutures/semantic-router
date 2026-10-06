package extproc

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Chat client streaming with include_usage from a Messages-only arm (as
// arc-0.3's Claude arms are) gets a usage chunk the cell synthesizes. It must
// carry "choices": [] beside usage, as OpenAI's does: opencode's stream schema
// refuses the chunk without the key.
func TestChatStreamFromMessagesArmUsageChunkCarriesChoices(t *testing.T) {
	includeUsage := true
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{
			Generation:    1,
			Model:         "public-model",
			Stream:        true,
			StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
		},
	}
	response := router.handleSemanticStreamingResponseBody(
		extProcStreamFixture(llmprotocol.AnthropicMessagesV1), true, ctx,
	)
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("a translated stream produced no client body")
	}
	usageChunks := 0
	for _, line := range strings.Split(string(mutation.GetBody()), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk map[string]json.RawMessage
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk is not JSON: %s", data)
		}
		if _, usage := chunk["usage"]; !usage {
			continue
		}
		usageChunks++
		if !bytes.Equal(chunk["choices"], []byte("[]")) {
			t.Fatalf("usage chunk choices = %q, want []: %s", chunk["choices"], data)
		}
	}
	if usageChunks != 1 {
		t.Fatalf("usage chunks = %d, want 1:\n%s", usageChunks, mutation.GetBody())
	}
}

// A Chat arm's own stream reaches a Chat client that asked for usage
// unchanged: the fix is in the synthesized chunk, not a rewrite of passed-
// through ones.
func TestSameFormatChatStreamWithUsageIsPassedThrough(t *testing.T) {
	includeUsage := true
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{
			Generation:    1,
			Model:         "public-model",
			Stream:        true,
			StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
		},
	}
	upstream := extProcStreamFixture(llmprotocol.OpenAIChatV1)
	response := router.handleSemanticStreamingResponseBody(upstream, true, ctx)
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil &&
		!bytes.Equal(mutation.GetBody(), upstream) {
		t.Fatalf("a same-format stream was rewritten:\n got %q\nwant %q", mutation.GetBody(), upstream)
	}
}
