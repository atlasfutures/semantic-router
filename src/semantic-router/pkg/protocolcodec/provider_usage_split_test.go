package protocolcodec

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// testdata/provider/openrouter/chat-response-usage-split.json was captured from
// arm 4, qwen/qwen3.6-35b-a3b, on 2026-09-03 with max_tokens 64. The provider
// reports completion_tokens 64 and reasoning_tokens 79: more reasoning than
// completion. The split cannot be true, but the totals it states can, and the
// client asked for a completion, not a breakdown.
const openRouterUsageSplit = "chat-response-usage-split.json"

func loadUsageFixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "provider", "openrouter", name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// A breakdown that cannot be true must not cost the client its answer.
func TestUnreconcilableUsageStillServesTheResponse(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, target := range map[string]llmprotocol.WireFormat{
		"chat":      llmprotocol.OpenAIChatV1,
		"anthropic": llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(name, func(t *testing.T) {
			result, err := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1, target, loadUsageFixture(t, openRouterUsageSplit),
				func(response *llmprotocol.Response) error {
					response.Model = "public-model"
					return nil
				},
			)
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			if len(result.Body) == 0 {
				t.Fatal("translated body is empty")
			}
		})
	}
}

// The totals the provider stated are authoritative and must survive. Only the
// split it could not state is unknown: reporting a zero would assert a split
// the provider never gave, and accounting reads these numbers.
func TestUnreconcilableUsageKeepsTotalsAndDropsTheSplit(t *testing.T) {
	// The fixture is a real body, so it also carries the members the contract
	// does not name. Translation is the path that tolerates those.
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, loadUsageFixture(t, openRouterUsageSplit),
		func(response *llmprotocol.Response) error {
			response.Model = "public-model"
			return nil
		},
	)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	usage := result.Response.Usage
	if usage.State != llmprotocol.UsageAvailable {
		t.Fatalf("usage state = %q, want available", usage.State)
	}
	assertTokenValue(t, "input total", usage.InputTotal, 20)
	assertTokenValue(t, "output total", usage.OutputTotal, 64)
	assertTokenValue(t, "total", usage.Total, 84)
	assertTokenUnknown(t, "output reasoning", usage.OutputReasoning)
	assertTokenUnknown(t, "output other", usage.OutputOther)
}

// A breakdown that does reconcile is still derived exactly as before.
func TestReconcilableUsageKeepsItsSplit(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":20,"completion_tokens":64,"total_tokens":84,` +
		`"prompt_tokens_details":{"cached_tokens":5,"cache_write_tokens":0},` +
		`"completion_tokens_details":{"reasoning_tokens":40}}}`)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	assertTokenValue(t, "output reasoning", response.Usage.OutputReasoning, 40)
	assertTokenValue(t, "output other", response.Usage.OutputOther, 24)
	assertTokenValue(t, "input cache read", response.Usage.InputCacheRead, 5)
	assertTokenValue(t, "input uncached", response.Usage.InputUncached, 15)
}

func assertTokenValue(t *testing.T, name string, count llmprotocol.TokenCount, want int64) {
	t.Helper()
	if count.Value == nil {
		t.Fatalf("%s is absent, want %d", name, want)
	}
	if *count.Value != want {
		t.Fatalf("%s = %d, want %d", name, *count.Value, want)
	}
}

func assertTokenUnknown(t *testing.T, name string, count llmprotocol.TokenCount) {
	t.Helper()
	if count.Value != nil {
		t.Fatalf("%s = %d, want it unknown because the provider's split cannot be true", name, *count.Value)
	}
}
