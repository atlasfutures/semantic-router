package protocolcodec

import (
	"bytes"
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

// A usage object may omit either breakdown entirely. The totals it does state
// must survive and the omitted sub-counts must read as unknown, never as a
// derived number the provider did not give.
func TestUsageWithoutBreakdownKeepsItsTotals(t *testing.T) {
	bodies := map[string]string{
		"no completion details": `"usage":{"prompt_tokens":20,"completion_tokens":64,"total_tokens":84,` +
			`"prompt_tokens_details":{"cached_tokens":5,"cache_write_tokens":0}}`,
		"no prompt details": `"usage":{"prompt_tokens":20,"completion_tokens":64,"total_tokens":84,` +
			`"completion_tokens_details":{"reasoning_tokens":40}}`,
		"no details at all": `"usage":{"prompt_tokens":20,"completion_tokens":64,"total_tokens":84}`,
	}
	engine := NewBuiltinEngine()
	for name, usage := range bodies {
		for targetName, target := range map[string]llmprotocol.WireFormat{
			"chat":      llmprotocol.OpenAIChatV1,
			"anthropic": llmprotocol.AnthropicMessagesV1,
		} {
			t.Run(name+"/"+targetName, func(t *testing.T) {
				body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
					`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
					usage + `}`)
				result, err := engine.TranslateResponse(
					llmprotocol.OpenAIChatV1, target, body,
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
				assertTokenValue(t, "input total", result.Response.Usage.InputTotal, 20)
				assertTokenValue(t, "output total", result.Response.Usage.OutputTotal, 64)
				assertTokenValue(t, "total", result.Response.Usage.Total, 84)
			})
		}
	}
}

// The stream path derives usage with the same function, so the terminal chunk
// of a stream whose breakdown cannot be true must not fail either -- and the
// totals it does state must reach the client, because that is what the turn
// is billed from.
func TestUnreconcilableUsageInAStreamStillCompletes(t *testing.T) {
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{PublicModel: "public-model", ProviderModel: "provider-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	head := `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[{"index":0,`
	body := "data: " + head + `"delta":{"content":"hi","role":"assistant"},"finish_reason":null}]}` + "\n\n" +
		"data: " + head + `"delta":{"content":"","role":"assistant"},"finish_reason":"length"}],` +
		`"usage":{"prompt_tokens":20,"completion_tokens":64,"total_tokens":84,` +
		`"completion_tokens_details":{"reasoning_tokens":79}}}` + "\n\n" +
		"data: [DONE]\n\n"
	var encoded bytes.Buffer
	frames, events, _, pushErr := stream.Push([]byte(body))
	if pushErr != nil {
		t.Fatalf("push: %v", pushErr)
	}
	writeFrames(&encoded, frames)
	finalFrames, finalEvents, _, finalErr := stream.Finalize(nil)
	if finalErr != nil {
		t.Fatalf("finalize: %v", finalErr)
	}
	writeFrames(&encoded, finalFrames)
	assertUnreconciledStreamUsage(t, append(events, finalEvents...))

	// On the wire the dropped split reads as a zero, because Messages has no
	// way to say "unknown". output_tokens stays 64, which is the number the
	// provider billed; a clamp would have written 64-79 as some other count.
	usage := messageDeltaUsage(t, "unreconcilable stream", encoded.Bytes())
	assertArmUsage(t, usage, armUsage(20, 64, 0, 0, 0))
}

func assertUnreconciledStreamUsage(t *testing.T, events []llmprotocol.Event) {
	t.Helper()
	for index := len(events) - 1; index >= 0; index-- {
		if events[index].Usage == nil {
			continue
		}
		usage := *events[index].Usage
		assertTokenValue(t, "input total", usage.InputTotal, 20)
		assertTokenValue(t, "output total", usage.OutputTotal, 64)
		assertTokenUnknown(t, "output reasoning", usage.OutputReasoning)
		assertTokenUnknown(t, "output other", usage.OutputOther)
		return
	}
	t.Fatalf("no event carried usage: %+v", events)
}
