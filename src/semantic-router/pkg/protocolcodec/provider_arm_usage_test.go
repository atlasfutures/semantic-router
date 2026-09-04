package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// What a Rayline turn is billed from.
//
// These five numbers are read off the Anthropic body the cell emits, not off
// the neutral Usage struct behind it, because the body is what the proxy
// meters. Every arm in the basket reaches the client through this shape, and
// each provider states its usage a little differently, so each arm is pinned
// against its own recorded response rather than against a synthetic one.
type armUsageWire struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	OutputTokensDetails      struct {
		ThinkingTokens int64 `json:"thinking_tokens"`
	} `json:"output_tokens_details"`
}

// armUsageExpectation is one arm in one mode.
type armUsageExpectation struct {
	fixture string
	want    armUsageWire
}

type armUsageCase struct {
	arm         string
	nonStreamed armUsageExpectation
	streamed    armUsageExpectation
}

func armUsage(input, output, cacheRead, cacheCreation, thinking int64) armUsageWire {
	wire := armUsageWire{
		InputTokens:              input,
		OutputTokens:             output,
		CacheReadInputTokens:     cacheRead,
		CacheCreationInputTokens: cacheCreation,
	}
	wire.OutputTokensDetails.ThinkingTokens = thinking
	return wire
}

// The basket, arm by arm. Numbers come from the recorded provider bodies in
// testdata/provider/openrouter; that directory's README names each capture.
func armUsageCases() []armUsageCase {
	return []armUsageCase{
		{
			// The plain shape: no cache, no reasoning on the answer turn.
			arm: "deepseek/deepseek-v4-pro",
			nonStreamed: armUsageExpectation{
				fixture: "chat-response.json",
				want:    armUsage(11, 8, 0, 0, 0),
			},
			streamed: armUsageExpectation{
				fixture: "chat-stream.sse",
				want:    armUsage(11, 68, 0, 0, 58),
			},
		},
		{
			// The cached shape. The provider states prompt_tokens 258 with
			// cached_tokens 192, and Anthropic counts the cached part
			// separately, so input_tokens must be 66 and not 258. Billing the
			// full 258 beside a cache_read of 192 would charge the prompt
			// twice.
			arm: "xiaomi/mimo-v2.5-pro",
			nonStreamed: armUsageExpectation{
				fixture: "chat-response-reasoning.json",
				want:    armUsage(66, 22, 192, 0, 13),
			},
			streamed: armUsageExpectation{
				fixture: "chat-stream-reasoning.sse",
				want:    armUsage(66, 22, 192, 0, 13),
			},
		},
		{
			// The unreconcilable shape, non-streamed: reasoning_tokens 79
			// against completion_tokens 64. The totals survive and the split
			// is dropped, so thinking_tokens reads 0 -- an absent breakdown,
			// not a claim that no reasoning happened. output_tokens stays 64,
			// which is what the provider billed. Streamed, the same arm
			// reconciles and keeps its split.
			arm: "qwen/qwen3.6-35b-a3b",
			nonStreamed: armUsageExpectation{
				fixture: "chat-response-usage-split.json",
				want:    armUsage(20, 64, 0, 0, 0),
			},
			streamed: armUsageExpectation{
				fixture: "chat-stream-length-reasoning.sse",
				want:    armUsage(17, 128, 0, 0, 102),
			},
		},
		{
			// Cached on a 13-token prompt: 11 of the 13 came from the
			// provider's cache, so only 2 are fresh input.
			arm: "z-ai/glm-5.2",
			nonStreamed: armUsageExpectation{
				fixture: "chat-response-glm.json",
				want:    armUsage(2, 9, 11, 0, 0),
			},
			streamed: armUsageExpectation{
				fixture: "chat-stream-glm.sse",
				want:    armUsage(2, 3, 11, 0, 0),
			},
		},
		{
			// The flash arm states every sub-count as zero, and spells its
			// small numbers in exponent form. Both captures carry that
			// spelling, so decoding them at all is part of what this pins.
			arm: "deepseek/deepseek-v4-flash",
			nonStreamed: armUsageExpectation{
				fixture: "chat-response-flash.json",
				want:    armUsage(11, 2, 0, 0, 0),
			},
			streamed: armUsageExpectation{
				fixture: "chat-stream-flash.sse",
				want:    armUsage(11, 3, 0, 0, 0),
			},
		},
	}
}

func TestEachArmReportsItsOwnUsageShape(t *testing.T) {
	for _, testCase := range armUsageCases() {
		t.Run(testCase.arm+"/non-streamed", func(t *testing.T) {
			expectation := testCase.nonStreamed
			assertArmUsage(t, anthropicUsageFromResponse(t, expectation.fixture), expectation.want)
		})
		t.Run(testCase.arm+"/streamed", func(t *testing.T) {
			expectation := testCase.streamed
			assertArmUsage(t, anthropicUsageFromStream(t, expectation.fixture), expectation.want)
		})
	}
}

// anthropicUsageFromResponse translates one recorded provider body to Messages
// and reads the usage object the client would receive.
func anthropicUsageFromResponse(t *testing.T, fixture string) armUsageWire {
	t.Helper()
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, loadUsageFixture(t, fixture),
		func(response *llmprotocol.Response) error {
			response.Model = "public-model"
			return nil
		},
	)
	if err != nil {
		t.Fatalf("translate %s: %v", fixture, err)
	}
	var body struct {
		Usage armUsageWire `json:"usage"`
	}
	if err := json.Unmarshal(result.Body, &body); err != nil {
		t.Fatalf("decode translated body of %s: %v\n%s", fixture, err, result.Body)
	}
	return body.Usage
}

// anthropicUsageFromStream translates one recorded provider stream to Messages
// and reads the usage off message_delta, which is where the whole count rides:
// message_start carries a zero placeholder on this path.
func anthropicUsageFromStream(t *testing.T, fixture string) armUsageWire {
	t.Helper()
	stream := mustNewMatrixStream(t, NewBuiltinEngine(), llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1)
	var encoded bytes.Buffer
	frames, _, _, err := stream.Push(loadUsageFixture(t, fixture))
	if err != nil {
		t.Fatalf("push %s: %v", fixture, err)
	}
	writeFrames(&encoded, frames)
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize %s: %v", fixture, err)
	}
	writeFrames(&encoded, finalFrames)
	return messageDeltaUsage(t, fixture, encoded.Bytes())
}

func messageDeltaUsage(t *testing.T, fixture string, wire []byte) armUsageWire {
	t.Helper()
	var found *armUsageWire
	for _, line := range strings.Split(string(wire), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event struct {
			Type  string        `json:"type"`
			Usage *armUsageWire `json:"usage"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			continue
		}
		if event.Type == "message_delta" && event.Usage != nil {
			found = event.Usage
		}
	}
	if found == nil {
		t.Fatalf("%s produced no message_delta usage:\n%s", fixture, wire)
	}
	return *found
}

func assertArmUsage(t *testing.T, got, want armUsageWire) {
	t.Helper()
	for _, field := range []struct {
		name      string
		got, want int64
	}{
		{"input_tokens", got.InputTokens, want.InputTokens},
		{"output_tokens", got.OutputTokens, want.OutputTokens},
		{"cache_read_input_tokens", got.CacheReadInputTokens, want.CacheReadInputTokens},
		{"cache_creation_input_tokens", got.CacheCreationInputTokens, want.CacheCreationInputTokens},
		{"output_tokens_details.thinking_tokens", got.OutputTokensDetails.ThinkingTokens, want.OutputTokensDetails.ThinkingTokens},
	} {
		if field.got != field.want {
			t.Errorf("%s = %d, want %d", field.name, field.got, field.want)
		}
	}
}

// A warm prefix is where the two protocols disagree about what a prompt is.
// OpenRouter states prompt_tokens 1702 with 1024 of it cached; Messages counts
// the cached part separately, so the client must see 678 fresh input beside a
// cache_read of 1024. Billing 1702 there charges the cached prefix twice, at
// the uncached rate.
func TestWarmPrefixLeavesOnlyFreshInputInInputTokens(t *testing.T) {
	want := armUsage(678, 2, 1024, 0, 0)
	t.Run("non-streamed", func(t *testing.T) {
		assertArmUsage(t, anthropicUsageFromResponse(t, "chat-response-warm-prefix.json"), want)
	})
	t.Run("streamed", func(t *testing.T) {
		assertArmUsage(t, anthropicUsageFromStream(t, "chat-stream-warm-prefix.sse"), want)
	})
}

// A turn can be all reasoning and no answer: glm-5.2 spent its whole 64-token
// allowance thinking and returned empty content. reasoning_tokens equals
// completion_tokens, which reconciles exactly, so the split is kept and the
// turn is served. It is the boundary next to the qwen case, where reasoning
// exceeds completion and the split has to be dropped.
func TestOutputThatIsAllReasoningKeepsItsSplit(t *testing.T) {
	assertArmUsage(
		t,
		anthropicUsageFromResponse(t, "chat-response-all-reasoning.json"),
		armUsage(19, 64, 0, 0, 64),
	)
}
