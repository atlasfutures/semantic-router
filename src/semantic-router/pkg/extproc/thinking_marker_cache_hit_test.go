package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Messages entry cached before minting was turned on holds unsigned
// thinking. Replayed after, it must carry the marker like a live turn, or the
// client resends another model's reasoning as text and the next Claude turn
// refuses it (#191). Claude's own signature still wins.
const preFlagCachedMessages = `{"id":"msg_cached","type":"message","role":"assistant","model":"moonshotai/kimi-k3",` +
	`"content":[{"type":"thinking","thinking":"weigh the two options","signature":""},` +
	`{"type":"text","text":"the second"}],"stop_reason":"end_turn","stop_sequence":null,` +
	`"usage":{"input_tokens":5,"output_tokens":7}}`

func cacheHitMintContext(t *testing.T, streaming bool) *RequestContext {
	t.Helper()
	return &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestID: "cache-mint", RequestModel: "moonshotai/kimi-k3",
		ExpectStreamingResponse: streaming, TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "moonshotai/kimi-k3", Stream: streaming},
	}
}

func mintRouter(mint bool) *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{RouterOptions: config.RouterOptions{ThinkingMarkerMint: mint}}}
}

func TestACacheHitMintsMarkersOnAPreFlagEntry(t *testing.T) {
	const marker = "vsr.thinking.v1.moonshotai."
	for _, streaming := range []bool{false, true} {
		name := map[bool]string{false: "buffered", true: "streamed"}[streaming]
		t.Run(name, func(t *testing.T) {
			response := mintRouter(true).createCacheHitResponse(cacheHitMintContext(t, streaming), []byte(preFlagCachedMessages), "", "", nil, 0)
			body := string(response.GetImmediateResponse().GetBody())
			if !strings.Contains(body, marker) {
				t.Fatalf("the cached thinking replayed unsigned:\n%s", body)
			}
			if streaming && !strings.Contains(body, `"type":"signature_delta"`) {
				t.Fatalf("the streamed replay carries no signature_delta:\n%s", body)
			}
		})
		t.Run(name+" with minting off", func(t *testing.T) {
			response := mintRouter(false).createCacheHitResponse(cacheHitMintContext(t, streaming), []byte(preFlagCachedMessages), "", "", nil, 0)
			if body := string(response.GetImmediateResponse().GetBody()); strings.Contains(body, "vsr.thinking.") {
				t.Fatalf("minting is off and the replay carries a marker:\n%s", body)
			}
		})
	}
}

func TestACacheHitKeepsClaudesOwnSignature(t *testing.T) {
	signed := strings.Replace(preFlagCachedMessages, `"signature":""`, `"signature":"EqQBCkYIBRgCKkBclaude"`, 1)
	for _, streaming := range []bool{false, true} {
		response := mintRouter(true).createCacheHitResponse(cacheHitMintContext(t, streaming), []byte(signed), "", "", nil, 0)
		body := string(response.GetImmediateResponse().GetBody())
		if strings.Contains(body, "vsr.thinking.") || !strings.Contains(body, "EqQBCkYIBRgCKkBclaude") {
			t.Fatalf("streaming=%v: Claude's signature was replaced:\n%s", streaming, body)
		}
	}
}
