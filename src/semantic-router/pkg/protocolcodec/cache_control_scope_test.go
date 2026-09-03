package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code negotiates prompt-caching-scope-2026-01-05 whenever it believes
// it holds a subscription session, and then puts a scope on the cache
// breakpoints of its system prompt. The cell refused every such turn with
// "request JSON contains a non-canonical field". A cache breakpoint is
// semantic request state the Router already models, so the scope belongs
// beside the type and the ttl rather than outside the contract.
const claudeCodeScopedCacheBody = `{"model":"m","max_tokens":64,` +
	`"system":[{"type":"text","text":"You are a Claude agent.",` +
	`"cache_control":{"type":"ephemeral","ttl":"1h","scope":"global"}}],` +
	`"messages":[{"role":"user","content":"hello"}]}`

func TestScopedCacheBreakpointSurvivesRouting(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(
		llmprotocol.AnthropicMessagesV1, []byte(claudeCodeScopedCacheBody),
	)
	if err != nil {
		t.Fatalf("a scoped cache breakpoint was refused: %v", err)
	}
	directive := firstCacheDirective(request)
	if directive == nil {
		t.Fatal("the decoded request holds no cache directive")
	}
	if directive.Type != "ephemeral" || directive.TTL != "1h" || directive.Scope != "global" {
		t.Fatalf("decoded cache directive = %+v", directive)
	}

	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	reEncoded := firstSystemCacheControl(t, encoded.Body)
	if reEncoded.Scope != "global" || reEncoded.TTL != "1h" {
		t.Fatalf("routing erased part of the cache breakpoint: %+v", reEncoded)
	}
}

func firstSystemCacheControl(t *testing.T, body []byte) anthropicCacheControlWire {
	t.Helper()
	var wire anthropicRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	var blocks []anthropicContentWire
	if err := json.Unmarshal(wire.System, &blocks); err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 || blocks[0].CacheControl == nil {
		t.Fatalf("re-encoded system = %s", wire.System)
	}
	return *blocks[0].CacheControl
}

func firstCacheDirective(request llmprotocol.Request) *llmprotocol.CacheDirective {
	for _, instruction := range request.Instructions {
		for _, content := range instruction.Content {
			if content.Cache != nil {
				return content.Cache
			}
		}
	}
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Cache != nil {
				return content.Cache
			}
		}
	}
	return nil
}
