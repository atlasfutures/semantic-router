package extproc

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Through the router's own ingress and dispatch: a Responses request with
// service_tier "auto" is admitted (semantic-router#163), a translated
// dispatch does not carry the tier, and a same-format dispatch replays only
// the client's "auto". Control: an explicit tier is refused at ingress.
func TestIngressAdmitsServiceTierAutoAndDispatchesNoOtherTier(t *testing.T) {
	router := &OpenAIRouter{}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		ctx := &RequestContext{
			Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: target,
			RequestID: "tier-auto", TraceContext: context.Background(),
		}
		request, immediate := router.prepareProtocolRequest([]byte(`{"model":"m","input":"hi","service_tier":"auto"}`), ctx)
		if immediate != nil || request == nil {
			t.Fatalf("-> %s: service_tier auto was refused at ingress", target)
		}
		body, err := router.encodeDispatchRequest(ctx)
		if err != nil {
			t.Fatalf("-> %s: %v", target, err)
		}
		carried := bytes.Contains(body, []byte("service_tier"))
		if (target != llmprotocol.OpenAIResponsesV1 && carried) ||
			(carried && !bytes.Contains(body, []byte(`"service_tier":"auto"`))) {
			t.Fatalf("-> %s: dispatched %s", target, body)
		}
	}
	ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1, RequestID: "tier-flex", TraceContext: context.Background()}
	if _, immediate := router.prepareProtocolRequest([]byte(`{"model":"m","input":"hi","service_tier":"flex"}`), ctx); immediate == nil ||
		ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != "unsupported_service_tier" {
		t.Fatalf("an explicit tier was not refused: %+v", ctx.ImmediateProtocolError)
	}
}
