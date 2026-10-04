package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Through the router's own ingress and dispatch: a Responses client's
// top-level cache_control, routed to a Messages worker, dispatches a cache
// breakpoint on the last block of the conversation. Without it Anthropic
// caches nothing, and a growing conversation is billed uncached every turn.
// Control: the same body without the member dispatches no breakpoint.
func TestIngressAutoCacheDispatchesAMessagesBreakpoint(t *testing.T) {
	const withDirective = `{"model":"m","instructions":"You are a coding agent.",` +
		`"cache_control":{"type":"ephemeral"},` +
		`"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"List the files."}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"There are two files."}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"Open main.go."}]}]}`
	const withoutDirective = `{"model":"m","instructions":"You are a coding agent.",` +
		`"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"List the files."}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"There are two files."}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"Open main.go."}]}]}`

	dispatch := func(body string) []byte {
		router := &OpenAIRouter{}
		ctx := &RequestContext{
			Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1,
			TargetFormat: llmprotocol.AnthropicMessagesV1,
			RequestID:    "auto-cache", TraceContext: context.Background(),
		}
		request, immediate := router.prepareProtocolRequest([]byte(body), ctx)
		if immediate != nil || request == nil {
			t.Fatalf("the request was refused at ingress: %+v", ctx.ImmediateProtocolError)
		}
		dispatched, err := router.encodeDispatchRequest(ctx)
		if err != nil {
			t.Fatal(err)
		}
		return dispatched
	}

	var wire struct {
		CacheControl json.RawMessage `json:"cache_control"`
		Messages     []struct {
			Content []map[string]json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	body := dispatch(withDirective)
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("dispatched %s: %v", body, err)
	}
	if len(wire.CacheControl) != 0 || len(wire.Messages) != 3 {
		t.Fatalf("dispatched %s", body)
	}
	breakpoints := 0
	for _, message := range wire.Messages {
		for _, block := range message.Content {
			if len(block["cache_control"]) > 0 {
				breakpoints++
			}
		}
	}
	last := wire.Messages[2].Content
	if breakpoints != 1 || string(last[len(last)-1]["cache_control"]) != `{"type":"ephemeral"}` {
		t.Fatalf("dispatched %s, want one breakpoint on the last block", body)
	}

	if control := dispatch(withoutDirective); json.Valid(control) && containsCacheControl(control) {
		t.Fatalf("a request without the directive dispatched %s", control)
	}
}

func containsCacheControl(body []byte) bool {
	var value any
	if json.Unmarshal(body, &value) != nil {
		return false
	}
	var walk func(any) bool
	walk = func(node any) bool {
		switch typed := node.(type) {
		case map[string]any:
			if _, present := typed["cache_control"]; present {
				return true
			}
			for _, member := range typed {
				if walk(member) {
					return true
				}
			}
		case []any:
			for _, member := range typed {
				if walk(member) {
					return true
				}
			}
		}
		return false
	}
	return walk(value)
}
