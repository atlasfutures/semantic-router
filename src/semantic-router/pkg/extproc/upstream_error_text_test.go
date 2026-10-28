//go:build !windows && cgo

package extproc

import (
	"fmt"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// prod 2026-10-28: OpenRouter's 400 names its cause only in its text and its
// metadata, and the cell logged a bare 400.
const openRouterEncryptedReasoning400 = `{"error":{"code":400,` +
	`"message":"This request contains encrypted reasoning or compaction content that can only be processed by the provider that created it (anthropic) or by azure/global, claude-on-aws, azure/us. See https://openrouter.ai/docs/reasoning",` +
	`"metadata":{"pinned_provider_slug":"amazon-bedrock","accepting_provider_slugs":["anthropic","azure/global","claude-on-aws","azure/us"],` +
	`"raw":"{\"messages\":[{\"role\":\"user\",\"content\":\"my secret prompt\"}]}","provider_name":"Amazon Bedrock"}}}`

// A provider 4xx's message and OpenRouter's provider slugs reach turn_failed
// and llm_usage; nothing else of the body does.
func TestUpstream4xxTextIsLoggedOnTurnFailedAndUsage(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		UpstreamStatusCode: 400, RequestID: "request_400", RequestModel: "sonnet",
	}
	(&OpenAIRouter{}).handleUpstreamTransportError([]byte(openRouterEncryptedReasoning400), ctx)

	wantSlugs := `[anthropic azure/global claude-on-aws azure/us]`
	for _, event := range []string{"turn_failed", "llm_usage"} {
		fields := findLogEvent(t, logs, event)
		message, _ := fields["upstream_error_message"].(string)
		if !strings.Contains(message, "can only be processed by the provider that created it (anthropic)") {
			t.Fatalf("%s upstream_error_message = %q", event, message)
		}
		if strings.Contains(message, "openrouter.ai") {
			t.Fatalf("%s kept the link: %q", event, message)
		}
		if fields["upstream_error_pinned_provider_slug"] != "amazon-bedrock" {
			t.Fatalf("%s pinned slug = %v", event, fields["upstream_error_pinned_provider_slug"])
		}
		if got := fmt.Sprint(fields["upstream_error_accepting_provider_slugs"]); got != wantSlugs {
			t.Fatalf("%s accepting slugs = %s, want %s", event, got, wantSlugs)
		}
	}
	for _, entry := range logs.All() {
		if line := fmt.Sprint(entry.ContextMap()); strings.Contains(line, "my secret prompt") {
			t.Fatalf("request content reached the log: %s", line)
		}
	}
}

func TestUpstreamErrorTextIsBoundedAndKeepsOnlySlugs(t *testing.T) {
	long := strings.Repeat("é", 400)
	body := []byte(`{"error":{"message":"` + long + `","metadata":{"pinned_provider_slug":"bad slug with spaces",` +
		`"accepting_provider_slugs":["ok-slug","{\"x\":1}"]}}}`)
	text := readUpstreamErrorText(400, body, nil)
	if len(text.message) > upstreamErrorTextMax || !strings.HasPrefix(long, text.message) || text.message == "" {
		t.Fatalf("message is %d bytes, or cut mid-rune", len(text.message))
	}
	if text.pinnedSlug != "" || fmt.Sprint(text.acceptingSlugs) != "[ok-slug]" {
		t.Fatalf("slugs = %q %v", text.pinnedSlug, text.acceptingSlugs)
	}
	// Account refusals name the cell's provider account; 5xx and 2xx are not 4xx.
	for _, status := range []int{401, 402, 500, 200} {
		if got := readUpstreamErrorText(status, body, nil); got.message != "" {
			t.Fatalf("status %d logged %q", status, got.message)
		}
	}
}

// A failure without a 4xx states no text: the fields are null.
func TestUsageLineStatesNoUpstreamErrorTextByDefault(t *testing.T) {
	logs := captureLogs(t)
	usageRecordRouter().reportNonStreamingUsage(&RequestContext{RequestID: "r", RequestModel: "priced-model"}, 0, responseUsageMetrics{invalid: true})
	fields := findLogEvent(t, logs, "llm_usage")
	for _, name := range []string{"upstream_error_message", "upstream_error_pinned_provider_slug", "upstream_error_accepting_provider_slugs"} {
		if value, present := fields[name]; !present || value != nil {
			t.Fatalf("%s = %v (present %v), want null", name, value, present)
		}
	}
}
