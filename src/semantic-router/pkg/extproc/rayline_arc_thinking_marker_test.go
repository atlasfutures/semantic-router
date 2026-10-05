package extproc

import (
	"encoding/base64"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// markerThinkingHistory is a Messages client's history whose earlier
// assistant turn holds thinking signed with the given signature, as pi and
// OpenClaw resend thinking the cell signed with a Router marker (#191).
func markerThinkingHistory(signature string) string {
	return `{"model":"auto","max_tokens":4096,"messages":[` +
		`{"role":"user","content":"fix the failing test"},` +
		`{"role":"assistant","content":[{"type":"thinking","thinking":"kimi reasoning","signature":"` + signature + `"},` +
		`{"type":"text","text":"Looking at it."}]},` +
		`{"role":"user","content":"go on"}]}`
}

const thinkingMarker = "vsr.thinking.v1.moonshotai.AbCdEfGhIjKlMnOpQrStUvWxYz012-_9"

// End to end: thinking a Messages client resends under a Router marker is
// another model's reasoning. A Claude worker, on Messages or over Chat, is
// never sent it, and never sent the marker: the block is dropped and counted
// as foreign. A forged or malformed value in the Router's namespace is
// handled the same way. An open-weight worker over Chat keeps the text as
// reasoning_content.
func TestRouterMarkerSignedThinkingNeverReachesAClaudeWorker(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)

	signatures := []struct{ name, signature, family string }{
		{"marker", thinkingMarker, "moonshotai"},
		{"forged marker", "vsr.thinking.v1.anthropic." + strings.Repeat("A", 32), "anthropic"},
		{"malformed", "vsr.thinking.v1.MoonShot.x", "malformed"},
	}
	workers := []struct{ action, model string }{
		{"claude", "anthropic/claude-opus-5"},     // Messages
		{"claude-off", "anthropic/claude-opus-5"}, // Chat via OpenRouter
	}
	for _, signed := range signatures {
		for _, worker := range workers {
			t.Run(signed.name+" to "+worker.action, func(t *testing.T) {
				action := actions[worker.action].ActionID
				fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
				logs := captureLogs(t)
				body := dispatchPolicyClientRequest(t, router, "episode-marker-"+worker.action+"-"+signed.name,
					"/v1/messages", markerThinkingHistory(signed.signature))
				assertJSONField(t, body, "model", `"`+worker.model+`"`)
				wire := string(body["messages"])
				if strings.Contains(wire, "vsr.") || strings.Contains(wire, "kimi reasoning") ||
					strings.Contains(wire, "reasoning_content") || strings.Contains(wire, `"thinking"`) {
					t.Fatalf("a Claude worker was sent marker-signed thinking: %s", wire)
				}
				if !strings.Contains(wire, "Looking at it.") {
					t.Fatalf("the assistant's visible text was lost: %s", wire)
				}
				dropped := findLogEvent(t, logs, "reasoning_dropped")
				if fmt.Sprint(dropped["foreign_dropped"]) != "1" {
					t.Fatalf("reasoning_dropped = %v, want 1 foreign drop", dropped)
				}
				stripped := findLogEvent(t, logs, "thinking_marker_stripped")
				if fmt.Sprint(stripped["family"]) != signed.family {
					t.Fatalf("thinking_marker_stripped = %v, want family %q", stripped, signed.family)
				}
			})
		}
	}

	t.Run("marker to an open-weight worker over Chat", func(t *testing.T) {
		off := actions["off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return off })
		body := dispatchPolicyClientRequest(t, router, "episode-marker-off", "/v1/messages", markerThinkingHistory(thinkingMarker))
		assertJSONField(t, body, "model", `"vendor/off"`)
		wire := string(body["messages"])
		if strings.Contains(wire, "vsr.") || !strings.Contains(wire, `"reasoning_content":"kimi reasoning"`) {
			t.Fatalf("an open-weight worker was not sent the reasoning unsigned: %s", wire)
		}
	})
}

// End to end, Chat source: a Chat client that resends marker-signed thinking
// as an anthropic-claude-v1 reasoning_details item has not shown Claude's
// provenance -- the marker is the Router's, whatever format the item names.
// A Claude worker over Chat is sent none of that reasoning, and the drop is
// counted as foreign. An open-weight Chat worker keeps reasoning_content.
func TestRouterMarkerSignedReasoningDetailsNeverReachAClaudeWorkerOverChat(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	history := piChatHistory(`"reasoning_content":"kimi reasoning",` +
		`"reasoning_details":[{"type":"reasoning.text","text":"kimi reasoning","signature":"` + thinkingMarker +
		`","format":"anthropic-claude-v1","index":0}]`)

	t.Run("Claude over Chat", func(t *testing.T) {
		claude := actions["claude-off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
		logs := captureLogs(t)
		body := dispatchPolicyClientRequest(t, router, "episode-marker-details-claude", "/v1/chat/completions", history)
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		wire := string(body["messages"])
		if strings.Contains(wire, "vsr.") || strings.Contains(wire, "kimi reasoning") ||
			strings.Contains(wire, "reasoning_content") || strings.Contains(wire, "reasoning_details") {
			t.Fatalf("a Claude worker was sent marker-signed reasoning: %s", wire)
		}
		if !strings.Contains(wire, `"tool_call_id":"call_1"`) {
			t.Fatalf("the tool result was lost: %s", wire)
		}
		dropped := findLogEvent(t, logs, "reasoning_dropped")
		if fmt.Sprint(dropped["foreign_dropped"]) != "1" {
			t.Fatalf("reasoning_dropped = %v, want 1 foreign drop", dropped)
		}
		stripped := findLogEvent(t, logs, "thinking_marker_stripped")
		if fmt.Sprint(stripped["family"]) != "moonshotai" || fmt.Sprint(stripped["source"]) != "openai.chat.v1" {
			t.Fatalf("thinking_marker_stripped = %v", stripped)
		}
	})
	t.Run("open-weight worker over Chat", func(t *testing.T) {
		off := actions["off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return off })
		body := dispatchPolicyClientRequest(t, router, "episode-marker-details-off", "/v1/chat/completions", history)
		wire := string(body["messages"])
		if strings.Contains(wire, "vsr.") || !strings.Contains(wire, `"reasoning_content":"kimi reasoning"`) {
			t.Fatalf("an open-weight worker was not sent the reasoning unsigned: %s", wire)
		}
	})
}

// mintedBlobOf is the encrypted_content the Router mints for a Responses
// client from a reasoning_details array (vsr.reasoning_details.v1.).
func mintedBlobOf(details string) string {
	return "vsr.reasoning_details.v1." + base64.RawURLEncoding.EncodeToString([]byte(details))
}

// End to end, Responses source: a Router-minted reasoning item whose blob
// holds only marker-signed items is still the Router's item. Stripping the
// items empties the blob, and the item's own reasoning text goes on as
// unsigned reasoning: carried to an open-weight worker, dropped as foreign for
// a Claude worker. Neither is sent the blob.
func TestMintedBlobOfOnlyMarkerItemsDecodesAsUnsignedReasoning(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	blob := mintedBlobOf(`[{"type":"reasoning.text","text":"kimi reasoning","signature":"` + thinkingMarker +
		`","format":"anthropic-claude-v1","index":0}]`)
	history := `{"model":"auto","input":[` +
		`{"role":"user","content":[{"type":"input_text","text":"weather in Paris?"}]},` +
		`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"kimi reasoning"}],"encrypted_content":"` + blob + `"},` +
		`{"role":"assistant","content":[{"type":"output_text","text":"It is sunny."}]},` +
		`{"role":"user","content":[{"type":"input_text","text":"and tomorrow?"}]}]}`

	t.Run("open-weight worker over Chat", func(t *testing.T) {
		off := actions["off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return off })
		body := dispatchPolicyClientRequest(t, router, "episode-minted-marker-off", "/v1/responses", history)
		assertJSONField(t, body, "model", `"vendor/off"`)
		wire := string(body["messages"])
		if strings.Contains(wire, "vsr.") || !strings.Contains(wire, `"reasoning_content":"kimi reasoning"`) {
			t.Fatalf("an open-weight worker was not sent the reasoning unsigned: %s", wire)
		}
	})
	t.Run("Claude over Chat", func(t *testing.T) {
		claude := actions["claude-off"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
		logs := captureLogs(t)
		body := dispatchPolicyClientRequest(t, router, "episode-minted-marker-claude", "/v1/responses", history)
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		wire := string(body["messages"])
		if strings.Contains(wire, "vsr.") || strings.Contains(wire, "kimi reasoning") || strings.Contains(wire, "reasoning_") {
			t.Fatalf("a Claude worker was sent marker-signed reasoning: %s", wire)
		}
		dropped := findLogEvent(t, logs, "reasoning_dropped")
		if fmt.Sprint(dropped["foreign_dropped"]) != "1" {
			t.Fatalf("reasoning_dropped = %v, want 1 foreign drop", dropped)
		}
	})
}
