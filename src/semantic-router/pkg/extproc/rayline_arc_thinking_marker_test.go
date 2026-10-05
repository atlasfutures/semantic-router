package extproc

import (
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
