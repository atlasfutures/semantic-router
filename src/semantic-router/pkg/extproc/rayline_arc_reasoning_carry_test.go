package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// End to end: a conversation whose history holds another model's reasoning
// is served on whichever worker the policy picks next, instead of being
// refused (a Chat worker and Claude's signed thinking) or sent with thinking
// the provider cannot verify (a Messages worker and kimi's unsigned thinking).
func TestReasoningFromAnotherModelIsCarriedToTheNextWorker(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), "strict"))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	history := func(thinking string) string {
		return `{"model":"auto","max_tokens":4096,"messages":[` +
			`{"role":"user","content":"fix the failing test"},` +
			`{"role":"assistant","content":[` + thinking + `,{"type":"text","text":"Looking at it."}]},` +
			`{"role":"user","content":"go on"}]}`
	}

	t.Run("Claude's signed thinking to a Chat worker", func(t *testing.T) {
		think := actions["think"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return think })
		body := dispatchPolicyClientRequest(t, router, "episode-carry-chat", "/v1/messages",
			history(`{"type":"thinking","thinking":"claude reasoning","signature":"claude-signature"}`))
		assertJSONField(t, body, "model", `"vendor/think"`)
		if wire := string(body["messages"]); strings.Contains(wire, "claude-signature") {
			t.Fatalf("a Chat worker was sent the Anthropic signature: %s", wire)
		}
	})
	t.Run("kimi's unsigned thinking to a Messages worker", func(t *testing.T) {
		claude := actions["claude"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
		body := dispatchPolicyClientRequest(t, router, "episode-carry-messages", "/v1/messages",
			history(`{"type":"thinking","thinking":"kimi reasoning","signature":""}`))
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		if wire := string(body["messages"]); strings.Contains(wire, "kimi reasoning") || strings.Contains(wire, `"thinking"`) {
			t.Fatalf("a Messages worker was sent unsigned thinking: %s", wire)
		}
		if !strings.Contains(string(body["messages"]), "Looking at it.") {
			t.Fatalf("the assistant's visible text was lost: %s", body["messages"])
		}
	})
}
