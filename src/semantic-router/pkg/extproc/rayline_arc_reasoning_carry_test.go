package extproc

import (
	"encoding/json"
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
		wire := string(body["messages"])
		if strings.Contains(wire, "claude-signature") || strings.Contains(wire, "claude reasoning") {
			t.Fatalf("a Chat worker was sent Claude's signed thinking: %s", wire)
		}
		if !strings.Contains(wire, "Looking at it.") {
			t.Fatalf("the assistant's visible text was lost: %s", wire)
		}
	})
	t.Run("an open-weight model's unsigned thinking to a Chat worker", func(t *testing.T) {
		think := actions["think"].ActionID
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return think })
		body := dispatchPolicyClientRequest(t, router, "episode-carry-chat-unsigned", "/v1/messages",
			history(`{"type":"thinking","thinking":"mimo reasoning","signature":""}`))
		var messages []map[string]json.RawMessage
		if err := json.Unmarshal(body["messages"], &messages); err != nil || len(messages) < 2 {
			t.Fatalf("messages = %s", body["messages"])
		}
		assistant := messages[1]
		if string(assistant["reasoning_content"]) != `"mimo reasoning"` {
			t.Fatalf("unsigned thinking did not reach the Chat worker as reasoning_content: %s", body["messages"])
		}
		if strings.Contains(string(assistant["content"]), "mimo reasoning") {
			t.Fatalf("reasoning was sent as visible text: %s", body["messages"])
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
