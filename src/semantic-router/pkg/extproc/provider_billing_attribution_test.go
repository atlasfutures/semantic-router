package extproc

import (
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// writeOpenWeightOverMessagesPolicyConfig is the e2e policy config with the
// "off" worker an open-weight model reached through OpenRouter's Messages
// API, as the prod cell dispatches glm-5.3-flash.
func writeOpenWeightOverMessagesPolicyConfig(t *testing.T, policyURL string) string {
	t.Helper()
	path := writeConsistentPolicyConfig(t, policyURL, "strict")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	swaps := [][2]string{
		{"    - name: \"off\"\n      provider_model_id: vendor/off\n      api_format: openai\n",
			"    - name: \"off\"\n      provider_model_id: z-ai/glm-5.3-flash\n      api_format: anthropic\n"},
		{"    - name: \"off\"\n      modality: text\n", "    - name: \"off\"\n      modality: text\n      publisher: z-ai\n"},
	}
	rendered := string(raw)
	for _, swap := range swaps {
		if !strings.Contains(rendered, swap[0]) {
			t.Fatalf("the policy e2e config no longer carries %q", swap[0])
		}
		rendered = strings.Replace(rendered, swap[0], swap[1], 1)
	}
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// claudeCodeMessagesTurn is a Claude Code Messages request whose system
// prompt opens with the billing attribution line for the given cch.
func claudeCodeMessagesTurn(cch string) string {
	return `{"model":"auto","max_tokens":32000,"system":[` +
		`{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli; ` +
		`cch=` + cch + `; cc_prompt_id=e69f0f9f-03e0-488b-b854-1415abcd0000;"},` +
		`{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"fix the failing test"},` +
		`{"role":"assistant","content":"Looking at it."},` +
		`{"role":"user","content":"go on"}]}`
}

// An open-weight worker on Messages is sent the line without its per-turn
// fields, so two turns that differ only in cch share one system prompt and the
// provider's prefix cache matches past it. The rewrite is logged.
func TestBillingAttributionIsStableForAnOpenWeightWorkerOverMessages(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeOpenWeightOverMessagesPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	logs := captureLogs(t)
	first := dispatchPolicyClientRequest(t, router, "episode-billing-1", "/v1/messages", claudeCodeMessagesTurn("8dc28"))
	assertJSONField(t, first, "model", `"z-ai/glm-5.3-flash"`)
	system := string(first["system"])
	if !strings.Contains(system, `"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli;"`) ||
		strings.Contains(system, "cch=") || strings.Contains(system, "cc_prompt_id=") ||
		!strings.Contains(system, "You are Claude Code.") {
		t.Fatalf("system sent to the open-weight worker = %s, want the line without its per-turn fields", system)
	}
	if event := findLogEvent(t, logs, "billing_attribution_stabilized"); event["blocks"] != int64(1) {
		t.Fatalf("billing_attribution_stabilized = %v, want one block", event)
	}
	next := dispatchPolicyClientRequest(t, router, "episode-billing-2", "/v1/messages", claudeCodeMessagesTurn("0cd8e"))
	if string(next["system"]) != system {
		t.Fatalf("turns differing only in cch were sent different system prompts:\n%s\n%s", system, next["system"])
	}
}

// A Claude worker gets the line as the client sent it: Anthropic reads it as
// metadata.
func TestBillingAttributionReachesAClaudeWorkerOverMessages(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	fake := newRelaxedPolicyFake(t)
	claude := relaxedPolicyActions()["claude-off"].ActionID
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })
	router, err := NewOpenAIRouter(writeOpenWeightOverMessagesPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	body := dispatchPolicyClientRequest(t, router, "episode-billing-claude", "/v1/messages", claudeCodeMessagesTurn("8dc28"))
	assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
	if !strings.Contains(string(body["system"]), "cch=8dc28; cc_prompt_id=e69f0f9f-03e0-488b-b854-1415abcd0000;") {
		t.Fatalf("a Claude worker did not get the line as sent: %s", body["system"])
	}
}
