//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The output clamp end to end, in process: the router is built from a config
// file whose worker card declares a context window and an output cap, and a
// client's Chat request that states no limit runs through the real ext_proc
// request phases. What is asserted is the limit on the provider-bound body:
// the card's cap when the prompt leaves room for it, and less than the cap
// when the prompt does not.
func TestDispatchOutputBoundReachesTheProviderBody(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	const window, cap = 6000, 4000
	action := policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, "")
	actions := map[string]config.RaylineARCPolicyBinding{
		"off":           action,
		"think-up":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"claude":        policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off-up": policyAction("claude-off", "up", "claude-opus-5", policyTestEffort("none"), nil, policyTestUp),
	}
	catalog := make([]string, 0, len(actions))
	for _, declared := range actions {
		catalog = append(catalog, declared.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	path := writePolicyDispatchConfig(t, fake.URL(), actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// The off worker's card declares the window and the cap.
	card := "    - name: \"off\"\n      modality: text\n"
	if strings.Count(string(raw), card) != 1 {
		t.Fatalf("the config template changed; update the off card rewrite:\n%s", card)
	}
	declared := card + "      context_window_size: " + strconv.Itoa(window) + "\n" +
		"      max_output_tokens: " + strconv.Itoa(cap) + "\n"
	if err := os.WriteFile(path, []byte(strings.Replace(string(raw), card, declared, 1)), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action.ActionID })

	limit := func(t *testing.T, episode, prompt string) int {
		t.Helper()
		client, err := json.Marshal(map[string]any{
			"model":    "auto",
			"messages": []map[string]string{{"role": "user", "content": prompt}},
		})
		if err != nil {
			t.Fatal(err)
		}
		body := dispatchPolicyClientRequest(t, router, episode, "/v1/chat/completions", string(client))
		assertJSONField(t, body, "model", `"vendor/off"`)
		var got int
		if err := json.Unmarshal(body["max_completion_tokens"], &got); err != nil {
			t.Fatalf("max_completion_tokens is %q: %v", body["max_completion_tokens"], err)
		}
		return got
	}

	t.Run("a short prompt is sent the card's cap", func(t *testing.T) {
		if got := limit(t, "episode-bound-short", "fix the failing test"); got != cap {
			t.Fatalf("max_completion_tokens = %d, want the card's %d", got, cap)
		}
	})
	t.Run("a long prompt is sent the room the window leaves", func(t *testing.T) {
		// Roughly four fifths of the window, so the room left is well under
		// the cap but well above the target minimum.
		prompt := strings.Repeat("the failing test prints the wrong line ", window*4*4/5/40)
		got := limit(t, "episode-bound-long", prompt)
		t.Logf("the window left %d of the card's %d", got, cap)
		if got >= cap || got < int(minimumOutputLimit(llmprotocol.OpenAIChatV1)) {
			t.Fatalf("max_completion_tokens = %d, want under the card's %d and at least %d", got, cap, minimumOutputLimit(llmprotocol.OpenAIChatV1))
		}
	})
}
