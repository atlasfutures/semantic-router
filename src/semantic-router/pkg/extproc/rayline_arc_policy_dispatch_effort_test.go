//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Under dispatch_effort: provider_default, v4 actions reach the provider
// without their declared effort, as the package's training turns did, while a
// budget still travels, the steering suffix still renders, and a thinking-off
// action stays off.
func TestPolicyDispatchAtProviderDefaultEffort(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	budget := int64(4096)
	actions := map[string]config.RaylineARCPolicyBinding{
		"think-up":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"think-budget":  policyAction("think", "none", "think-trained", nil, &budget, ""),
		"claude":        policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-budget": policyAction("claude", "none", "claude-opus-5", nil, &budget, ""),
		"off":           policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"off-up":        policyAction("off", "up", "off-trained", policyTestEffort("none"), nil, policyTestUp),
		"claude-off-up": policyAction("claude-off", "up", "claude-opus-5", policyTestEffort("none"), nil, policyTestUp),
	}
	catalog := make([]string, 0, len(actions))
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	path := writePolicyDispatchConfig(t, fake.URL(), actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	anchor := "            package_sha256: " + policyTestPackage + "\n"
	if !strings.Contains(string(raw), anchor) {
		t.Fatal("the config template changed; update the dispatch_effort anchor")
	}
	rendered := strings.Replace(string(raw), anchor, anchor+"            dispatch_effort: provider_default\n", 1)
	if err = os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	dispatch := func(t *testing.T, action string) map[string]json.RawMessage {
		t.Helper()
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[action].ActionID })
		return dispatchPolicyRequest(t, router, "episode-default-"+action)
	}

	t.Run("a steered effort action on Chat sends no effort, and the steer", func(t *testing.T) {
		body := dispatch(t, "think-up")
		assertJSONField(t, body, "model", `"vendor/think"`)
		if _, present := body["reasoning"]; present {
			t.Fatalf("the declared effort travelled: %s", body["reasoning"])
		}
		if _, present := body["reasoning_effort"]; present {
			t.Fatalf("reasoning_effort travelled: %s", body["reasoning_effort"])
		}
		messages := chatMessages(t, body)
		if last := string(messages[len(messages)-1]); !strings.Contains(last, policyTestUp) {
			t.Fatalf("the steer was lost: %s", last)
		}
	})
	t.Run("a budget action on Chat keeps its budget", func(t *testing.T) {
		assertJSONField(t, dispatch(t, "think-budget"), "reasoning", `{"max_tokens":4096}`)
	})
	t.Run("a budget action on Messages keeps enabled thinking with its budget", func(t *testing.T) {
		body := dispatch(t, "claude-budget")
		assertJSONField(t, body, "thinking", `{"type":"enabled","budget_tokens":4096}`)
		if strings.Contains(string(body["output_config"]), "effort") {
			t.Fatalf("an effort travelled beside the budget: %s", body["output_config"])
		}
	})
	t.Run("an effort action on Messages sends no effort", func(t *testing.T) {
		body := dispatch(t, "claude")
		assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
		if strings.Contains(string(body["output_config"]), "effort") {
			t.Fatalf("the declared effort travelled: %s", body["output_config"])
		}
	})
	t.Run("a steered thinking-off action on Messages stays off", func(t *testing.T) {
		body := dispatch(t, "claude-off-up")
		assertJSONField(t, body, "thinking", `{"type":"disabled"}`)
		if !strings.Contains(string(body["messages"]), policyTestUp) {
			t.Fatalf("the steer was lost: %s", body["messages"])
		}
	})
}
