//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// overrideRouter serves the v5 fixture with dispatch_effort_overrides on one
// worker.
func overrideRouter(t *testing.T, glmFormat, worker, effort string) (*OpenAIRouter, *fakePolicyService) {
	t.Helper()
	return v5RouterWith(t, glmFormat, func(rendered string) string {
		return strings.Replace(rendered, "            allow_experimental_controls: true\n",
			"            allow_experimental_controls: true\n            dispatch_effort_overrides:\n              "+worker+": "+effort+"\n", 1)
	})
}

// On Messages the override is output_config.effort on top of the native
// default base, the steer is unchanged, and the selection names it.
func TestDispatchEffortOverrideOnMessages(t *testing.T) {
	router, fake := overrideRouter(t, "anthropic", "glm", "max")
	client := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"fix the failing test"}]}`
	body, ctx := v5Turn(t, router, fake, v5GLMUp, "override-messages", client)
	if got := gjson.GetBytes(body, "output_config.effort").String(); got != "max" {
		t.Fatalf("output_config.effort = %q:\n%s", got, body)
	}
	if !strings.Contains(string(body), v5UpText) {
		t.Fatalf("the steer did not travel:\n%s", body)
	}
	fields := map[string]interface{}{}
	addPolicyDispatchEffortFields(fields, ctx)
	if fields["policy_dispatch_effort"] != config.RaylineARCPolicyDispatchEffortOverride || fields["policy_dispatched_effort"] != "max" {
		t.Fatalf("fields = %v", fields)
	}
}

// On Chat the override is reasoning.effort.
func TestDispatchEffortOverrideOnChat(t *testing.T) {
	router, fake := overrideRouter(t, "openai", "glm", "high")
	client := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}`
	body, _ := v5Turn(t, router, fake, v5GLMNone, "override-chat", client)
	if got := gjson.GetBytes(body, "reasoning.effort").String(); got != "high" {
		t.Fatalf("reasoning.effort = %q:\n%s", got, body)
	}
}

// A worker without an entry keeps its base's wire: no effort.
func TestDispatchEffortOverrideLeavesOtherWorkers(t *testing.T) {
	router, fake := overrideRouter(t, "openai", "glm", "max")
	client := `{"model":"auto","max_tokens":1024,"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hello"}]}`
	body, ctx := v5Turn(t, router, fake, v5OpusAction, "override-other", client)
	var provider map[string]json.RawMessage
	if err := json.Unmarshal(body, &provider); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(provider["output_config"]), "effort") || gjson.GetBytes(body, "reasoning.effort").Exists() {
		t.Fatalf("an effort travelled on a worker with no override:\n%s", body)
	}
	fields := map[string]interface{}{}
	addPolicyDispatchEffortFields(fields, ctx)
	if _, set := fields["policy_dispatch_effort"]; set {
		t.Fatalf("fields = %v", fields)
	}
}

// dispatch_model_overrides sends a steered GLM action on Opus with the native
// default control (no steer) and Opus's effort override; the selection names
// the remap.
func TestDispatchModelOverrideRemapsWithNativeDefault(t *testing.T) {
	router, fake := v5RouterWith(t, "openai", func(rendered string) string {
		return strings.Replace(rendered, "            allow_experimental_controls: true\n",
			"            allow_experimental_controls: true\n            dispatch_model_overrides:\n              glm: opus\n"+
				"            dispatch_effort_overrides:\n              opus: max\n", 1)
	})
	client := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"fix the failing test"}]}`
	body, ctx := v5Turn(t, router, fake, v5GLMUp, "remap-glm-up", client)
	if got := gjson.GetBytes(body, "model").String(); got != "anthropic/claude-opus-5" {
		t.Fatalf("model = %q:\n%s", got, body)
	}
	if strings.Contains(string(body), "steering instruction") {
		t.Fatalf("a steer travelled on the remapped worker:\n%s", body)
	}
	if got := gjson.GetBytes(body, "output_config.effort").String(); got != "max" {
		t.Fatalf("output_config.effort = %q:\n%s", got, body)
	}
	if ctx.VSRRaylineARC == nil || ctx.VSRRaylineARC.PolicyActionID != v5GLMUp {
		t.Fatalf("the recorded action changed: %+v", ctx.VSRRaylineARC)
	}
	fields := map[string]interface{}{}
	addPolicyDispatchEffortFields(fields, ctx)
	if fields["policy_dispatch_model"] != config.RaylineARCPolicyDispatchModelRemap || fields["policy_dispatched_worker"] != "opus" ||
		fields["policy_dispatched_effort"] != "max" {
		t.Fatalf("fields = %v", fields)
	}
}

// An action on a worker with no remap is unchanged under a remap of another.
func TestDispatchModelOverrideLeavesOtherWorkers(t *testing.T) {
	router, fake := v5RouterWith(t, "openai", func(rendered string) string {
		return strings.Replace(rendered, "            allow_experimental_controls: true\n",
			"            allow_experimental_controls: true\n            dispatch_model_overrides:\n              glm: opus\n", 1)
	})
	client := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}`
	body, ctx := v5Turn(t, router, fake, v5OpusAction, "remap-other", client)
	if got := gjson.GetBytes(body, "model").String(); got != "anthropic/claude-opus-5" {
		t.Fatalf("model = %q", got)
	}
	fields := map[string]interface{}{}
	addPolicyDispatchEffortFields(fields, ctx)
	if _, set := fields["policy_dispatch_model"]; set {
		t.Fatalf("fields = %v", fields)
	}
}
