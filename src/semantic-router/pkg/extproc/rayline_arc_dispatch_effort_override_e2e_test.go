//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tidwall/gjson"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
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
	// The policy's arm, which the episode holds and offers from, stays GLM's;
	// only the dispatch moves to Opus.
	refs := ctx.VSRSelectedDecision.ModelRefs
	if got := refs[ctx.VSRRaylineARC.SelectedArm].Model; got != "glm" {
		t.Fatalf("SelectedArm names %q, want the bound worker glm", got)
	}
	if ctx.VSRRaylineARC.DispatchArm == nil || refs[*ctx.VSRRaylineARC.DispatchArm].Model != "opus" {
		t.Fatalf("DispatchArm = %v, want opus", ctx.VSRRaylineARC.DispatchArm)
	}
	// A refusal on the remapped turn is Opus's, not GLM's.
	if got := refusedModel(ctx); got == "" || got != ctx.VSRRaylineARC.DispatchModel || got == ctx.VSRRaylineARC.PolicyActionModel {
		t.Fatalf("refusedModel = %q, DispatchModel %q, PolicyActionModel %q", got, ctx.VSRRaylineARC.DispatchModel, ctx.VSRRaylineARC.PolicyActionModel)
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

// The override's field follows the worker's reasoning transport.
func TestDispatchEffortOverridePath(t *testing.T) {
	for _, test := range []struct {
		format    string
		transport modelcatalog.ReasoningTransport
		want      string
	}{
		{thinkingcontrol.FormatMessages, "", "output_config.effort"},
		{thinkingcontrol.FormatResponses, "", "reasoning.effort"},
		{thinkingcontrol.FormatChat, modelcatalog.ReasoningTransportReasoningObject, "reasoning.effort"},
		{thinkingcontrol.FormatChat, modelcatalog.ReasoningTransportTopLevelEffort, "reasoning_effort"},
	} {
		got, err := dispatchEffortOverridePath(test.format, test.transport)
		if err != nil || got != test.want {
			t.Fatalf("%s/%s: %q, %v; want %q", test.format, test.transport, got, err, test.want)
		}
	}
	if _, err := dispatchEffortOverridePath(thinkingcontrol.FormatChat, modelcatalog.ReasoningTransportTopLevelBoolean); err == nil {
		t.Fatal("a Chat transport with no effort field was accepted")
	}
}

// A remapped action's route, model and hold resolve through its to-worker;
// an unmapped action's through its own.
func TestPolicyScorerDispatchIdentity(t *testing.T) {
	scorer := &policyServiceScorer{
		workerIDs:   []string{"glm", "opus"},
		actionOrder: []string{"glm-up", "opus-max"},
		bindings: map[string]policyBinding{
			"glm-up":   {arm: 0, dispatchArm: 1, model: "glm-5"},
			"opus-max": {arm: 1, dispatchArm: 1, model: "claude-opus-5"},
		},
	}
	if got := scorer.armDispatch(0); got != 1 {
		t.Fatalf("armDispatch(glm) = %d, want opus", got)
	}
	if got := scorer.armDispatch(1); got != 1 {
		t.Fatalf("armDispatch(opus) = %d", got)
	}
	if got := scorer.dispatchedModel("glm-up"); got != "claude-opus-5" {
		t.Fatalf("dispatchedModel(glm-up) = %q", got)
	}
	if got := scorer.dispatchedModel("opus-max"); got != "claude-opus-5" {
		t.Fatalf("dispatchedModel(opus-max) = %q", got)
	}
	if got := scorer.actionModel("glm-up"); got != "glm-5" {
		t.Fatalf("actionModel(glm-up) = %q, the policy attribution must stay", got)
	}
}

// routing.exclusive and a lookup's results follow the worker a remapped
// action is sent on.
func TestPolicyScorerRouteControlsFollowDispatch(t *testing.T) {
	scorer := &policyServiceScorer{
		workerIDs:   []string{"glm", "opus"},
		workers:     []raylinearc.WorkerManifest{{ID: "glm", Model: "z-ai/glm-5"}, {ID: "opus", Model: "anthropic/claude-opus-5"}},
		actionOrder: []string{"glm-up", "opus-max"},
		bindings: map[string]policyBinding{
			"glm-up":   {arm: 0, dispatchArm: 1},
			"opus-max": {arm: 1, dispatchArm: 1},
		},
	}
	if got := routeExclusiveOffer(scorer, scorer.actionOrder, []string{"z-ai/glm-5"}); len(got) != 0 {
		t.Fatalf("GLM-only prices kept %v, but every action is sent on Opus", got)
	}
	if got := routeExclusiveOffer(scorer, scorer.actionOrder, []string{"anthropic/claude-opus-5"}); len(got) != 2 {
		t.Fatalf("Opus prices kept %v, want both actions", got)
	}
	controls := &selection.RaylineARCRouteControls{Points: []raylinearc.PolicyTierPoint{{Name: "cheap"}}}
	response := &raylinearc.PolicyDecisionResponse{}
	response.Decision.TierPicks = map[string]raylinearc.PolicyTierPick{"cheap": {SelectedActionID: "glm-up"}}
	trace := &selection.RaylineARCTrace{}
	if failure := recordRouteControls(trace, scorer, controls, response, scorer.actionOrder, false); failure != "" {
		t.Fatalf("recordRouteControls = %q", failure)
	}
	if got := trace.RouteResults["cheap"].Arm; got != 1 {
		t.Fatalf("result arm = %d, want the to-worker opus", got)
	}
}
