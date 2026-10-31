//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// familyFixture is fallbackFixture with vendor models: the "off" worker,
// which the service prefers, serves claude-opus-5.5, and "think" serves
// glm-5.3, a model of no excluded family.
func familyFixture(t *testing.T, familyExclusion bool) (*policySelectorFixture, *raylinearc.MemoryEpisodeStore, string) {
	t.Helper()
	decision := policyTestDecision(config.RaylineARCModelScheduleTaskTurnCompaction)
	bind := func(worker, level, model string, effort *string, suffix string) config.RaylineARCPolicyBinding {
		return config.RaylineARCPolicyBinding{
			ActionID: config.RaylineARCPolicyActionID(model, effort, nil, suffix),
			Worker:   worker, Level: level, Model: model, Effort: effort,
		}
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	policy.Bindings = []config.RaylineARCPolicyBinding{
		bind("think", "none", "zhipu/glm-5.3", policyTestEffort("high"), ""),
		bind("think", "up", "zhipu/glm-5.3", policyTestEffort("high"), policyTestUp),
		bind("off", "none", "anthropic/claude-opus-5.5", policyTestEffort("none"), ""),
	}
	policy.Fallback = &config.RaylineARCPolicyFallbackConfig{Enabled: true, FamilyExclusion: familyExclusion}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, policyTestActionIDs(policy.Bindings))
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fake.URL(), TotalTimeout: 5 * time.Second}),
	})
	claudeAction := policy.Bindings[2].ActionID
	fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		if slices.Contains(request.Selection.AvailableActionIDs, claudeAction) {
			return claudeAction
		}
		return request.Selection.AvailableActionIDs[0]
	})
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 8, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	return &policySelectorFixture{fake: fake, selector: selector, decision: decision}, store, claudeAction
}

// selectWithFamilies is selectOn on episode, with the session's family
// exclusions read as the request path reads them.
func (fixture *policySelectorFixture) selectWithFamilies(
	t *testing.T, router *OpenAIRouter, reqCtx *RequestContext, episode string, state *raylinearc.EpisodeState,
	body []byte, privacyExcluded []bool,
) (*selection.SelectionResult, error) {
	t.Helper()
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash:       episode,
			RequestID:           "req-policy-test",
			State:               state,
			RawRequest:          body,
			RequestFormat:       policyFormatAnthropic,
			PrivacyExcludedArms: privacyExcluded,
			FamilyExclusions:    router.loadRaylineARCFamilyExclusions(fixture.decision.Algorithm.RaylineARC.PolicyService, reqCtx),
		},
	})
}

// The prod sequence of 2026-10-29 (refusal 1): opus refused a new
// conversation's first turn; Claude Code's retry 38 s later edited the
// message and opened a new context (a fork, turn 0), whose context
// exclusions start empty; the retry went to opus again and was refused
// again. With the family exclusion on, the refusal excludes claude for the
// session, so the retry in its new context is offered no claude model.
// Control: with it off the retry is offered opus again, as in prod.
func TestRaylineARCFamilyExclusionOutlivesTheRetrysNewContext(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		fixture, store, claudeAction := familyFixture(t, enabled)
		router := &OpenAIRouter{RaylineARCEpisodeStore: store}
		session := raylinearc.SessionKeyDigest("user-1", "conv-1")
		body := policyTestRequest(t, map[string]any{"role": "user", "content": "audit this binary"})

		first := strings.Repeat("a", 64)
		lease, state, err := store.Prepare(context.Background(), first, 2)
		if err != nil {
			t.Fatal(err)
		}
		reqCtx := &RequestContext{RequestID: "req-1", TraceContext: context.Background(), RaylineARCSessionKey: session}
		decided, err := fixture.selectWithFamilies(t, router, reqCtx, first, state, body, nil)
		if err != nil || decided.RaylineARC.PolicyActionID != claudeAction {
			t.Fatalf("enabled=%v: first turn chose %+v, %v", enabled, decided, err)
		}
		reqCtx.RaylineARCTransaction = newRaylineARCEpisodeTransaction(store, lease, state, first, time.Minute, nil)
		reqCtx.RaylineARCTransaction.markSelection(decided.RaylineARC.SelectedArm, 10)
		stageRaylineARCPolicySelection(reqCtx, decided.RaylineARC, "")
		reqCtx.VSRRaylineARC, reqCtx.VSRSelectedDecision = decided.RaylineARC, fixture.decision
		logs := captureLogs(t)
		declineRefusedTurn(reqCtx)
		if err := reqCtx.RaylineARCTransaction.abort(context.Background(), "test"); err != nil {
			t.Fatal(err)
		}
		families, _ := store.LoadSessionFamilyExclusions(context.Background(), session)
		if slices.Equal(families, []string{"claude"}) != enabled {
			t.Fatalf("enabled=%v: session families %v", enabled, families)
		}
		if enabled {
			event := findLogEvent(t, logs, "rayline_arc_family_exclusion")
			raw, _ := json.Marshal(event)
			if !strings.Contains(string(raw), `"family":"claude"`) || !strings.Contains(string(raw), `"ttl_seconds":86400`) {
				t.Fatalf("family exclusion log: %s", raw)
			}
		}

		// The retry_edit: another episode (a fork), a fresh context.
		retry := strings.Repeat("b", 64)
		_, retryState, err := store.Prepare(context.Background(), retry, 2)
		if err != nil {
			t.Fatal(err)
		}
		retryCtx := &RequestContext{RequestID: "req-2", TraceContext: context.Background(), RaylineARCSessionKey: session}
		retried, err := fixture.selectWithFamilies(t, router, retryCtx, retry, retryState, body, nil)
		if err != nil {
			t.Fatal(err)
		}
		offered := offeredActions(fixture, 1)
		if slices.Contains(offered, claudeAction) == enabled || (retried.RaylineARC.PolicyActionID == claudeAction) == enabled {
			t.Fatalf("enabled=%v: retry offered %v, chose %s", enabled, offered, retried.RaylineARC.PolicyActionID)
		}

		// Another session is not touched.
		otherCtx := &RequestContext{RequestID: "req-3", RaylineARCSessionKey: raylinearc.SessionKeyDigest("user-2", "conv-9")}
		other := strings.Repeat("c", 64)
		_, otherState, _ := store.Prepare(context.Background(), other, 2)
		if chosen, err := fixture.selectWithFamilies(t, router, otherCtx, other, otherState, body, nil); err != nil ||
			chosen.RaylineARC.PolicyActionID != claudeAction {
			t.Fatalf("enabled=%v: another session chose %+v, %v", enabled, chosen, err)
		}
	}
}

// A family exclusion never empties the offer: a session whose every
// remaining model is of the excluded family (a private turn whose allowed
// arms are all claude) is offered that family again instead of failing.
func TestRaylineARCFamilyExclusionYieldsWhenItWouldEmptyTheOffer(t *testing.T) {
	fixture, store, claudeAction := familyFixture(t, true)
	router := &OpenAIRouter{RaylineARCEpisodeStore: store}
	session := raylinearc.SessionKeyDigest("user-1", "conv-1")
	if err := store.AddSessionFamilyExclusion(context.Background(), session, "claude", time.Hour); err != nil {
		t.Fatal(err)
	}
	episode := strings.Repeat("d", 64)
	_, state, _ := store.Prepare(context.Background(), episode, 2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "audit this binary"})
	reqCtx := &RequestContext{RequestID: "req-1", RaylineARCSessionKey: session}
	result, err := fixture.selectWithFamilies(t, router, reqCtx, episode, state, body, []bool{true, false})
	if err != nil || result.RaylineARC.PolicyActionID != claudeAction {
		t.Fatalf("private turn with only claude allowed: %+v, %v", result, err)
	}
}

// With the family exclusion off, or without a session, nothing is read.
func TestRaylineARCFamilyExclusionReadsNothingWhenOff(t *testing.T) {
	store, _ := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 1, IdleTTL: time.Minute})
	_ = store.AddSessionFamilyExclusion(context.Background(), "s", "claude", time.Hour)
	router := &OpenAIRouter{RaylineARCEpisodeStore: store}
	on := &config.RaylineARCPolicyServiceConfig{Fallback: &config.RaylineARCPolicyFallbackConfig{Enabled: true, FamilyExclusion: true}}
	off := &config.RaylineARCPolicyServiceConfig{Fallback: &config.RaylineARCPolicyFallbackConfig{Enabled: true}}
	if got := router.loadRaylineARCFamilyExclusions(on, &RequestContext{RaylineARCSessionKey: "s"}); !slices.Equal(got, []string{"claude"}) {
		t.Fatalf("on: %v", got)
	}
	if got := router.loadRaylineARCFamilyExclusions(off, &RequestContext{RaylineARCSessionKey: "s"}); got != nil {
		t.Fatalf("off: %v", got)
	}
	if got := router.loadRaylineARCFamilyExclusions(on, &RequestContext{}); got != nil {
		t.Fatalf("no session: %v", got)
	}
}
