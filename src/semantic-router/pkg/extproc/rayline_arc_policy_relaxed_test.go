package extproc

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// newPolicySelectorFixtureWith arms a policy selector whose decision carries
// the given episode consistency.
func newPolicySelectorFixtureWith(t *testing.T, consistency string) *policySelectorFixture {
	t.Helper()
	fixture := newPolicySelectorFixture(t, "")
	fixture.decision.Algorithm.RaylineARC.Episode.Consistency = consistency
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
	return fixture
}

// A relaxed cell asks for a relaxed decision, which answers without a session
// revision; a strict cell sends no episode_mode, as before the field existed.
func TestRelaxedPolicyCellSendsEpisodeModeRelaxed(t *testing.T) {
	for _, tc := range []struct {
		consistency string
		want        string
	}{
		{config.RaylineARCConsistencyRelaxed, raylinearc.PolicyEpisodeModeRelaxed},
		{config.RaylineARCConsistencyStrict, ""},
		{"", ""},
	} {
		fixture := newPolicySelectorFixtureWith(t, tc.consistency)
		actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
		fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
		if err != nil {
			t.Fatalf("consistency %q: select: %v", tc.consistency, err)
		}
		if got := fixture.fake.received()[0].EpisodeMode; got != tc.want {
			t.Fatalf("consistency %q sent episode_mode %q, want %q", tc.consistency, got, tc.want)
		}
		if tc.want != "" && result.RaylineARC.SessionRevision != 0 {
			t.Fatalf("relaxed session revision = %d, want 0", result.RaylineARC.SessionRevision)
		}
	}
}

// Only a relaxed call may come back without a session revision: a strict
// cell that gets none has not been held as it asked.
func TestStrictPolicyCellRefusesAMissingSessionRevision(t *testing.T) {
	fixture := newPolicySelectorFixtureWith(t, "")
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	fixture.fake.mu.Lock()
	fixture.fake.nullRevision = true
	fixture.fake.mu.Unlock()
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err == nil || !strings.Contains(err.Error(), "policy_session_revision") {
		t.Fatalf("strict select with a null revision = %v", err)
	}
}

// The readiness probe asks for one relaxed decision, so a service that cannot
// serve relaxed keeps the cell not ready instead of failing every turn.
func TestRelaxedPolicyProbeRefusesAServiceWithoutRelaxed(t *testing.T) {
	fixture := newPolicySelectorFixtureWith(t, config.RaylineARCConsistencyRelaxed)
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
		BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
	})
	for _, reason := range []string{"relaxed_unsupported_by_encoder", "relaxed_needs_unretained_runtime"} {
		fixture.fake.refuseRelaxed(reason)
		err := probeRelaxedPolicyDecide(context.Background(), client, scorer)
		var failure *raylinearc.PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != raylinearc.PolicyRelaxedUnsupportedClass {
			t.Fatalf("%s: probe = %v, want class %s", reason, err, raylinearc.PolicyRelaxedUnsupportedClass)
		}
	}
	fixture.fake.refuseRelaxed("")
	if err := probeRelaxedPolicyDecide(context.Background(), client, scorer); err != nil {
		t.Fatalf("probe against a relaxed-capable service = %v", err)
	}
	sent := fixture.fake.received()
	probe := sent[len(sent)-1]
	if probe.EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed ||
		len(probe.Selection.AvailableActionIDs) != len(scorer.actionOrder) {
		t.Fatalf("probe request = mode %q, %d actions", probe.EpisodeMode, len(probe.Selection.AvailableActionIDs))
	}
}

// relaxedPolicyActions binds one action to each worker of the e2e config.
func relaxedPolicyActions() map[string]config.RaylineARCPolicyBinding {
	return map[string]config.RaylineARCPolicyBinding{
		"think":      policyAction("think", "none", "think-trained", policyTestEffort("high"), nil, ""),
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"off":        policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude-off": policyAction("claude-off", "none", "claude-opus-5", policyTestEffort("none"), nil, ""),
	}
}

// newRelaxedPolicyFake is a fake policy service over relaxedPolicyActions
// that always chooses the thinking-off action.
func newRelaxedPolicyFake(t *testing.T) *fakePolicyService {
	t.Helper()
	actions := relaxedPolicyActions()
	catalog := make([]string, 0, len(actions))
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	off := actions["off"].ActionID
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return off })
	return fake
}

// writeConsistentPolicyConfig is the e2e policy config with the episode's
// consistency set.
func writeConsistentPolicyConfig(t *testing.T, policyURL string, consistency string) string {
	t.Helper()
	path := writePolicyDispatchConfig(t, policyURL, relaxedPolicyActions())
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	anchor := "            idle_ttl_seconds: 900\n"
	if !strings.Contains(string(raw), anchor) {
		t.Fatal("the policy e2e config no longer carries the idle TTL anchor")
	}
	rendered := strings.Replace(string(raw), anchor, anchor+"            consistency: "+consistency+"\n", 1)
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// policyTurnStatus runs one client turn through the request phases and
// returns the immediate status the router answered, or 0 when it forwarded
// the turn to a provider.
func policyTurnStatus(router *OpenAIRouter, episode, content string) (int, error) {
	return policyTurnStatusWithHeaders(router, episode, content, nil)
}

// policyTurnStatusWithHeaders is policyTurnStatus with extra request headers.
func policyTurnStatusWithHeaders(router *OpenAIRouter, episode, content string, extra map[string]string) (int, error) {
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "relaxed-" + content,
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
		}},
	}}
	for key, value := range extra {
		headers.RequestHeaders.Headers.Headers = append(headers.RequestHeaders.Headers.Headers,
			&core.HeaderValue{Key: key, Value: value})
	}
	if _, err := router.handleRequestHeaders(headers, ctx); err != nil {
		return 0, err
	}
	body := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"` + content + `"}]}`
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx)
	if err != nil {
		return 0, err
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return int(immediate.GetStatus().GetCode()), nil
	}
	return 0, nil
}

// End to end: concurrent turns on one episode of a relaxed policy cell all
// reach the policy service together and are all forwarded; on a strict cell
// they queue behind the episode lease.
func TestRelaxedPolicyCellDecidesConcurrentTurnsTogether(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	const turns = 3
	for _, tc := range []struct {
		consistency string
		wantPeak    int
	}{
		{config.RaylineARCConsistencyRelaxed, turns},
		{config.RaylineARCConsistencyStrict, 1},
	} {
		t.Run(tc.consistency, func(t *testing.T) {
			fake := newRelaxedPolicyFake(t)
			router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), tc.consistency))
			if err != nil {
				t.Fatalf("build router: %v", err)
			}
			awaitPolicySelectorArmed(t, router)
			fake.holdUntilConcurrent(turns)

			statuses := make([]int, turns)
			errs := make([]error, turns)
			var wg sync.WaitGroup
			for i := range turns {
				wg.Add(1)
				go func() {
					defer wg.Done()
					statuses[i], errs[i] = policyTurnStatus(router, "episode-concurrent", fmt.Sprintf("turn %c", 'a'+i))
				}()
			}
			wg.Wait()
			for i := range turns {
				if errs[i] != nil {
					t.Fatalf("turn %d: %v", i, errs[i])
				}
				if tc.consistency == config.RaylineARCConsistencyRelaxed && statuses[i] != 0 {
					t.Fatalf("relaxed turn %d answered %d, want forwarded", i, statuses[i])
				}
			}
			if peak := fake.peakConcurrency(); peak != tc.wantPeak {
				t.Fatalf("%s: %d decide calls in flight at once, want %d", tc.consistency, peak, tc.wantPeak)
			}
		})
	}
}

// A relaxed policy cell does not arm while its service refuses relaxed calls,
// and arms once the service serves one.
func TestRelaxedPolicyCellArmsOnlyOnceTheServiceServesRelaxed(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	fake := newRelaxedPolicyFake(t)
	fake.refuseRelaxed("relaxed_unsupported_by_encoder")
	router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), config.RaylineARCConsistencyRelaxed))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for len(fake.received()) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("the relaxed probe never reached the service")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if status, err := policyTurnStatus(router, "episode-unready", "hello"); err != nil || status != 503 {
		t.Fatalf("a turn while relaxed is refused = %d, %v; want 503", status, err)
	}
	fake.refuseRelaxed("")
	awaitPolicySelectorArmed(t, router)
}

// A service that predates episode_mode may accept the field and serve the
// call strict. The probe refuses such an answer, as it does one naming
// another package, and a relaxed selection refuses a revisioned decision.
func TestRelaxedPolicyCellRefusesAServiceThatIgnoresTheMode(t *testing.T) {
	fixture := newPolicySelectorFixtureWith(t, config.RaylineARCConsistencyRelaxed)
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
		BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
	})

	fixture.fake.mu.Lock()
	fixture.fake.ignoreEpisodeMode = true
	fixture.fake.mu.Unlock()
	if err := probeRelaxedPolicyDecide(context.Background(), client, scorer); !errors.Is(err, errRelaxedPolicyIgnored) {
		t.Fatalf("probe against a service ignoring episode_mode = %v", err)
	}
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err == nil || !strings.Contains(err.Error(), "policy_session_revision") {
		t.Fatalf("relaxed select answered strict = %v", err)
	}

	fixture.fake.mu.Lock()
	fixture.fake.ignoreEpisodeMode = false
	fixture.fake.answerAs = &raylinearc.PolicyPackageRef{Alias: "other", PackageSHA256: policyTestPackage}
	fixture.fake.mu.Unlock()
	if err := probeRelaxedPolicyDecide(context.Background(), client, scorer); !errors.Is(err, errRelaxedPolicyIgnored) {
		t.Fatalf("probe answered for another package = %v", err)
	}
}

// End to end: a policy service that reports package verification
// (pathfinder#3120) on its listing and decide responses still arms the cell
// and has its decisions dispatched.
func TestPolicyCellServesAServiceReportingVerification(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	for _, verification := range []raylinearc.PolicyVerification{
		raylinearc.PolicyPackageVerified, raylinearc.PolicyPackageUnverified,
	} {
		t.Run(string(verification), func(t *testing.T) {
			fake := newRelaxedPolicyFake(t)
			fake.mu.Lock()
			fake.verification = verification
			fake.mu.Unlock()
			if listing := fake.packages(); listing.Packages[0].Verification != verification {
				t.Fatalf("the fake listing reports %q", listing.Packages[0].Verification)
			}
			router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), config.RaylineARCConsistencyStrict))
			if err != nil {
				t.Fatalf("build router: %v", err)
			}
			awaitPolicySelectorArmed(t, router)
			if status, err := policyTurnStatus(router, "episode-verified", "hello"); err != nil || status != 0 {
				t.Fatalf("turn against a service reporting %q = %d, %v; want forwarded", verification, status, err)
			}
			if len(fake.received()) != 1 {
				t.Fatalf("decide calls = %d, want 1", len(fake.received()))
			}
		})
	}
}
