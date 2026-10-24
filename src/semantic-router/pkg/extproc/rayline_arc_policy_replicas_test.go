package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// newPolicyReplicaFixture arms a selector on two policy replicas, s1 and s2.
func newPolicyReplicaFixture(t *testing.T) (*policySelectorFixture, map[string]*fakePolicyService) {
	t.Helper()
	decision := policyTestDecision("")
	catalog := policyTestActionIDs(decision.Algorithm.RaylineARC.PolicyService.Bindings)
	fakes := map[string]*fakePolicyService{
		"s1": newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog),
		"s2": newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog),
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	policy.Replicas = []config.RaylineARCEncoderReplicaConfig{
		{ID: "s1", BaseURL: fakes["s1"].URL(), State: config.RaylineARCEncoderActive},
		{ID: "s2", BaseURL: fakes["s2"].URL(), State: config.RaylineARCEncoderActive},
	}
	policy.Failover = config.RaylineARCEncoderFailoverConfig{
		SchemaVersion: config.RaylineARCPolicyFailoverV1, UnavailableStatusCodes: []int{503},
		UnavailableCooldownSeconds: 60, MaxRemaps: 1,
	}
	for _, fake := range fakes {
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return catalog[0] })
	}
	pool, err := newRaylineARCPolicyServicePool(policy, func(baseURL string) *raylinearc.PolicyServiceClient {
		return raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: baseURL, TotalTimeout: 5 * time.Second})
	})
	if err != nil {
		t.Fatal(err)
	}
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	selector.arm(&raylineARCArmedComponents{
		scorer:     newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission:  raylinearc.NewAdmissionGate(0),
		policyPool: pool,
	})
	return &policySelectorFixture{selector: selector, decision: decision}, fakes
}

// An episode decides on one replica, keeps it through the owner its state
// persists, and moves once, to the other replica, when its owner answers
// 503. The trace carries the owner the episode commit persists.
func TestPolicyReplicasKeepAnEpisodeOnItsOwnerAndFailOverOnce(t *testing.T) {
	fixture, fakes := newPolicyReplicaFixture(t)
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})

	first, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatal(err)
	}
	owner := first.RaylineARC.EncoderReplicaID
	other := map[string]string{"s1": "s2", "s2": "s1"}[owner]
	if owner == "" || first.RaylineARC.EncoderFailover || len(fakes[owner].received()) != 1 || len(fakes[other].received()) != 0 {
		t.Fatalf("first turn owner %q failover %t", owner, first.RaylineARC.EncoderFailover)
	}

	// The commit persists the owner; the next turns read it.
	state.EncoderOwner = owner
	state.EncoderVisitedOwners = first.RaylineARC.EncoderVisitedReplicaIDs
	for turn := 0; turn < 3; turn++ {
		next, err := fixture.selectOn(t, state, body)
		if err != nil {
			t.Fatal(err)
		}
		if next.RaylineARC.EncoderReplicaID != owner || next.RaylineARC.EncoderFailover {
			t.Fatalf("turn %d moved to %q", turn, next.RaylineARC.EncoderReplicaID)
		}
	}
	if len(fakes[other].received()) != 0 {
		t.Fatal("a turn reached the other replica")
	}

	fakes[owner].failNext("backend_unavailable")
	moved, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatal(err)
	}
	if moved.RaylineARC.EncoderReplicaID != other || !moved.RaylineARC.EncoderFailover ||
		moved.RaylineARC.EncoderAttempts != 2 || len(moved.RaylineARC.EncoderVisitedReplicaIDs) != 2 {
		t.Fatalf("failover trace %+v", moved.RaylineARC)
	}
	if len(fakes[other].received()) != 1 {
		t.Fatalf("other replica calls = %d, want 1", len(fakes[other].received()))
	}
}

// Without replicas the cell has no pool, and a turn's trace names no replica:
// the commit keeps the episode's state exactly as before.
func TestPolicyWithoutReplicasRecordsNoOwner(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "hi"}))
	if err != nil {
		t.Fatal(err)
	}
	if result.RaylineARC.EncoderReplicaID != "" || len(result.RaylineARC.EncoderVisitedReplicaIDs) != 0 ||
		result.RaylineARC.EncoderReplicaIndex != -1 {
		t.Fatalf("single-service trace names a replica: %+v", result.RaylineARC)
	}
}
