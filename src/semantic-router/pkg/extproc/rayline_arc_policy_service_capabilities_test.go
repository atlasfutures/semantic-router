//go:build !windows && cgo

package extproc

import (
	"context"
	"errors"
	"net/url"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCPolicyServiceRequirementsFollowTheCellsMembers(t *testing.T) {
	weight := func(value float64) *float64 { return &value }
	for name, check := range map[string]struct {
		v2           *config.RaylineARCDecisionsV2Config
		explore      bool
		evalCell     bool
		untrusted    bool
		capabilities []string
		weights      []float64
	}{
		"no opt-in members":  {},
		"empty decisions_v2": {v2: &config.RaylineARCDecisionsV2Config{}},
		"name only": {
			v2:           &config.RaylineARCDecisionsV2Config{Name: "keen"},
			capabilities: []string{"decisions_v2"},
		},
		"cost weight only": {
			v2:           &config.RaylineARCDecisionsV2Config{CostWeight: weight(0.5)},
			capabilities: []string{"decisions_v2"}, weights: []float64{0.5},
		},
		"the fixture's members": {
			v2:           fixtureDecisionsV2(),
			capabilities: []string{"decisions_v2", "tier_points", "response_fields"},
			weights:      []float64{1, 2, 1, 0},
		},
		"exploration only": {explore: true, capabilities: []string{"exploration"}},
		// Either excludes every turn, so the member is never sent.
		"exploration on an eval cell":        {explore: true, evalCell: true},
		"exploration with untrusted headers": {explore: true, untrusted: true},
		"response fields and exploration": {
			v2:           &config.RaylineARCDecisionsV2Config{ResponseFields: []string{"thinking"}},
			explore:      true,
			capabilities: []string{"decisions_v2", "response_fields", "exploration"},
		},
	} {
		policy := &config.RaylineARCPolicyServiceConfig{DecisionsV2: check.v2}
		if check.explore {
			policy.Exploration = fixtureExploration()
			policy.EvalCell, policy.TrustTurnSignalHeaders = check.evalCell, !check.untrusted
		}
		got := raylineARCPolicyServiceRequirements(policy)
		if !slices.Equal(got.Capabilities, check.capabilities) || !slices.Equal(got.CostWeights, check.weights) {
			t.Errorf("%s: requirements = %+v, want %v %v", name, got, check.capabilities, check.weights)
		}
	}
}

// The ready/not-ready matrix: a cell is ready only when the service
// advertises every capability its members need and the served package's
// bounds hold every cost weight it sends. A cell with no opt-in members is
// ready against a service of any age.
func TestPolicyReadinessChecksTheAdvertisedCapabilities(t *testing.T) {
	weight := func(value float64) *float64 { return &value }
	unit := &raylinearc.PolicyAdjustableCostWeight{Stated: true, Bounds: &raylinearc.PolicyCostWeightBounds{Min: 0, Max: 1}}
	null := &raylinearc.PolicyAdjustableCostWeight{Stated: true}
	for name, check := range map[string]struct {
		v2         *config.RaylineARCDecisionsV2Config
		explore    bool
		evalCell   bool
		legacy     bool
		advertised []string
		bounds     *raylinearc.PolicyAdjustableCostWeight
		class      string
		capability string
	}{
		"v1 cell, older service":      {legacy: true},
		"v1 cell, current service":    {},
		"v2 cell, current service":    {v2: fixtureDecisionsV2()},
		"exploring, current service":  {explore: true},
		"v2 and exploring, current":   {v2: fixtureDecisionsV2(), explore: true},
		"v2 cell, older service":      {v2: fixtureDecisionsV2(), legacy: true, class: raylinearc.PolicyCapabilityUnsupportedClass, capability: "decisions_v2"},
		"named cell, older service":   {v2: &config.RaylineARCDecisionsV2Config{Name: "keen"}, legacy: true, class: raylinearc.PolicyCapabilityUnsupportedClass, capability: "decisions_v2"},
		"exploring, older service":    {explore: true, legacy: true, class: raylinearc.PolicyCapabilityUnsupportedClass, capability: "exploration"},
		"eval cell, older service":    {explore: true, evalCell: true, legacy: true},
		"tier points not advertised":  {v2: fixtureDecisionsV2(), advertised: []string{"decisions_v2", "response_fields"}, class: raylinearc.PolicyCapabilityUnsupportedClass, capability: "tier_points"},
		"response fields not offered": {v2: fixtureDecisionsV2(), advertised: []string{"decisions_v2", "tier_points"}, class: raylinearc.PolicyCapabilityUnsupportedClass, capability: "response_fields"},
		"exploration not advertised":  {explore: true, advertised: []string{"decisions_v2", "tier_points", "response_fields"}, class: raylinearc.PolicyCapabilityUnsupportedClass, capability: "exploration"},
		"cost weight, no bounds": {
			v2: &config.RaylineARCDecisionsV2Config{CostWeight: weight(0.5)}, bounds: null,
			class: raylinearc.PolicyCostWeightNotAdjustableClass,
		},
		"cost weight inside the bounds": {v2: &config.RaylineARCDecisionsV2Config{CostWeight: weight(1)}, bounds: unit},
		"tier point outside the bounds": {
			v2: fixtureDecisionsV2(), bounds: unit, class: raylinearc.PolicyCostWeightOutOfBoundsClass,
		},
		"members without a weight, no bounds": {
			v2: &config.RaylineARCDecisionsV2Config{Name: "keen", ResponseFields: []string{"thinking"}}, bounds: null,
		},
	} {
		fixture := newPolicySelectorFixture(t, "")
		policy := fixture.decision.Algorithm.RaylineARC.PolicyService
		policy.DecisionsV2 = check.v2
		if check.explore {
			policy.Exploration = fixtureExploration()
			policy.EvalCell, policy.TrustTurnSignalHeaders = check.evalCell, true
		}
		fixture.fake.mu.Lock()
		fixture.fake.legacyListing, fixture.fake.advertised, fixture.fake.bounds = check.legacy, check.advertised, check.bounds
		fixture.fake.mu.Unlock()
		client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		})
		err := requireRaylineARCPolicyServing(context.Background(), client, policy)
		if check.class == "" {
			if err != nil {
				t.Errorf("%s: not ready: %v", name, err)
			}
			continue
		}
		var failure *raylinearc.PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != check.class {
			t.Errorf("%s: readiness = %v, want class %s", name, err, check.class)
			continue
		}
		if check.capability != "" && failure.Detail["capability"] != check.capability {
			t.Errorf("%s: refusal names %v, want %s", name, failure.Detail, check.capability)
		}
	}
}

// Against pathfinder's pinned capabilities listing (exploration advertised
// since pathfinder#5436), a cell configured for exploration is ready, alone
// or beside the v2 members that need no cost weight.
func TestPolicyReadinessOfAnExplorationCellAgainstThePinnedListing(t *testing.T) {
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/packages_response_capabilities.v1.json")
	if err != nil {
		t.Fatal(err)
	}
	listing, err := raylinearc.DecodePolicyPackagesResponse(raw)
	if err != nil {
		t.Fatal(err)
	}
	for name, v2 := range map[string]*config.RaylineARCDecisionsV2Config{
		"exploration only":                nil,
		"response fields and exploration": {Name: "keen", ResponseFields: []string{"thinking"}},
	} {
		policy := &config.RaylineARCPolicyServiceConfig{DecisionsV2: v2, Exploration: fixtureExploration(), TrustTurnSignalHeaders: true}
		required := raylineARCPolicyServiceRequirements(policy)
		if !slices.Contains(required.Capabilities, raylinearc.PolicyCapabilityExploration) {
			t.Fatalf("%s: requirements %v do not ask for exploration", name, required.Capabilities)
		}
		for index := range listing.Packages {
			if err := raylinearc.CheckPolicyServiceRequirements(listing, &listing.Packages[index], required); err != nil {
				t.Errorf("%s, %s: not ready: %v", name, listing.Packages[index].Alias, err)
			}
		}
	}
}

// The package checks still come first: a package the service has not loaded
// is not ready whatever it advertises.
func TestPolicyReadinessStillRequiresTheLoadedPackage(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	fixture.fake.cold.Store(true)
	client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
		BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
	})
	err := requireRaylineARCPolicyServing(context.Background(), client, fixture.decision.Algorithm.RaylineARC.PolicyService)
	var failure *raylinearc.PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != "package_not_loaded" {
		t.Fatalf("readiness = %v, want package_not_loaded", err)
	}
}

// End to end: a decisions_v2 cell against a service that predates the
// capabilities listing stays not ready, and arms once the service advertises
// what it sends. A v1 cell arms against the older service at once.
func TestPolicyCellArmsOnlyWhenTheServiceAdvertisesItsMembers(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	t.Run("v1 cell, older service", func(t *testing.T) {
		fake := newRelaxedPolicyFake(t)
		fake.mu.Lock()
		fake.legacyListing = true
		fake.mu.Unlock()
		router, err := NewOpenAIRouter(writePolicyDispatchConfig(t, fake.URL(), relaxedPolicyActions()))
		if err != nil {
			t.Fatalf("build router: %v", err)
		}
		awaitPolicySelectorArmed(t, router)
	})
	t.Run("decisions_v2 cell", func(t *testing.T) {
		fake := newRelaxedPolicyFake(t)
		fake.mu.Lock()
		fake.legacyListing = true
		fake.mu.Unlock()
		router, err := NewOpenAIRouter(writeDecisionsV2PolicyConfig(t, fake.URL()))
		if err != nil {
			t.Fatalf("build router: %v", err)
		}
		selector := policySelectorOf(t, router)
		deadline := time.Now().Add(5 * time.Second)
		for len(fakeListingQueries(fake)) == 0 && time.Now().Before(deadline) {
			time.Sleep(20 * time.Millisecond)
		}
		queries := fakeListingQueries(fake)
		if len(queries) == 0 || queries[0].Get("detail") != "capabilities" {
			t.Fatalf("listing queries = %v, want detail=capabilities", queries)
		}
		time.Sleep(200 * time.Millisecond)
		if selector.armedComponents() != nil {
			t.Fatal("a decisions_v2 cell armed against a service that advertises nothing")
		}
		fake.mu.Lock()
		fake.legacyListing = false
		fake.mu.Unlock()
		awaitPolicySelectorArmed(t, router)
	})
}

func fakeListingQueries(fake *fakePolicyService) []url.Values {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([]url.Values(nil), fake.listingQueries...)
}

func policySelectorOf(t *testing.T, router *OpenAIRouter) *raylineARCSelector {
	t.Helper()
	registered, ok := router.ModelSelector.Get(selection.MethodRaylineARC)
	if !ok {
		t.Fatal("the ARC selector is not registered")
	}
	selector, ok := registered.(*raylineARCSelector)
	if !ok {
		t.Fatalf("the ARC selector is %T", registered)
	}
	return selector
}

// writeDecisionsV2PolicyConfig is the e2e policy config with a decisions_v2
// cost weight and tier point.
func writeDecisionsV2PolicyConfig(t *testing.T, policyURL string) string {
	t.Helper()
	path := writePolicyDispatchConfig(t, policyURL, relaxedPolicyActions())
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	anchor := "            package_sha256: " + policyTestPackage + "\n"
	if !strings.Contains(string(raw), anchor) {
		t.Fatal("the policy e2e config no longer carries the package anchor")
	}
	members := "            decisions_v2:\n              cost_weight: 0.5\n" +
		"              tier_points:\n                - name: swift\n                  cost_weight: 0.8\n"
	rendered := strings.Replace(string(raw), anchor, anchor+members, 1)
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
