package extproc

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A request that arrives while the selector is still warming up waits for
// readiness rather than being refused not_ready at once. Run 16 on lab tag
// pf-ea8ae435-5aa4ec, 2026-10-07: a cold instance's first probe of a
// scaled-to-zero policy app took 2.5 min, and 38 requests in that window
// failed with a 503 the moment they arrived.

func TestAwaitArmedReturnsOnceReadinessLands(t *testing.T) {
	selector := newRaylineARCSelector(nil, nil, nil, "rev")
	selector.readinessWait = 5 * time.Second
	components := &raylineARCArmedComponents{}
	go func() {
		time.Sleep(50 * time.Millisecond)
		selector.arm(components)
	}()
	started := time.Now()
	if got := selector.awaitArmed(context.Background()); got != components {
		t.Fatalf("awaitArmed = %v, want the armed components", got)
	}
	if waited := time.Since(started); waited > 2*time.Second {
		t.Fatalf("waited %s: the wait did not end when readiness landed", waited)
	}
}

func TestAwaitArmedGivesUpAtItsBound(t *testing.T) {
	selector := newRaylineARCSelector(nil, nil, nil, "rev")
	selector.readinessWait = 50 * time.Millisecond
	started := time.Now()
	if got := selector.awaitArmed(context.Background()); got != nil {
		t.Fatalf("awaitArmed = %v on a selector that never armed", got)
	}
	if waited := time.Since(started); waited < 40*time.Millisecond || waited > 2*time.Second {
		t.Fatalf("waited %s, want about the 50 ms bound", waited)
	}
}

func TestAwaitArmedDoesNotWaitWhenItCannotHelp(t *testing.T) {
	cases := map[string]func(*raylineARCSelector) context.Context{
		"wait off": func(s *raylineARCSelector) context.Context { s.readinessWait = 0; return context.Background() },
		"unrecoverable": func(s *raylineARCSelector) context.Context {
			s.readinessWait, s.unrecoverable = time.Minute, true
			return context.Background()
		},
		"request canceled": func(s *raylineARCSelector) context.Context {
			s.readinessWait = time.Minute
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx
		},
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			selector := newRaylineARCSelector(nil, nil, nil, "rev")
			ctx := setup(selector)
			started := time.Now()
			if got := selector.awaitArmed(ctx); got != nil {
				t.Fatalf("awaitArmed = %v", got)
			}
			if waited := time.Since(started); waited > time.Second {
				t.Fatalf("waited %s where waiting cannot help", waited)
			}
		})
	}
}

func TestReadinessWaitDefaultsAndBounds(t *testing.T) {
	cases := []struct {
		seconds int
		want    time.Duration
	}{{0, 30 * time.Second}, {-1, 0}, {90, 90 * time.Second}, {300, 300 * time.Second}}
	for _, c := range cases {
		cfg := &config.RaylineARCAlgorithmConfig{ReadinessWaitSeconds: c.seconds}
		if got := cfg.ReadinessWait(); got != c.want {
			t.Fatalf("readiness_wait_seconds %d = %s, want %s", c.seconds, got, c.want)
		}
	}
}

// End to end: a policy-service cell whose service is still loading its
// package holds the first request until the service is ready, and serves it.
func TestARequestWaitsForAColdPolicyServiceAndIsServed(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	catalog := make([]string, 0, len(actions))
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	fake.cold.Store(true)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["think"].ActionID })
	path := writePolicyDispatchConfig(t, fake.URL(), actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	waiting := strings.Replace(string(raw), "readiness_wait_seconds: -1", "readiness_wait_seconds: 30", 1)
	if waiting == string(raw) {
		t.Fatal("the policy e2e config no longer turns the readiness wait off")
	}
	if err = os.WriteFile(path, []byte(waiting), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	// The service finishes loading a moment after the request arrives; the
	// readiness probe retries every 5 s by default.
	go func() {
		time.Sleep(200 * time.Millisecond)
		fake.cold.Store(false)
	}()
	status, err := policyTurnStatus(router, "cold-episode", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if status != 0 {
		t.Fatalf("the request was refused with %d while the service warmed; want it served", status)
	}
}

// An omitted readiness wait and the default spelled out are one setting, so
// two decisions that differ only in how they spell it share a selector; a
// different wait still conflicts.
func TestOmittedReadinessWaitIsTheSameSelectionConfigAsTheDefault(t *testing.T) {
	omitted := *raylineARCAlgorithmConfigForTest().RaylineARC
	spelled := omitted
	spelled.ReadinessWaitSeconds = config.DefaultRaylineARCReadinessWaitSeconds
	if !sameRaylineARCSelectionConfig(&omitted, &spelled) {
		t.Fatal("an omitted and a spelled-out default readiness wait read as conflicting selector configs")
	}
	off, alsoOff := omitted, omitted
	off.ReadinessWaitSeconds, alsoOff.ReadinessWaitSeconds = -1, -5
	if !sameRaylineARCSelectionConfig(&off, &alsoOff) {
		t.Fatal("two negative readiness waits read as different settings")
	}
	longer := omitted
	longer.ReadinessWaitSeconds = 90
	if sameRaylineARCSelectionConfig(&omitted, &longer) {
		t.Fatal("different readiness waits read as one selector config")
	}
}
