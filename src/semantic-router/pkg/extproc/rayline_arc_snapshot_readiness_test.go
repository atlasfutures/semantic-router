package extproc

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The first pin starts the one verification and waits for it; pins that
// arrive while it runs share it; a verified snapshot is ready from then on.
func TestSnapshotReadinessVerifiesOnceOnFirstPin(t *testing.T) {
	var calls atomic.Int32
	release := make(chan struct{})
	readiness := newRaylineARCSnapshotReadiness(context.Background(), []string{"b"}, func(context.Context, string) error {
		calls.Add(1)
		<-release
		return nil
	})
	if class := readiness.ensure(context.Background(), "a", time.Second); class != "" {
		t.Fatalf("an ungated (default) snapshot answered %q", class)
	}
	var wg sync.WaitGroup
	classes := make([]string, 4)
	for index := range classes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			classes[index] = readiness.ensure(context.Background(), "b", 5*time.Second)
		}()
	}
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()
	if calls.Load() != 1 || slices.ContainsFunc(classes, func(class string) bool { return class != "" }) {
		t.Fatalf("verifications %d, classes %v", calls.Load(), classes)
	}
	if class := readiness.ensure(context.Background(), "b", 0); class != "" || calls.Load() != 1 {
		t.Fatalf("a verified snapshot answered %q after %d verifications", class, calls.Load())
	}
}

// A verification that outlasts the turn's wait answers loading, and the turn
// after it completes is served without another verification.
func TestSnapshotReadinessAnswersLoadingPastTheWait(t *testing.T) {
	release := make(chan struct{})
	var calls atomic.Int32
	readiness := newRaylineARCSnapshotReadiness(context.Background(), []string{"b"}, func(context.Context, string) error {
		calls.Add(1)
		<-release
		return nil
	})
	if class := readiness.ensure(context.Background(), "b", 20*time.Millisecond); class != arcFailureSnapshotLoading {
		t.Fatalf("class %q, want loading", class)
	}
	close(release)
	if class := readiness.ensure(context.Background(), "b", time.Second); class != "" || calls.Load() != 1 {
		t.Fatalf("class %q after %d verifications", class, calls.Load())
	}
}

// A failed verification refuses the snapshot, and keeps refusing it without
// asking again until the retry window passes; then it verifies again.
func TestSnapshotReadinessRefusesAFailedSnapshotUntilRetry(t *testing.T) {
	var calls atomic.Int32
	fail := atomic.Bool{}
	fail.Store(true)
	readiness := newRaylineARCSnapshotReadiness(context.Background(), []string{"b"}, func(context.Context, string) error {
		calls.Add(1)
		if fail.Load() {
			return &raylinearc.PolicyServiceError{Class: "package_not_loaded"}
		}
		return nil
	})
	now := time.Unix(1_000, 0)
	readiness.now = func() time.Time { return now }
	if class := readiness.ensure(context.Background(), "b", time.Second); class != arcFailureSnapshotUnavailable {
		t.Fatalf("class %q, want unavailable", class)
	}
	if class := readiness.ensure(context.Background(), "b", time.Second); class != arcFailureSnapshotUnavailable || calls.Load() != 1 {
		t.Fatalf("within the retry window: class %q after %d verifications", class, calls.Load())
	}
	now = now.Add(snapshotVerifyRetry)
	fail.Store(false)
	if class := readiness.ensure(context.Background(), "b", time.Second); class != "" || calls.Load() != 2 {
		t.Fatalf("after the retry window: class %q after %d verifications", class, calls.Load())
	}
}

// Through the selector: a pin to a non-default snapshot is verified against
// the service on its first turn and then decided with its own package; a
// service that does not serve it refuses the turn, which never reaches a
// decide (no other package serves in its place).
func TestSnapshotLazyReadinessThroughTheSelector(t *testing.T) {
	cases := map[string]struct {
		serves bool
		want   string
	}{"served": {true, ""}, "not served": {false, arcFailureSnapshotUnavailable}}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			fixture := newSnapshotFixture(t)
			if !tc.serves {
				fixture.fake.mu.Lock()
				fixture.fake.others = nil
				fixture.fake.mu.Unlock()
			}
			armed := fixture.selector.armed.Load()
			cell := armed.scorer.(*policyServiceScorer)
			policy := fixture.decision.Algorithm.RaylineARC.PolicyService
			cell.lazy = newRaylineARCSnapshotReadiness(context.Background(), lazySnapshotIDs(policy), func(ctx context.Context, id string) error {
				return probeRaylineARCSnapshot(ctx, armed.policy, policy, cell, id)
			})
			fixture.selector.readinessWait = 5 * time.Second
			state, _ := raylinearc.NewEpisodeState(2)
			_, err := fixture.selectUnder(t, snapshotSecond, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
			var failure *raylineARCSelectionFailure
			switch {
			case tc.want == "" && err != nil:
				t.Fatalf("select: %v", err)
			case tc.want == "":
				if sent := fixture.fake.received(); len(sent) != 1 || sent[0].Package.Alias != snapshotSecondAlias {
					t.Fatalf("decided with %+v", sent)
				}
			case !errors.As(err, &failure) || failure.class != tc.want:
				t.Fatalf("err = %v, want %s", err, tc.want)
			case len(fixture.fake.received()) != 0:
				t.Fatalf("a refused snapshot reached decide: %+v", fixture.fake.received())
			}
		})
	}
}

// Loading answers 503 with Retry-After and the public class not_ready; a
// failed verification answers 503 unavailable.
func TestSnapshotReadinessResponses(t *testing.T) {
	router := &OpenAIRouter{}
	loading := router.authoritativeSelectionFailureResponse(
		&modelSelectionFailure{algorithm: "rayline_arc", class: arcFailureSnapshotLoading}, &RequestContext{})
	if loading == nil || loading.GetImmediateResponse().GetStatus().GetCode() != http.StatusServiceUnavailable ||
		immediateHeaderValue(loading, "retry-after") == "" || immediateHeaderValue(loading, selectionFailureHeader) != selectionFailureNotReady {
		t.Fatalf("loading response %v", loading)
	}
	failed := router.authoritativeSelectionFailureResponse(
		&modelSelectionFailure{algorithm: "rayline_arc", class: arcFailureSnapshotUnavailable}, &RequestContext{})
	if failed == nil || failed.GetImmediateResponse().GetStatus().GetCode() != http.StatusServiceUnavailable ||
		immediateHeaderValue(failed, selectionFailureHeader) != selectionFailureUnavailable {
		t.Fatalf("failed response %v", failed)
	}
}
