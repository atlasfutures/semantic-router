package extproc

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func relaxedRouter(t *testing.T) (*OpenAIRouter, *config.AlgorithmConfig) {
	t.Helper()
	router, _, algorithm := missingSessionRequestContext(t, "")
	algorithm.RaylineARC.Episode.Consistency = config.RaylineARCConsistencyRelaxed
	return router, algorithm
}

func relaxedTurn(t *testing.T, router *OpenAIRouter, algorithm *config.AlgorithmConfig, body string, arm int) *RequestContext {
	t.Helper()
	ctx := coalesceRequestContext(body)
	arc := router.buildRaylineARCSelectionContext(algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
	if arc.PreparationFailure != "" {
		t.Fatalf("relaxed turn %s failure = %q", body, arc.PreparationFailure)
	}
	if !ctx.RaylineARCTransaction.relaxed {
		t.Fatal("a relaxed cell must prepare a relaxed transaction")
	}
	ctx.RaylineARCTransaction.markSelection(arm, 11)
	return ctx
}

func relaxedDrops(class string) float64 {
	return testutil.ToFloat64(metrics.RaylineARCEpisodeTransactions.WithLabelValues("relaxed_dropped", class))
}

// Two different turns on one relaxed episode never wait on each other and
// never fail: both prepare at once, the first commit lands, and the second is
// dropped as a conflict and counted, while its request still succeeds.
func TestRelaxedTurnsNeverWaitOrFail(t *testing.T) {
	router, algorithm := relaxedRouter(t)
	start := time.Now()
	first := relaxedTurn(t, router, algorithm, `{"turn":"a"}`, 0)
	second := relaxedTurn(t, router, algorithm, `{"turn":"b"}`, 1)
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Fatalf("relaxed prepares took %v: a relaxed turn must not wait for a lease", elapsed)
	}
	conflicts := relaxedDrops("conflict")
	if err := first.RaylineARCTransaction.commit(context.Background(), first); err != nil {
		t.Fatalf("first relaxed commit = %v", err)
	}
	if err := second.RaylineARCTransaction.commit(context.Background(), second); err != nil {
		t.Fatalf("a lost relaxed race must not fail the request: %v", err)
	}
	if got := relaxedDrops("conflict") - conflicts; got != 1 {
		t.Fatalf("conflicts counted = %v, want 1", got)
	}
	probe := relaxedTurn(t, router, algorithm, `{"turn":"probe"}`, 0)
	if probe.RaylineARCTransaction.state.TurnIndex != 1 {
		t.Fatalf("turn index = %d, want 1: one commit landed", probe.RaylineARCTransaction.state.TurnIndex)
	}
}

// failingSnapshotStore is a relaxed store whose reads fail.
type failingSnapshotStore struct {
	*raylinearc.MemoryEpisodeStore
}

func (failingSnapshotStore) Snapshot(context.Context, string, int) (*raylinearc.EpisodeState, raylinearc.EpisodeReadToken, error) {
	return nil, raylinearc.EpisodeReadToken{}, errors.New("store unavailable")
}

// A relaxed read that fails decides from a fresh state, as a first turn would,
// and commits nothing; the request is still served.
func TestRelaxedReadFailureDecidesStatelessly(t *testing.T) {
	router, algorithm := relaxedRouter(t)
	memory := router.RaylineARCEpisodeStore.(*raylinearc.MemoryEpisodeStore)
	router.RaylineARCEpisodeStore = failingSnapshotStore{memory}
	ctx := relaxedTurn(t, router, algorithm, `{"turn":"a"}`, 1)
	if !ctx.RaylineARCTransaction.stateless || ctx.RaylineARCTransaction.state.TurnIndex != 0 {
		t.Fatal("a failed relaxed read must decide from a fresh state")
	}
	stateless := relaxedDrops("stateless")
	if err := ctx.RaylineARCTransaction.commit(context.Background(), ctx); err != nil {
		t.Fatalf("stateless relaxed commit = %v", err)
	}
	if got := relaxedDrops("stateless") - stateless; got != 1 {
		t.Fatalf("stateless drops counted = %v, want 1", got)
	}
	router.RaylineARCEpisodeStore = memory
	probe := relaxedTurn(t, router, algorithm, `{"turn":"probe"}`, 0)
	if probe.RaylineARCTransaction.state.TurnIndex != 0 {
		t.Fatal("a stateless turn must commit nothing")
	}
}

// Strict cells are unchanged: a second turn still contends for the lease.
func TestStrictCellStillContends(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	first := coalesceRequestContext(`{"turn":"a"}`)
	if failure := router.buildRaylineARCSelectionContext(algorithm, first, missingSessionModelRefs(), raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("strict leader failure = %q", failure)
	}
	other := router.buildRaylineARCSelectionContext(algorithm, coalesceRequestContext(`{"turn":"b"}`), missingSessionModelRefs(), raylineARCEpisodeRequired)
	if other.PreparationFailure != "episode_timeout" {
		t.Fatalf("strict second turn failure = %q, want episode_timeout", other.PreparationFailure)
	}
	_ = first.RaylineARCTransaction.abort(context.Background(), "test")
}

// Omitted and "strict" are one setting, so two decisions that spell it
// differently share a selector; strict and relaxed still conflict.
func TestOmittedConsistencyIsTheSameSelectionConfigAsStrict(t *testing.T) {
	omitted := *raylineARCAlgorithmConfigForTest().RaylineARC
	strict := omitted
	strict.Episode.Consistency = config.RaylineARCConsistencyStrict
	if !sameRaylineARCSelectionConfig(&omitted, &strict) {
		t.Fatal("omitted and strict consistency read as conflicting selector configs")
	}
	relaxed := omitted
	relaxed.Episode.Consistency = config.RaylineARCConsistencyRelaxed
	if sameRaylineARCSelectionConfig(&strict, &relaxed) {
		t.Fatal("strict and relaxed consistency read as one selector config")
	}
}
