package extproc

import (
	"context"
	"errors"
	"io"
	"net"
	"os"
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

func unreachableRedisEpisodeConfig(consistency string) config.RaylineARCEpisodeConfig {
	return config.RaylineARCEpisodeConfig{
		IDHeader:              testEpisodeIDHeader,
		Backend:               config.RaylineARCBackendRedis,
		KeyPrefix:             "test:rayline-arc-unready:",
		AcquireTimeoutSeconds: 1,
		LeaseTTLSeconds:       60,
		IdleTTLSeconds:        900,
		Consistency:           consistency,
		// Nothing listens on the discard port.
		Redis: config.RaylineARCRedisConfig{Address: "127.0.0.1:9"},
	}
}

// A strict cell whose Redis is down at startup keeps its client, reported not
// ready, with every capability a strict turn asserts, so it serves once the
// store answers instead of staying disarmed for the life of the process. Its
// turns fail closed meanwhile: a lease attempt fails as a store failure.
func TestStrictCellKeepsItsStoreWhenItIsDownAtStartup(t *testing.T) {
	store, closeStore, err := createRaylineARCEpisodeStore(unreachableRedisEpisodeConfig(config.RaylineARCConsistencyStrict))
	if err != nil || store == nil {
		t.Fatalf("strict cell store = %v, err %v; want the store kept", store, err)
	}
	t.Cleanup(func() { _ = closeStore() })
	if raylineARCEpisodeStoreReady(store) {
		t.Fatal("a store that did not answer at startup is reported ready")
	}
	if _, ok := store.(raylinearc.EpisodeLeaseRenewer); !ok {
		t.Fatal("the kept strict store lost lease renewal")
	}
	if _, ok := store.(raylinearc.EpisodeForkRegistry); !ok {
		t.Fatal("the kept strict store lost the fork registry")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, _, err = store.Prepare(ctx, strictTestEpisodeIDHash, 2)
	// go-redis retries the dial until the deadline, so the failure surfaces as
	// the store timing out, never as contention.
	if err == nil || !isStoreFailureClass(boundedARCPrepareFailure(err)) {
		t.Fatalf("Prepare with the store down = %v, want a store failure", err)
	}
}

// strictTestEpisodeIDHash is a well-formed episode hash, so a failing lease
// attempt fails on the store, not on hash validation.
const strictTestEpisodeIDHash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

// The kept strict store recovers by itself: a cell that booted while its
// Redis was unreachable fails its lease attempts, then wins one once the same
// address answers, with no restart. Needs a real Redis
// (RAYLINE_ARC_TEST_REDIS_ADDR); the address the cell boots against is a local
// port that starts forwarding to it only after the store was built.
func TestStrictCellStoreDownAtStartupServesOnceRedisAnswers(t *testing.T) {
	target := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if target == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	reserved, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := reserved.Addr().String()
	_ = reserved.Close()

	episodeConfig := unreachableRedisEpisodeConfig(config.RaylineARCConsistencyStrict)
	episodeConfig.Redis.Address = address
	episodeConfig.KeyPrefix = "test:rayline-arc-strict-recovery:" + time.Now().Format("150405.000000000") + ":"
	store, closeStore, err := createRaylineARCEpisodeStore(episodeConfig)
	if err != nil || raylineARCEpisodeStoreReady(store) {
		t.Fatalf("strict store with Redis down = %v (ready %v), err %v; want kept and not ready",
			store, raylineARCEpisodeStoreReady(store), err)
	}
	t.Cleanup(func() { _ = closeStore() })
	down, cancelDown := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelDown()
	if _, _, err := store.Prepare(down, strictTestEpisodeIDHash, 2); err == nil {
		t.Fatal("a strict lease was won with Redis down")
	}

	// Redis comes back at the address the cell was configured with.
	listener, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatalf("relisten on %s: %v", address, err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go forwardTCP(listener, target)

	// go-redis stops dialing after a run of dial errors and probes in the
	// background about once a second, so the first attempts may still fail.
	var lease raylinearc.Lease
	for deadline := time.Now().Add(10 * time.Second); ; time.Sleep(200 * time.Millisecond) {
		attempt, cancelAttempt := context.WithTimeout(context.Background(), 2*time.Second)
		lease, _, err = store.Prepare(attempt, strictTestEpisodeIDHash, 2)
		cancelAttempt()
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("Prepare once Redis answers = %v, want the lease", err)
		}
	}
	up, cancelUp := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelUp()
	if err := store.Abort(up, lease); err != nil {
		t.Fatalf("Abort = %v", err)
	}
}

// forwardTCP proxies every accepted connection to target until the listener
// closes.
func forwardTCP(listener net.Listener, target string) {
	for {
		client, err := listener.Accept()
		if err != nil {
			return
		}
		go func() {
			defer client.Close()
			server, err := net.Dial("tcp", target)
			if err != nil {
				return
			}
			defer server.Close()
			// The cell closing its side ends both copies.
			go func() { _, _ = io.Copy(server, client); _ = server.Close() }()
			_, _ = io.Copy(client, server)
		}()
	}
}

// Every capability a strict turn asserts on its store survives the wrapper.
var (
	_ raylinearc.EpisodeLeaseRenewer    = unreadyStrictRaylineARCEpisodeStore{}
	_ raylinearc.EpisodeStateStager     = unreadyStrictRaylineARCEpisodeStore{}
	_ raylinearc.EpisodeForkRegistry    = unreadyStrictRaylineARCEpisodeStore{}
	_ raylinearc.EpisodeSnapshotStore   = unreadyStrictRaylineARCEpisodeStore{}
	_ raylinearc.EpisodeStagedTurnStore = unreadyStrictRaylineARCEpisodeStore{}
	_ raylinearc.ServedModelStore       = unreadyStrictRaylineARCEpisodeStore{}
	_ raylinearc.EpisodeStoreReadiness  = unreadyStrictRaylineARCEpisodeStore{}
)

func isStoreFailureClass(class string) bool {
	return class == "episode_store" || class == "episode_store_timeout"
}

// A relaxed cell whose Redis is down at startup keeps its store and serves:
// each turn decides statelessly until the store is back.
func TestRelaxedCellStartsAndServesWithItsStoreDown(t *testing.T) {
	store, closeStore, err := createRaylineARCEpisodeStore(unreachableRedisEpisodeConfig(config.RaylineARCConsistencyRelaxed))
	if err != nil || store == nil {
		t.Fatalf("relaxed cell store = %v, err %v; want the store kept", store, err)
	}
	t.Cleanup(func() { _ = closeStore() })

	if raylineARCEpisodeStoreReady(store) {
		t.Fatal("a store that did not answer at startup is reported ready")
	}

	router, algorithm := relaxedRouter(t)
	memory := router.RaylineARCEpisodeStore
	router.RaylineARCEpisodeStore = store
	ctx := relaxedTurn(t, router, algorithm, `{"turn":"a"}`, 1)
	if !ctx.RaylineARCTransaction.stateless {
		t.Fatal("a relaxed turn with its store down must decide statelessly")
	}
	if err := ctx.RaylineARCTransaction.commit(context.Background(), ctx); err != nil {
		t.Fatalf("stateless commit = %v", err)
	}
	if got := episodeStoreGauge(); got != 0 {
		t.Fatalf("episode_store gauge = %v while reads fail, want 0", got)
	}
	// The store answering again turns the gauge back on.
	router.RaylineARCEpisodeStore = memory
	relaxedTurn(t, router, algorithm, `{"turn":"b"}`, 0)
	if got := episodeStoreGauge(); got != 1 {
		t.Fatalf("episode_store gauge = %v once reads succeed, want 1", got)
	}
}

func episodeStoreGauge() float64 {
	return testutil.ToFloat64(metrics.RaylineARCComponentReady.WithLabelValues("episode_store"))
}
