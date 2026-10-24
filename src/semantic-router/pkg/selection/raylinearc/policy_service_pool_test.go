package raylinearc

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fakePolicyReplica answers decide with the v1 fixture, or with status and a
// contract error body while status is non-zero.
type fakePolicyReplica struct {
	server *httptest.Server
	calls  atomic.Int32
	status atomic.Int32
	sleep  atomic.Int64
	// stale answers for another package while set.
	stale atomic.Bool
}

func newFakePolicyReplica(t *testing.T) *fakePolicyReplica {
	t.Helper()
	body := readPolicyFixture(t, "decision_response.v1.json")
	staleBody := []byte(strings.Replace(string(body), fixturePackage(t).PackageSHA256, strings.Repeat("f", 64), 1))
	replica := &fakePolicyReplica{}
	replica.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		replica.calls.Add(1)
		if sleep := time.Duration(replica.sleep.Load()); sleep > 0 {
			time.Sleep(sleep)
		}
		if status := int(replica.status.Load()); status != 0 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":"backend_unavailable","detail":{}}`))
			return
		}
		if replica.stale.Load() {
			_, _ = w.Write(staleBody)
			return
		}
		_, _ = w.Write(body)
	}))
	t.Cleanup(replica.server.Close)
	return replica
}

func testPolicyPool(t *testing.T, states []EncoderReplicaState, timeout time.Duration) (*PolicyServicePool, []*fakePolicyReplica) {
	t.Helper()
	fakes := make([]*fakePolicyReplica, len(states))
	replicas := make([]PolicyServiceReplica, len(states))
	for index, state := range states {
		fakes[index] = newFakePolicyReplica(t)
		replicas[index] = PolicyServiceReplica{
			ID:     string(rune('a'+index)) + "-replica",
			State:  state,
			Client: NewPolicyServiceClient(PolicyServiceConfig{BaseURL: fakes[index].server.URL, TotalTimeout: timeout}),
		}
	}
	pool, err := NewPolicyServicePool(replicas, PolicyServicePoolConfig{
		SchemaVersion:          PolicyFailoverSchemaV1,
		UnavailableStatusCodes: []int{http.StatusServiceUnavailable},
		UnavailableCooldown:    time.Minute,
		MaxRemaps:              1,
	})
	if err != nil {
		t.Fatal(err)
	}
	return pool, fakes
}

// fixturePackage is the package the decide fixture answers for, which the
// test requests name.
func fixturePackage(t *testing.T) PolicyPackageRef {
	t.Helper()
	response, err := DecodePolicyDecisionResponse(readPolicyFixture(t, "decision_response.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	return response.Package
}

func episodeHash(seed string) string {
	return strings.Repeat(seed, 64)[:64]
}

func decideOnce(t *testing.T, pool *PolicyServicePool, episode string, affinity EncoderAffinity) (EncoderResult, error) {
	t.Helper()
	binding := pool.Bind(episode, affinity)
	_, err := binding.Decide(context.Background(), PolicyDecisionRequest{SchemaVersion: PolicyDecisionRequestSchema, EpisodeIDHash: episode, Package: fixturePackage(t)})
	var routed EncoderResult
	binding.Annotate(&routed)
	return routed, err
}

// A new episode lands on the same replica every time, and placements spread
// across the active replicas.
func TestPolicyPoolRendezvousIsStableAndSpreads(t *testing.T) {
	pool, _ := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive, EncoderReplicaActive, EncoderReplicaActive}, time.Second)
	owners := map[string]int{}
	for _, seed := range []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"} {
		first, err := decideOnce(t, pool, episodeHash(seed), EncoderAffinity{})
		if err != nil {
			t.Fatal(err)
		}
		second, _ := decideOnce(t, pool, episodeHash(seed), EncoderAffinity{})
		if first.ReplicaID != second.ReplicaID || first.ReplicaFailover {
			t.Fatalf("episode %s placed %q then %q (failover %t)", seed, first.ReplicaID, second.ReplicaID, first.ReplicaFailover)
		}
		owners[first.ReplicaID]++
	}
	if len(owners) < 2 {
		t.Fatalf("16 episodes all placed on %v", owners)
	}
}

// A persisted owner is kept over the rendezvous choice, draining or not, and
// adding a replica moves no episode that already has one.
func TestPolicyPoolHonoursThePersistedOwner(t *testing.T) {
	pool, fakes := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive, EncoderReplicaDraining}, time.Second)
	routed, err := decideOnce(t, pool, episodeHash("1"), EncoderAffinity{Owner: "b-replica", Visited: []string{"b-replica"}})
	if err != nil {
		t.Fatal(err)
	}
	if routed.ReplicaID != "b-replica" || routed.ReplicaFailover || fakes[1].calls.Load() != 1 {
		t.Fatalf("routed %+v, draining calls %d", routed, fakes[1].calls.Load())
	}
	// A new episode never lands on the draining replica.
	for _, seed := range []string{"2", "3", "4", "5", "6", "7"} {
		fresh, _ := decideOnce(t, pool, episodeHash(seed), EncoderAffinity{})
		if fresh.ReplicaID != "a-replica" {
			t.Fatalf("new episode placed on %q", fresh.ReplicaID)
		}
	}
	// An owner no longer configured is placed afresh and recorded as a failover.
	gone, _ := decideOnce(t, pool, episodeHash("8"), EncoderAffinity{Owner: "z-replica", Visited: []string{"z-replica"}})
	if gone.ReplicaID != "a-replica" || !gone.ReplicaFailover || len(gone.VisitedReplicaIDs) != 1 {
		t.Fatalf("removed owner routed %+v", gone)
	}
}

// One configured unavailable status moves the episode once and cools the
// replica down; the new owner is reported for the commit.
func TestPolicyPoolRemapsOnceOnAConfiguredStatus(t *testing.T) {
	pool, fakes := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive, EncoderReplicaActive}, time.Second)
	episode := episodeHash("1")
	placed, _ := decideOnce(t, pool, episode, EncoderAffinity{})
	ownerIndex := placed.ReplicaIndex
	fakes[ownerIndex].status.Store(http.StatusServiceUnavailable)
	moved, err := decideOnce(t, pool, episode, EncoderAffinity{Owner: placed.ReplicaID, Visited: []string{placed.ReplicaID}})
	if err != nil {
		t.Fatal(err)
	}
	if moved.ReplicaID == placed.ReplicaID || !moved.ReplicaFailover || moved.ReplicaAttempts != 2 || len(moved.VisitedReplicaIDs) != 2 {
		t.Fatalf("remap routed %+v", moved)
	}
	// While it cools down, the old owner is skipped without a call.
	before := fakes[ownerIndex].calls.Load()
	again, _ := decideOnce(t, pool, episode, EncoderAffinity{Owner: placed.ReplicaID, Visited: []string{placed.ReplicaID}})
	if again.ReplicaID != moved.ReplicaID || fakes[ownerIndex].calls.Load() != before {
		t.Fatalf("cooling owner called again: %+v", again)
	}
	// After the cooldown the persisted owner is used again.
	pool.now = func() time.Time { return time.Now().Add(2 * time.Minute) }
	fakes[ownerIndex].status.Store(0)
	back, _ := decideOnce(t, pool, episode, EncoderAffinity{Owner: placed.ReplicaID, Visited: []string{placed.ReplicaID}})
	if back.ReplicaID != placed.ReplicaID {
		t.Fatalf("recovered owner not used: %+v", back)
	}
}

// Both replicas unavailable: one remap, then the failure stands.
func TestPolicyPoolRemapsAtMostOnce(t *testing.T) {
	pool, fakes := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive, EncoderReplicaActive}, time.Second)
	fakes[0].status.Store(http.StatusServiceUnavailable)
	fakes[1].status.Store(http.StatusServiceUnavailable)
	_, err := decideOnce(t, pool, episodeHash("1"), EncoderAffinity{})
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Status != http.StatusServiceUnavailable {
		t.Fatalf("err = %v", err)
	}
	if total := fakes[0].calls.Load() + fakes[1].calls.Load(); total != 2 {
		t.Fatalf("calls = %d, want 2", total)
	}
}

// A status outside the configured set, session_busy included, and a timeout
// never move the episode: the replica may hold it in flight.
func TestPolicyPoolFailsClosedOnOtherFailures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		sleep  time.Duration
	}{
		{"session_busy", http.StatusConflict, 0},
		{"server_error", http.StatusInternalServerError, 0},
		{"timeout", 0, 300 * time.Millisecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool, fakes := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive, EncoderReplicaActive}, 100*time.Millisecond)
			episode := episodeHash("1")
			placed, _ := decideOnce(t, pool, episode, EncoderAffinity{})
			owner := fakes[placed.ReplicaIndex]
			other := fakes[1-placed.ReplicaIndex]
			owner.status.Store(int32(tc.status))
			owner.sleep.Store(int64(tc.sleep))
			otherCalls := other.calls.Load()
			binding := pool.Bind(episode, EncoderAffinity{Owner: placed.ReplicaID, Visited: []string{placed.ReplicaID}})
			if _, err := binding.Decide(context.Background(), PolicyDecisionRequest{SchemaVersion: PolicyDecisionRequestSchema, Package: fixturePackage(t)}); err == nil {
				t.Fatal("decide succeeded")
			}
			if other.calls.Load() != otherCalls {
				t.Fatal("failure moved the episode")
			}
			// A retry in the same turn stays on the owner.
			owner.status.Store(0)
			owner.sleep.Store(0)
			if _, err := binding.Decide(context.Background(), PolicyDecisionRequest{SchemaVersion: PolicyDecisionRequestSchema, Package: fixturePackage(t)}); err != nil {
				t.Fatal(err)
			}
			var routed EncoderResult
			binding.Annotate(&routed)
			if routed.ReplicaID != placed.ReplicaID || routed.ReplicaFailover {
				t.Fatalf("retry routed %+v", routed)
			}
		})
	}
}

// Readiness is met by one serving active replica; one that fails cools down
// and takes no new episode.
func TestPolicyPoolProbe(t *testing.T) {
	pool, _ := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive, EncoderReplicaActive}, time.Second)
	broken := pool.replicas[0].Client
	err := pool.Probe(context.Background(), func(_ context.Context, client *PolicyServiceClient) error {
		if client == broken {
			return &PolicyServiceError{Class: "package_not_loaded", Status: http.StatusNotFound}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("probe = %v", err)
	}
	for _, seed := range []string{"1", "2", "3", "4", "5", "6"} {
		routed, _ := decideOnce(t, pool, episodeHash(seed), EncoderAffinity{})
		if routed.ReplicaID != "b-replica" {
			t.Fatalf("episode placed on the replica that failed its probe")
		}
	}
	// The cooldown ending does not return it: only a passing re-check does.
	pool.now = func() time.Time { return time.Now().Add(time.Hour) }
	if routed, _ := decideOnce(t, pool, episodeHash("1"), EncoderAffinity{Owner: "a-replica", Visited: []string{"a-replica"}}); routed.ReplicaID != "b-replica" {
		t.Fatalf("unready owner used after its cooldown: %+v", routed)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		pool.Reprobe(ctx, func(context.Context, *PolicyServiceClient) error { return nil }, time.Millisecond)
		close(done)
	}()
	deadline := time.Now().Add(2 * time.Second)
	for !pool.isReady("a-replica") && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	cancel()
	<-done
	if routed, _ := decideOnce(t, pool, episodeHash("1"), EncoderAffinity{Owner: "a-replica", Visited: []string{"a-replica"}}); routed.ReplicaID != "a-replica" {
		t.Fatalf("re-checked owner not used: %+v", routed)
	}
	all := pool.Probe(context.Background(), func(context.Context, *PolicyServiceClient) error {
		return &PolicyServiceError{Class: "transport"}
	})
	if all == nil {
		t.Fatal("probe passed with no serving replica")
	}
}

func TestPolicyPoolRefusesAnUnsafeContract(t *testing.T) {
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: "http://a", TotalTimeout: time.Second})
	valid := PolicyServicePoolConfig{SchemaVersion: PolicyFailoverSchemaV1, UnavailableStatusCodes: []int{503}, UnavailableCooldown: time.Second, MaxRemaps: 1}
	replica := PolicyServiceReplica{ID: "a", State: EncoderReplicaActive, Client: client}
	for name, config := range map[string]PolicyServicePoolConfig{
		"session_busy":   {SchemaVersion: PolicyFailoverSchemaV1, UnavailableStatusCodes: []int{409}, UnavailableCooldown: time.Second, MaxRemaps: 1},
		"gateway":        {SchemaVersion: PolicyFailoverSchemaV1, UnavailableStatusCodes: []int{504}, UnavailableCooldown: time.Second, MaxRemaps: 1},
		"schema":         {SchemaVersion: "x", UnavailableStatusCodes: []int{503}, UnavailableCooldown: time.Second, MaxRemaps: 1},
		"two_remaps":     {SchemaVersion: PolicyFailoverSchemaV1, UnavailableStatusCodes: []int{503}, UnavailableCooldown: time.Second, MaxRemaps: 2},
		"no_cooldown":    {SchemaVersion: PolicyFailoverSchemaV1, UnavailableStatusCodes: []int{503}, MaxRemaps: 1},
		"empty_statuses": {SchemaVersion: PolicyFailoverSchemaV1, UnavailableCooldown: time.Second, MaxRemaps: 1},
	} {
		if _, err := NewPolicyServicePool([]PolicyServiceReplica{replica}, config); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := NewPolicyServicePool([]PolicyServiceReplica{{ID: "a", State: EncoderReplicaDraining, Client: client}}, valid); err == nil {
		t.Error("accepted a set with no active replica")
	}
	if _, err := NewPolicyServicePool([]PolicyServiceReplica{replica, {ID: "a", State: EncoderReplicaActive, Client: NewPolicyServiceClient(PolicyServiceConfig{BaseURL: "http://b"})}}, valid); err == nil {
		t.Error("accepted a duplicated id")
	}
	if _, err := NewPolicyServicePool([]PolicyServiceReplica{replica}, valid); err != nil {
		t.Errorf("valid pool refused: %v", err)
	}
}

// An owner answering for another package (a replica still on the previous
// build during a rollover) cools down, and the episode moves once. With no
// replica left to move to, the stale answer is returned for the caller to
// refuse.
func TestPolicyPoolMovesOffAStaleOwner(t *testing.T) {
	pool, fakes := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive, EncoderReplicaActive}, time.Second)
	episode := episodeHash("1")
	placed, _ := decideOnce(t, pool, episode, EncoderAffinity{})
	fakes[placed.ReplicaIndex].stale.Store(true)
	moved, err := decideOnce(t, pool, episode, EncoderAffinity{Owner: placed.ReplicaID, Visited: []string{placed.ReplicaID}})
	if err != nil {
		t.Fatal(err)
	}
	if moved.ReplicaID == placed.ReplicaID || !moved.ReplicaFailover || moved.ReplicaAttempts != 2 {
		t.Fatalf("stale owner kept: %+v", moved)
	}

	single, singleFakes := testPolicyPool(t, []EncoderReplicaState{EncoderReplicaActive}, time.Second)
	singleFakes[0].stale.Store(true)
	binding := single.Bind(episode, EncoderAffinity{})
	response, err := binding.Decide(context.Background(), PolicyDecisionRequest{SchemaVersion: PolicyDecisionRequestSchema, Package: fixturePackage(t)})
	if err != nil || response == nil || response.Package == fixturePackage(t) {
		t.Fatalf("lone stale replica: response %+v err %v", response, err)
	}
}
