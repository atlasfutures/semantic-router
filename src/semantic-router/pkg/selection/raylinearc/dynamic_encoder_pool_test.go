/*
Copyright 2026 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

//nolint:cyclop // This contract test intentionally follows the full drain lifecycle.
func TestDynamicEncoderPoolAdoptsDrainingAndRetainsRemovedCloser(t *testing.T) {
	var closeStatus atomic.Int32
	serverA, callsA, closesA := newCloseAwareEncoderReplicaTestServer(t, &closeStatus)
	defer serverA.Close()
	serverB, callsB, _ := newCloseAwareEncoderReplicaTestServer(t, &closeStatus)
	defer serverB.Close()
	serverC, callsC, _ := newCloseAwareEncoderReplicaTestServer(t, &closeStatus)
	defer serverC.Close()
	initial := dynamicMembershipSnapshot(1, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: serverA.URL, State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
		{ID: "replica-c", BaseURL: serverC.URL, State: EncoderReplicaActive},
	})
	source := &memoryEncoderMembershipSource{snapshot: initial}
	pool := newDynamicEncoderTestPool(t, source)
	defer pool.Close()

	turns := []Turn{{Role: "user", Text: "public dynamic membership turn"}}
	if _, err := pool.EncodeWithAffinity(
		context.Background(),
		HashEpisodeID("draining-owner"),
		turns,
		EncoderAffinity{Owner: "replica-a", Visited: []string{"replica-a"}},
	); err != nil {
		t.Fatal(err)
	}
	started := time.Now().UTC().Add(-time.Minute)
	source.set(dynamicMembershipSnapshot(2, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: serverA.URL, State: EncoderReplicaDraining, DrainStartedAt: &started},
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
		{ID: "replica-c", BaseURL: serverC.URL, State: EncoderReplicaActive},
	}))
	if refreshErr := pool.Refresh(context.Background()); refreshErr != nil {
		t.Fatal(refreshErr)
	}
	if _, err := pool.EncodeWithAffinity(
		context.Background(),
		HashEpisodeID("draining-owner"),
		turns,
		EncoderAffinity{Owner: "replica-a", Visited: []string{"replica-a"}},
	); err != nil {
		t.Fatal(err)
	}
	newEpisode := HashEpisodeID("new-after-drain")
	if rendezvousWinner(newEpisode, []string{"replica-a", "replica-b", "replica-c"}) != "replica-a" {
		newEpisode = episodeForDynamicReplica(t, "replica-a")
	}
	result, err := pool.Encode(context.Background(), newEpisode, turns)
	if err != nil {
		t.Fatal(err)
	}
	if result.ReplicaID == "replica-a" {
		t.Fatalf("new episode selected draining replica-a")
	}

	source.set(dynamicMembershipSnapshot(3, []EncoderMembershipReplica{
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
		{ID: "replica-c", BaseURL: serverC.URL, State: EncoderReplicaActive},
	}))
	if refreshErr := pool.Refresh(context.Background()); refreshErr != nil {
		t.Fatal(refreshErr)
	}
	report, err := pool.CloseSession(
		context.Background(),
		HashEpisodeID("draining-owner"),
		[]string{"replica-a"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if report.Closed != 1 || closesA.Load() != 1 {
		t.Fatalf("close report = %#v closes-a=%d, want one removed-owner close", report, closesA.Load())
	}
	if callsA.Load() != 2 || callsB.Load()+callsC.Load() != 1 {
		t.Fatalf("post calls a/b/c = %d/%d/%d, want 2/one non-a", callsA.Load(), callsB.Load(), callsC.Load())
	}
}

func TestDynamicEncoderPoolRejectsRevisionSkipWithoutReplacingSnapshot(t *testing.T) {
	serverA, _ := newEncoderReplicaTestServer(t)
	defer serverA.Close()
	serverB, _ := newEncoderReplicaTestServer(t)
	defer serverB.Close()
	source := &memoryEncoderMembershipSource{snapshot: dynamicMembershipSnapshot(1, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: serverA.URL, State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
	})}
	pool := newDynamicEncoderTestPool(t, source)
	defer pool.Close()
	source.set(dynamicMembershipSnapshot(3, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: serverA.URL, State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
	}))
	if err := pool.Refresh(context.Background()); err == nil {
		t.Fatal("Refresh() error = nil, want skipped revision failure")
	}
	membership, err := pool.Membership()
	if membership.Revision != 1 || err == nil {
		t.Fatalf("membership=%#v err=%v, want retained revision one and error", membership, err)
	}
}

//nolint:cyclop // This contract test follows unavailable and recovered capacity stages.
func TestDynamicEncoderPoolProbesNewCapacityBeforeAdoption(t *testing.T) {
	serverA, _, _ := newCloseAwareEncoderReplicaTestServer(t, new(atomic.Int32))
	defer serverA.Close()
	serverB, _, _ := newCloseAwareEncoderReplicaTestServer(t, new(atomic.Int32))
	defer serverB.Close()
	var status atomic.Int32
	status.Store(http.StatusServiceUnavailable)
	serverC := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			switch request.Method {
			case http.MethodPost:
				if code := int(status.Load()); code != 0 && code != http.StatusOK {
					writer.WriteHeader(code)
					return
				}
				writeSessionEncoderResponse(t, writer, nil)
			case http.MethodDelete:
				writer.Header().Set("content-type", "application/json")
				_, _ = writer.Write([]byte(`{"closed":true}`))
			default:
				writer.WriteHeader(http.StatusMethodNotAllowed)
			}
		},
	))
	defer serverC.Close()

	source := &memoryEncoderMembershipSource{snapshot: dynamicMembershipSnapshot(1, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: serverA.URL, State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
	})}
	pool := newDynamicEncoderTestPool(t, source)
	defer pool.Close()
	source.set(dynamicMembershipSnapshot(2, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: serverA.URL, State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
		{ID: "replica-c", BaseURL: serverC.URL, State: EncoderReplicaActive},
	}))
	if err := pool.Refresh(context.Background()); err == nil {
		t.Fatal("Refresh() error = nil for unavailable new capacity")
	}
	current, err := pool.Membership()
	if current.Revision != 1 || err == nil {
		t.Fatalf("membership=%#v err=%v, want retained revision one", current, err)
	}
	status.Store(http.StatusOK)
	if refreshErr := pool.Refresh(context.Background()); refreshErr != nil {
		t.Fatalf("Refresh() after capacity readiness: %v", refreshErr)
	}
	current, err = pool.Membership()
	if err != nil || current.Revision != 2 || len(current.Replicas) != 3 {
		t.Fatalf("membership=%#v err=%v, want adopted revision two", current, err)
	}
}

func TestDynamicEncoderPoolCloseRejectsFurtherRefresh(t *testing.T) {
	serverA, _ := newEncoderReplicaTestServer(t)
	defer serverA.Close()
	serverB, _ := newEncoderReplicaTestServer(t)
	defer serverB.Close()
	source := &memoryEncoderMembershipSource{snapshot: dynamicMembershipSnapshot(1, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: serverA.URL, State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: serverB.URL, State: EncoderReplicaActive},
	})}
	pool := newDynamicEncoderTestPool(t, source)
	pool.Close()
	if err := pool.Refresh(context.Background()); err == nil {
		t.Fatal("Refresh() error = nil after Close()")
	}
}

//nolint:cyclop // This contract test intentionally follows controller drain/reconcile stages.
func TestEncoderMembershipControllerDrainsBeforeRemoval(t *testing.T) {
	now := time.Date(2026, time.August, 3, 12, 0, 0, 0, time.UTC)
	source := &memoryEncoderMembershipSource{snapshot: dynamicMembershipSnapshot(1, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: "http://encoder-a.test:8000", State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: "http://encoder-b.test:8000", State: EncoderReplicaActive},
		{ID: "replica-c", BaseURL: "http://encoder-c.test:8000", State: EncoderReplicaActive},
	})}
	references := &memoryEncoderMembershipReferences{counts: map[string]int{"replica-a": 1}}
	controller, err := NewEncoderMembershipController(EncoderMembershipControllerConfig{
		Source: source, References: references, IdleTTL: time.Minute,
		Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, drainErr := controller.BeginDrain(context.Background(), "replica-a"); drainErr != nil {
		t.Fatal(drainErr)
	}
	if report, reconcileErr := controller.Reconcile(context.Background()); reconcileErr != nil || report.Eligible != 0 {
		t.Fatalf("early reconcile = %#v err=%v, want ineligible drain", report, reconcileErr)
	}
	now = now.Add(time.Minute)
	if report, reconcileErr := controller.Reconcile(context.Background()); reconcileErr != nil || report.Referenced != 1 || report.Removed != 0 {
		t.Fatalf("referenced reconcile = %#v err=%v", report, reconcileErr)
	}
	references.set("replica-a", 0)
	report, err := controller.Reconcile(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report.Removed != 1 || report.Revision != 3 {
		t.Fatalf("drained reconcile = %#v, want one removal at revision three", report)
	}
	current, err := source.Load(context.Background())
	if err != nil || len(current.Replicas) != 2 || membershipReplicaIndex(current, "replica-a") >= 0 {
		t.Fatalf("membership after remove = %#v err=%v", current, err)
	}
}

func TestEncoderMembershipControllerRegistersActiveCapacityIdempotently(t *testing.T) {
	source := &memoryEncoderMembershipSource{snapshot: dynamicMembershipSnapshot(1, []EncoderMembershipReplica{
		{ID: "replica-a", BaseURL: "http://encoder-a.test:8000", State: EncoderReplicaActive},
		{ID: "replica-b", BaseURL: "http://encoder-b.test:8000", State: EncoderReplicaActive},
	})}
	controller, err := NewEncoderMembershipController(EncoderMembershipControllerConfig{
		Source: source, References: &memoryEncoderMembershipReferences{}, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	registered, err := controller.RegisterActive(
		context.Background(),
		"replica-c",
		"http://encoder-c.test:8000/",
	)
	if err != nil {
		t.Fatal(err)
	}
	if registered.Revision != 2 || len(registered.Replicas) != 3 ||
		registered.Replicas[2].State != EncoderReplicaActive {
		t.Fatalf("registered membership = %#v", registered)
	}
	idempotent, err := controller.RegisterActive(
		context.Background(),
		"replica-c",
		"http://encoder-c.test:8000",
	)
	if err != nil || idempotent.Revision != registered.Revision {
		t.Fatalf("idempotent registration = %#v err=%v", idempotent, err)
	}
	if _, err := controller.RegisterActive(
		context.Background(),
		"replica-d",
		"http://encoder-c.test:8000",
	); err == nil {
		t.Fatal("duplicate endpoint registration succeeded")
	}
	if _, err := controller.RegisterActive(
		context.Background(),
		"replica-c",
		"http://replacement.test:8000",
	); err == nil {
		t.Fatal("stable identity endpoint replacement succeeded")
	}
}

func newDynamicEncoderTestPool(
	t *testing.T,
	source EncoderMembershipReader,
) *DynamicEncoderPool {
	t.Helper()
	pool, err := NewDynamicEncoderPool(context.Background(), DynamicEncoderPoolConfig{
		Source: source,
		ClientFactory: func(member EncoderMembershipReplica) (*EncoderClient, error) {
			return newRetainedSessionTestClient(t, member.BaseURL), nil
		},
		PoolConfig: EncoderPoolConfig{
			SchemaVersion:          EncoderFailoverSchemaV1,
			UnavailableStatusCodes: []int{503},
			UnavailableCooldown:    time.Minute,
			MaxRemaps:              1,
		},
		RefreshInterval: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func dynamicMembershipSnapshot(
	revision uint64,
	replicas []EncoderMembershipReplica,
) EncoderMembershipSnapshot {
	return EncoderMembershipSnapshot{
		SchemaVersion: EncoderMembershipSchemaV1,
		Revision:      revision,
		Replicas:      replicas,
	}
}

func episodeForDynamicReplica(t *testing.T, replicaID string) string {
	t.Helper()
	for index := 0; index < 4096; index++ {
		episode := HashEpisodeID("dynamic-membership-" + strconv.Itoa(index))
		if rendezvousWinner(episode, []string{"replica-a", "replica-b", "replica-c"}) == replicaID {
			return episode
		}
	}
	t.Fatalf("could not find episode for %s", replicaID)
	return ""
}

type memoryEncoderMembershipSource struct {
	mu       sync.Mutex
	snapshot EncoderMembershipSnapshot
}

func (source *memoryEncoderMembershipSource) Load(
	context.Context,
) (EncoderMembershipSnapshot, error) {
	source.mu.Lock()
	defer source.mu.Unlock()
	if source.snapshot.Revision == 0 {
		return EncoderMembershipSnapshot{}, ErrEncoderMembershipNotFound
	}
	return cloneEncoderMembershipSnapshot(source.snapshot), nil
}

func (source *memoryEncoderMembershipSource) CompareAndSwap(
	_ context.Context,
	expectedRevision uint64,
	next EncoderMembershipSnapshot,
) error {
	source.mu.Lock()
	defer source.mu.Unlock()
	if source.snapshot.Revision != expectedRevision {
		return ErrEncoderMembershipConflict
	}
	if next.Revision != expectedRevision+1 {
		return errors.New("revision must advance by one")
	}
	source.snapshot = cloneEncoderMembershipSnapshot(next)
	return nil
}

func (source *memoryEncoderMembershipSource) set(snapshot EncoderMembershipSnapshot) {
	source.mu.Lock()
	defer source.mu.Unlock()
	source.snapshot = cloneEncoderMembershipSnapshot(snapshot)
}

type memoryEncoderMembershipReferences struct {
	mu     sync.Mutex
	counts map[string]int
}

func (references *memoryEncoderMembershipReferences) CountEncoderReferences(
	_ context.Context,
	ids []string,
) (map[string]int, error) {
	references.mu.Lock()
	defer references.mu.Unlock()
	result := make(map[string]int, len(ids))
	for _, id := range ids {
		result[id] = references.counts[id]
	}
	return result, nil
}

func (references *memoryEncoderMembershipReferences) set(id string, count int) {
	references.mu.Lock()
	defer references.mu.Unlock()
	references.counts[id] = count
}
