/*
Copyright 2025 vLLM Semantic Router.

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
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"
	"time"
)

func TestEncoderPoolDeterministicAffinityAndDrainingMembership(t *testing.T) {
	t.Parallel()
	activeServer, activeCalls := newEncoderReplicaTestServer(t)
	defer activeServer.Close()
	drainingServer, drainingCalls := newEncoderReplicaTestServer(t)
	defer drainingServer.Close()

	pool := newEncoderTestPool(t, []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, activeServer.URL)},
		{ID: "replica-b", State: EncoderReplicaDraining, Client: newRetainedSessionTestClient(t, drainingServer.URL)},
	})
	episodeHash := HashEpisodeID("deterministic-membership")
	turns := []Turn{{Role: "user", Text: "public test turn"}}

	first, err := pool.EncodeWithAffinity(
		context.Background(),
		episodeHash,
		turns,
		EncoderAffinity{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if first.ReplicaID != "replica-a" || first.ReplicaIndex != 0 {
		t.Fatalf("new episode owner = %q/%d, want active replica-a/0", first.ReplicaID, first.ReplicaIndex)
	}
	second, err := pool.EncodeWithAffinity(
		context.Background(),
		episodeHash,
		turns,
		EncoderAffinity{Owner: "replica-b", Visited: []string{"replica-b"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if second.ReplicaID != "replica-b" || second.ReplicaIndex != 1 {
		t.Fatalf("existing episode owner = %q/%d, want draining replica-b/1", second.ReplicaID, second.ReplicaIndex)
	}
	if activeCalls.Load() != 1 || drainingCalls.Load() != 1 {
		t.Fatalf("POST calls = active %d draining %d, want 1/1", activeCalls.Load(), drainingCalls.Load())
	}
}

func TestEncoderPoolRemapsOnceOnConfiguredStatusAndPersistsSurvivor(t *testing.T) {
	t.Parallel()
	var statusA atomic.Int32
	var statusB atomic.Int32
	serverA, callsA := newMutableEncoderReplicaTestServer(t, &statusA)
	defer serverA.Close()
	serverB, callsB := newMutableEncoderReplicaTestServer(t, &statusB)
	defer serverB.Close()
	pool := newEncoderTestPool(t, []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverA.URL)},
		{ID: "replica-b", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverB.URL)},
	})
	episodeHash, primary := episodeForEncoderReplica(t, pool, "replica-a")
	statusA.Store(http.StatusServiceUnavailable)

	result, err := pool.EncodeWithAffinity(
		context.Background(),
		episodeHash,
		[]Turn{{Role: "user", Text: "public test turn"}},
		EncoderAffinity{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if primary.ID != "replica-a" || result.ReplicaID != "replica-b" ||
		!result.ReplicaFailover || result.ReplicaAttempts != 2 {
		t.Fatalf("failover result = primary %q owner %q failover %t attempts %d", primary.ID, result.ReplicaID, result.ReplicaFailover, result.ReplicaAttempts)
	}
	if !slices.Equal(result.VisitedReplicaIDs, []string{"replica-a", "replica-b"}) {
		t.Fatalf("visited = %v, want [replica-a replica-b]", result.VisitedReplicaIDs)
	}

	// Even after the original replica becomes healthy, persisted owner affinity
	// keeps the episode on the rebuilt survivor.
	statusA.Store(http.StatusOK)
	second, err := pool.EncodeWithAffinity(
		context.Background(),
		episodeHash,
		[]Turn{{Role: "user", Text: "public second turn"}},
		EncoderAffinity{Owner: result.ReplicaID, Visited: result.VisitedReplicaIDs},
	)
	if err != nil {
		t.Fatal(err)
	}
	if second.ReplicaID != "replica-b" || second.ReplicaAttempts != 1 {
		t.Fatalf("sticky survivor = %q attempts %d", second.ReplicaID, second.ReplicaAttempts)
	}
	if callsA.Load() != 1 || callsB.Load() != 2 {
		t.Fatalf("POST calls = replica-a %d replica-b %d, want 1/2", callsA.Load(), callsB.Load())
	}
}

func TestEncoderPoolFailsClosedOnAmbiguousTransportFailure(t *testing.T) {
	t.Parallel()
	episodeHash := HashEpisodeID("transport-ambiguity")
	primaryID := rendezvousWinner(episodeHash, []string{"replica-a", "replica-b"})
	var survivorCalls atomic.Int32
	transportFailureClient := retainedClientWithTransport(t, roundTripFunc(
		func(*http.Request) (*http.Response, error) {
			return nil, errors.New("private transport detail")
		},
	))
	survivorServer, _ := newEncoderReplicaTestServer(t)
	defer survivorServer.Close()
	survivorClient := newRetainedSessionTestClient(t, survivorServer.URL)
	survivorClient.httpClient.Transport = roundTripFunc(
		func(request *http.Request) (*http.Response, error) {
			survivorCalls.Add(1)
			return http.DefaultTransport.RoundTrip(request)
		},
	)
	replicas := []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: survivorClient},
		{ID: "replica-b", State: EncoderReplicaActive, Client: survivorClient},
	}
	if primaryID == "replica-a" {
		replicas[0].Client = transportFailureClient
	} else {
		replicas[1].Client = transportFailureClient
	}
	// The two logical replica entries need distinct client instances.
	if replicas[0].Client == replicas[1].Client {
		replicas[1].Client = newRetainedSessionTestClient(t, survivorServer.URL)
	}
	pool := newEncoderTestPool(t, replicas)

	_, err := pool.EncodeWithAffinity(
		context.Background(),
		episodeHash,
		[]Turn{{Role: "user", Text: "public test turn"}},
		EncoderAffinity{},
	)
	var failure *EncoderFailure
	if !errors.As(err, &failure) || failure.Class != EncoderFailureTransport {
		t.Fatalf("encode error = %v, want bounded transport failure", err)
	}
	if survivorCalls.Load() != 0 {
		t.Fatalf("survivor calls = %d, want zero ambiguous remaps", survivorCalls.Load())
	}
}

func TestEncoderPoolCooldownExpiresForNewAssignment(t *testing.T) {
	t.Parallel()
	var statusA atomic.Int32
	serverA, callsA := newMutableEncoderReplicaTestServer(t, &statusA)
	defer serverA.Close()
	serverB, _ := newEncoderReplicaTestServer(t)
	defer serverB.Close()
	pool := newEncoderTestPool(t, []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverA.URL)},
		{ID: "replica-b", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverB.URL)},
	})
	clock := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)
	pool.now = func() time.Time { return clock }
	episodeHash, _ := episodeForEncoderReplica(t, pool, "replica-a")
	statusA.Store(http.StatusServiceUnavailable)
	if _, err := pool.Encode(
		context.Background(),
		episodeHash,
		[]Turn{{Role: "user", Text: "public test turn"}},
	); err != nil {
		t.Fatal(err)
	}
	statusA.Store(http.StatusOK)
	clock = clock.Add(pool.config.UnavailableCooldown + time.Nanosecond)
	result, err := pool.Encode(
		context.Background(),
		episodeHash,
		[]Turn{{Role: "user", Text: "public new episode assignment"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.ReplicaID != "replica-a" || callsA.Load() != 2 {
		t.Fatalf("post-cooldown owner/calls = %q/%d, want replica-a/2", result.ReplicaID, callsA.Load())
	}
}

func TestEncoderPoolCachedUnavailableOwnerReportsRemap(t *testing.T) {
	t.Parallel()
	serverA, _ := newEncoderReplicaTestServer(t)
	defer serverA.Close()
	serverB, _ := newEncoderReplicaTestServer(t)
	defer serverB.Close()
	pool := newEncoderTestPool(t, []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverA.URL)},
		{ID: "replica-b", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverB.URL)},
	})
	episodeHash, _ := episodeForEncoderReplica(t, pool, "replica-a")
	pool.markUnavailable("replica-a")
	result, err := pool.EncodeWithAffinity(
		context.Background(),
		episodeHash,
		[]Turn{{Role: "user", Text: "public test turn"}},
		EncoderAffinity{Owner: "replica-a", Visited: []string{"replica-a"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.ReplicaID != "replica-b" || !result.ReplicaFailover ||
		result.ReplicaAttempts != 1 {
		t.Fatalf(
			"cached remap = owner %q failover %t attempts %d",
			result.ReplicaID,
			result.ReplicaFailover,
			result.ReplicaAttempts,
		)
	}
}

func TestEncoderPoolFailsClosedIfPersistedOwnerWasRemoved(t *testing.T) {
	t.Parallel()
	serverA, callsA := newEncoderReplicaTestServer(t)
	defer serverA.Close()
	serverC, callsC := newEncoderReplicaTestServer(t)
	defer serverC.Close()
	pool := newEncoderTestPool(t, []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverA.URL)},
		{ID: "replica-c", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverC.URL)},
	})
	_, err := pool.EncodeWithAffinity(
		context.Background(),
		HashEpisodeID("premature-removal"),
		[]Turn{{Role: "user", Text: "public test turn"}},
		EncoderAffinity{Owner: "replica-b", Visited: []string{"replica-b"}},
	)
	var failure *EncoderFailure
	if !errors.As(err, &failure) ||
		failure.Class != EncoderFailureContract ||
		failure.Stage != "replica_owner_missing" {
		t.Fatalf("removed-owner error = %v", err)
	}
	if callsA.Load() != 0 || callsC.Load() != 0 {
		t.Fatalf("removed owner dispatched to replacement: calls=%d/%d", callsA.Load(), callsC.Load())
	}
}

func TestEncoderPoolCloseFansOutAndAcceptsUnavailableReplica(t *testing.T) {
	t.Parallel()
	var closeStatusA atomic.Int32
	var closeStatusB atomic.Int32
	serverA, _, closeCallsA := newCloseAwareEncoderReplicaTestServer(t, &closeStatusA)
	defer serverA.Close()
	serverB, _, closeCallsB := newCloseAwareEncoderReplicaTestServer(t, &closeStatusB)
	defer serverB.Close()
	closeStatusB.Store(http.StatusServiceUnavailable)
	pool := newEncoderTestPool(t, []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverA.URL)},
		{ID: "replica-b", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverB.URL)},
	})
	report, err := pool.CloseSession(
		context.Background(),
		HashEpisodeID("close-fanout"),
		[]string{"replica-a", "replica-b"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if report.Attempted != 2 || report.Closed != 1 || report.Unavailable != 1 {
		t.Fatalf("close report = %+v, want attempted=2 closed=1 unavailable=1", report)
	}
	if closeCallsA.Load() != 1 || closeCallsB.Load() != 1 {
		t.Fatalf("close calls = %d/%d, want 1/1", closeCallsA.Load(), closeCallsB.Load())
	}
}

func TestEncoderPoolProbeCoversActiveAndDrainingReplicas(t *testing.T) {
	t.Parallel()
	serverA, postA, closeA := newCloseAwareEncoderReplicaTestServer(t, new(atomic.Int32))
	defer serverA.Close()
	serverB, postB, closeB := newCloseAwareEncoderReplicaTestServer(t, new(atomic.Int32))
	defer serverB.Close()
	pool := newEncoderTestPool(t, []EncoderReplica{
		{ID: "replica-a", State: EncoderReplicaActive, Client: newRetainedSessionTestClient(t, serverA.URL)},
		{ID: "replica-b", State: EncoderReplicaDraining, Client: newRetainedSessionTestClient(t, serverB.URL)},
	})
	if err := pool.Probe(context.Background(), "startup-probe"); err != nil {
		t.Fatal(err)
	}
	if postA.Load() != 1 || postB.Load() != 1 || closeA.Load() != 1 || closeB.Load() != 1 {
		t.Fatalf("probe calls post=%d/%d close=%d/%d, want all 1", postA.Load(), postB.Load(), closeA.Load(), closeB.Load())
	}
}

func newEncoderTestPool(t *testing.T, replicas []EncoderReplica) *EncoderPool {
	t.Helper()
	pool, err := NewEncoderPool(replicas, EncoderPoolConfig{
		SchemaVersion:          EncoderFailoverSchemaV1,
		UnavailableStatusCodes: []int{http.StatusServiceUnavailable},
		UnavailableCooldown:    30 * time.Second,
		MaxRemaps:              1,
	})
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func episodeForEncoderReplica(
	t *testing.T,
	pool *EncoderPool,
	replicaID string,
) (string, *EncoderReplica) {
	t.Helper()
	for index := 0; index < 10_000; index++ {
		episodeHash := HashEpisodeID(fmt.Sprintf("episode-%d", index))
		replica, err := pool.primaryReplica(episodeHash, "", nil)
		if err != nil {
			t.Fatal(err)
		}
		if replica.ID == replicaID {
			return episodeHash, replica
		}
	}
	t.Fatalf("no episode assigned to %q", replicaID)
	return "", nil
}

func rendezvousWinner(episodeHash string, replicaIDs []string) string {
	winner := replicaIDs[0]
	winnerScore := encoderReplicaScore(episodeHash, winner)
	for _, replicaID := range replicaIDs[1:] {
		score := encoderReplicaScore(episodeHash, replicaID)
		if string(score[:]) > string(winnerScore[:]) {
			winner = replicaID
			winnerScore = score
		}
	}
	return winner
}

func retainedClientWithTransport(
	t *testing.T,
	transport http.RoundTripper,
) *EncoderClient {
	t.Helper()
	config := validEncoderClientConfig("http://encoder.test")
	config.ServingRung = encoderServingRungB
	config.RequiredCapabilities = []string{
		"chunked_causal_mean",
		"resumable_causal_mean",
	}
	config.RetainedSession = true
	client, err := newEncoderClient(config, &http.Client{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func newEncoderReplicaTestServer(
	t *testing.T,
) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	status := new(atomic.Int32)
	return newMutableEncoderReplicaTestServer(t, status)
}

func newMutableEncoderReplicaTestServer(
	t *testing.T,
	status *atomic.Int32,
) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	calls := new(atomic.Int32)
	server := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			if request.Method != http.MethodPost {
				t.Errorf("method = %s, want POST", request.Method)
				writer.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			calls.Add(1)
			if code := int(status.Load()); code != 0 && code != http.StatusOK {
				writer.WriteHeader(code)
				return
			}
			writeSessionEncoderResponse(t, writer, nil)
		},
	))
	return server, calls
}

func newCloseAwareEncoderReplicaTestServer(
	t *testing.T,
	closeStatus *atomic.Int32,
) (*httptest.Server, *atomic.Int32, *atomic.Int32) {
	t.Helper()
	postCalls := new(atomic.Int32)
	closeCalls := new(atomic.Int32)
	server := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			switch request.Method {
			case http.MethodPost:
				postCalls.Add(1)
				writeSessionEncoderResponse(t, writer, nil)
			case http.MethodDelete:
				closeCalls.Add(1)
				if code := int(closeStatus.Load()); code != 0 && code != http.StatusOK {
					writer.WriteHeader(code)
					return
				}
				writer.Header().Set("content-type", "application/json")
				_, _ = writer.Write([]byte(`{"closed":true}`))
			default:
				writer.WriteHeader(http.StatusMethodNotAllowed)
			}
		},
	))
	return server, postCalls, closeCalls
}
