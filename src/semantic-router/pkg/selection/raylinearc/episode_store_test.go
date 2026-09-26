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
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
)

func TestMemoryEpisodeStoreSerializesAndCommits(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	store := newTestMemoryEpisodeStore(t, 4, func() time.Time {
		return now
	})
	episode := HashEpisodeID("memory-episode")
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	requireARCNoError(t, err)
	if state.TurnIndex != 0 || lease.Version() != 1 {
		t.Fatalf("initial state=%#v lease=%d", state, lease.Version())
	}

	waiting := make(chan struct{})
	acquired := make(chan *EpisodeState, 1)
	go func() {
		close(waiting)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		nextLease, nextState, prepareErr := store.Prepare(ctx, episode, 2)
		if prepareErr != nil {
			acquired <- nil
			return
		}
		_ = store.Abort(context.Background(), nextLease)
		acquired <- nextState
	}()
	<-waiting
	select {
	case <-acquired:
		t.Fatal("same episode acquired before first lease finalized")
	case <-time.After(20 * time.Millisecond):
	}

	requireARCNoError(t, state.Commit(1, 123, now))
	requireARCNoError(t, store.Commit(
		context.Background(),
		lease,
		lease.Version(),
		state,
	))
	next := <-acquired
	if next == nil || next.TurnIndex != 1 ||
		next.PreviousArm == nil || *next.PreviousArm != 1 {
		t.Fatalf("committed state not observed: %#v", next)
	}
}

func TestMemoryEpisodeStoreDifferentEpisodesDoNotBlock(t *testing.T) {
	store := newTestMemoryEpisodeStore(t, 2, time.Now)
	first, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("first"),
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	second, _, err := store.Prepare(ctx, HashEpisodeID("second"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if abortErr := store.Abort(
		context.Background(),
		first,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
	if abortErr := store.Abort(
		context.Background(),
		second,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
}

func TestMemoryEpisodeStoreTimeoutStaleLeaseAndCapacity(t *testing.T) {
	store := newTestMemoryEpisodeStore(t, 1, time.Now)
	episode := HashEpisodeID("held")
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, _, err := store.Prepare(ctx, episode, 2); !errors.Is(
		err,
		context.DeadlineExceeded,
	) {
		t.Fatalf("same-episode timeout error = %v", err)
	}
	if _, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("capacity"),
		2,
	); !errors.Is(err, ErrEpisodeCapacity) {
		t.Fatalf("capacity error = %v", err)
	}
	stale := lease
	stale.ownerToken = strings.Repeat("0", len(stale.ownerToken))
	if err := store.Commit(
		context.Background(),
		stale,
		stale.Version(),
		state,
	); !errors.Is(err, ErrEpisodeLeaseLost) {
		t.Fatalf("stale commit error = %v", err)
	}
	if err := store.Abort(context.Background(), lease); err != nil {
		t.Fatal(err)
	}
	if err := store.Abort(context.Background(), lease); err != nil {
		t.Fatalf("idempotent abort error = %v", err)
	}
}

func TestMemoryEpisodeStoreReapsIdleEntries(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	store := newTestMemoryEpisodeStore(t, 1, func() time.Time {
		return now
	})
	first, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("old"),
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if abortErr := store.Abort(
		context.Background(),
		first,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
	now = now.Add(2 * time.Minute)
	second, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("new"),
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if abortErr := store.Abort(
		context.Background(),
		second,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
}

func TestEpisodeStateWireRejectsFutureAndUnknownFields(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	state.Warmth[0] = &WorkerWarmth{
		LastUsed:        now.Add(maxFutureClockSkew + time.Millisecond),
		LastInputTokens: 1,
	}
	if _, err := marshalEpisodeState(state, 1, now); err == nil {
		t.Fatal("future timestamp accepted")
	}
	payload := []byte(
		`{"schema_version":"rayline.arc.episode-state.v1",` +
			`"version":1,"previous_arm":null,"turn_index":0,` +
			`"warmth":[null],"secret":"no"}`,
	)
	if _, _, err := unmarshalEpisodeState(payload, 1, now); err == nil {
		t.Fatal("unknown field accepted")
	}
}

func TestEpisodeStateWireV2PersistsEncoderAffinity(t *testing.T) {
	now := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	state.EncoderOwner = "replica-b"
	state.EncoderVisitedOwners = []string{"replica-a", "replica-b"}
	payload, err := marshalEpisodeState(state, 7, now)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v2"`) ||
		!strings.Contains(string(payload), `"encoder_owner":"replica-b"`) {
		t.Fatalf("v2 payload omitted affinity: %s", payload)
	}
	decoded, version, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if version != 7 || decoded.EncoderOwner != "replica-b" ||
		!slices.Equal(
			decoded.EncoderVisitedOwners,
			[]string{"replica-a", "replica-b"},
		) {
		t.Fatalf("decoded v2 state/version = %#v/%d", decoded, version)
	}
}

func TestEpisodeStateWireReadsV1WithoutEncoderAffinity(t *testing.T) {
	now := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)
	legacy := []byte(
		`{"schema_version":"rayline.arc.episode-state.v1",` +
			`"version":6,"previous_arm":null,"turn_index":0,` +
			`"warmth":[null]}`,
	)
	decoded, version, err := unmarshalEpisodeState(legacy, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if version != 6 || decoded.EncoderOwner != "" ||
		len(decoded.EncoderVisitedOwners) != 0 {
		t.Fatalf("legacy migration = %#v/%d", decoded, version)
	}
}

func TestEpisodeStateWireRejectsInvalidOrIncompleteV2Affinity(t *testing.T) {
	now := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	state.EncoderOwner = "replica-b"
	state.EncoderVisitedOwners = []string{"replica-a"}
	if _, err := marshalEpisodeState(state, 1, now); err == nil {
		t.Fatal("owner absent from visited set was accepted")
	}
	incomplete := []byte(
		`{"schema_version":"rayline.arc.episode-state.v2",` +
			`"version":1,"previous_arm":null,"turn_index":0,` +
			`"warmth":[null],"encoder_owner":""}`,
	)
	if _, _, err := unmarshalEpisodeState(incomplete, 1, now); err == nil {
		t.Fatal("v2 state without visited-owner field was accepted")
	}
}

func newTestMemoryEpisodeStore(
	t *testing.T,
	maxEpisodes int,
	now func() time.Time,
) *MemoryEpisodeStore {
	t.Helper()
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{
		MaxEpisodes: maxEpisodes,
		IdleTTL:     time.Minute,
		Now:         now,
	})
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func requireARCNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func fullThinkingLedger() *thinkinglever.Ledger {
	ledger := &thinkinglever.Ledger{}
	for index := 0; index < thinkinglever.MaxPayloads; index++ {
		ledger.Payloads = append(ledger.Payloads, thinkinglever.Payload{
			Lever:  thinkinglever.LeverSteeringSuffix,
			Suffix: strings.Repeat(string(rune('a'+index)), thinkinglever.MaxSuffixBytes),
		})
	}
	for index := 0; index < thinkinglever.MaxLedgerLength; index++ {
		ledger.Entries = append(ledger.Entries, thinkinglever.LedgerEntry{
			Index:     uint32(1<<31 + index),
			Placement: thinkinglever.PlaceUserAfterToolRun,
			Digest:    strings.Repeat("f", thinkinglever.DigestBytes*2),
			Payload:   index % thinkinglever.MaxPayloads,
			Turn:      1<<63 + uint64(index),
		})
	}
	ledger.InForce = []thinkinglever.LeverState{{
		Lever: thinkinglever.LeverSteeringSuffix, Payload: thinkinglever.MaxPayloads - 1,
		Level: strings.Repeat("l", thinkinglever.MaxLevelName), LastChangeTurn: 1 << 63,
	}}
	return ledger
}

func TestEpisodeStateWireV3CarriesTheThinkingLedger(t *testing.T) {
	now := time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v2"`) ||
		strings.Contains(string(payload), `"thinking"`) {
		t.Fatalf("an episode without a ledger must keep its v2 bytes: %s", payload)
	}

	// The worst case: every entry and name at its limit still fits.
	state.Thinking = fullThinkingLedger()
	payload, err = marshalEpisodeState(state, 2, now)
	if err != nil {
		t.Fatalf("a full ledger does not fit: %v", err)
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v3"`) {
		t.Fatalf("ledger written without v3: %.120s", payload)
	}
	decoded, version, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if version != 2 || !reflect.DeepEqual(decoded.Thinking, state.Thinking) {
		t.Fatal("the ledger did not round-trip")
	}
	cloned := cloneEpisodeState(decoded)
	cloned.Thinking.Payloads[0].Suffix = "changed"
	if decoded.Thinking.Payloads[0].Suffix == "changed" {
		t.Fatal("clone shares ledger entries")
	}
}

func TestEpisodeStateWireGatesTheLedgerOnTheSchema(t *testing.T) {
	now := time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)
	ledger := `"thinking":{"epoch":0,"payloads":[],"entries":[],"in_force":[]}`
	for name, payload := range map[string]string{
		"v2 with a ledger": `{"schema_version":"rayline.arc.episode-state.v2","version":1,` +
			`"previous_arm":null,"turn_index":0,"warmth":[null],"encoder_owner":"",` +
			`"encoder_visited_owners":[],` + ledger + `}`,
		"v3 without a ledger": `{"schema_version":"rayline.arc.episode-state.v3","version":1,` +
			`"previous_arm":null,"turn_index":0,"warmth":[null],"encoder_owner":"",` +
			`"encoder_visited_owners":[]}`,
		"v3 with an unknown placement code": `{"schema_version":"rayline.arc.episode-state.v3","version":1,` +
			`"previous_arm":null,"turn_index":0,"warmth":[null],"encoder_owner":"",` +
			`"encoder_visited_owners":[],"thinking":{"epoch":0,` +
			`"payloads":[{"lever":"prompt_steering_suffix","suffix":"x"}],` +
			`"entries":[{"i":0,"p":"nowhere","d":"` + strings.Repeat("f", 32) + `","k":0,"t":0}],"in_force":[]}}`,
		"v3 with an entry naming a missing payload": `{"schema_version":"rayline.arc.episode-state.v3","version":1,` +
			`"previous_arm":null,"turn_index":0,"warmth":[null],"encoder_owner":"",` +
			`"encoder_visited_owners":[],"thinking":{"epoch":0,"payloads":[],` +
			`"entries":[{"i":0,"p":"a","d":"` + strings.Repeat("f", 32) + `","k":0,"t":0}],"in_force":[]}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := unmarshalEpisodeState([]byte(payload), 1, now); err == nil {
				t.Fatal("payload accepted")
			}
		})
	}
}

func TestEpisodeStateWireCarriesUpstreamPrefixesUnderV3(t *testing.T) {
	now := time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < MaxUpstreamPrefixes+2; index++ {
		state.Upstream = WithUpstreamPrefix(state.Upstream, UpstreamPrefix{
			Worker: strings.Repeat("w", 400) + string(rune('a'+index)), Messages: index, Digest: strings.Repeat("0", 32),
		})
	}
	if len(state.Upstream) != MaxUpstreamPrefixes || state.Upstream[0].Messages != 2 {
		t.Fatalf("records not bounded to the most recent: %d, oldest %d", len(state.Upstream), state.Upstream[0].Messages)
	}
	payload, err := marshalEpisodeState(state, 3, now)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v3"`) {
		t.Fatalf("upstream records written without v3: %.100s", payload)
	}
	decoded, _, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded.Upstream, state.Upstream) {
		t.Fatal("upstream records did not round-trip")
	}
	if prefix, ok := decoded.UpstreamPrefixFor(state.Upstream[3].Worker); !ok || prefix.Messages != 5 {
		t.Fatalf("lookup = %+v, %v", prefix, ok)
	}
}
