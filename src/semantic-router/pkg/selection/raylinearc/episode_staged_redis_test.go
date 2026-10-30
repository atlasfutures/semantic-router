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
	"os"
	"strings"
	"testing"
	"time"
)

func TestRedisEpisodeStoreStagedTurnRecord(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc:" + HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, time.Second)
	ctx := context.Background()
	episode := HashEpisodeID("staged-turn-episode")

	if record, err := store.LoadStagedTurn(ctx, episode); err != nil || record != nil {
		t.Fatalf("absent record: %q %v", record, err)
	}
	requireARCNoError(t, store.StoreStagedTurn(ctx, episode, []byte(`{"v":1}`)))
	if record, err := store.LoadStagedTurn(ctx, episode); err != nil || string(record) != `{"v":1}` {
		t.Fatalf("stored record: %q %v", record, err)
	}
	ttl, err := store.client.PTTL(ctx, store.stagedTurnKey(episode)).Result()
	requireARCNoError(t, err)
	if ttl <= 0 || ttl > time.Minute {
		t.Fatalf("record TTL %v, want the idle TTL", ttl)
	}
	// A conditional delete leaves a record another turn replaced.
	requireARCNoError(t, store.DeleteStagedTurn(ctx, episode, []byte(`{"v":0}`)))
	if record, _ := store.LoadStagedTurn(ctx, episode); string(record) != `{"v":1}` {
		t.Fatalf("conditional delete removed another record: %q", record)
	}
	requireARCNoError(t, store.DeleteStagedTurn(ctx, episode, []byte(`{"v":1}`)))
	if record, _ := store.LoadStagedTurn(ctx, episode); record != nil {
		t.Fatalf("conditional delete kept its record: %q", record)
	}
	requireARCNoError(t, store.StoreStagedTurn(ctx, episode, []byte(`{"v":1}`)))
	requireARCNoError(t, store.DeleteStagedTurn(ctx, episode, nil))
	if record, _ := store.LoadStagedTurn(ctx, episode); record != nil {
		t.Fatalf("delete kept the record: %q", record)
	}
	// The record never touches the episode's own keys.
	for _, key := range store.keys(episode) {
		if exists, _ := store.client.Exists(ctx, key).Result(); exists != 0 {
			t.Fatalf("record wrote %s", key)
		}
	}
	if store.StoreStagedTurn(ctx, episode, []byte(strings.Repeat("x", maxStagedTurnRecordBytes+1))) == nil {
		t.Fatal("an oversized record was written")
	}
	if store.StoreStagedTurn(ctx, "not-a-hash", []byte(`{}`)) == nil {
		t.Fatal("a record was written for an invalid episode hash")
	}
}
