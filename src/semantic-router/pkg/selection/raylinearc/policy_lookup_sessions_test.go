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
	"testing"
	"time"
)

// The Redis store pins a lookup scope's session beside the episodes, for
// every router instance, until its idle timeout; it is not the episode's
// state, which stays absent.
func TestRedisEpisodeStoreLookupSessionPin(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc:" + HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, time.Second)
	ctx := context.Background()
	scope := HashEpisodeID("lookup:user-1:conv-1")

	if pinned, err := store.LoadLookupSession(ctx, scope); err != nil || pinned != nil {
		t.Fatalf("absent pin: %v %v", pinned, err)
	}
	session := PolicySession{ID: "sd-1", Token: "tok.1.secret"}
	requireARCNoError(t, store.StoreLookupSession(ctx, scope, session, 600*time.Second))
	pinned, err := store.LoadLookupSession(ctx, scope)
	if err != nil || pinned == nil || *pinned != session {
		t.Fatalf("stored pin: %v %v", pinned, err)
	}
	ttl, err := store.client.PTTL(ctx, store.lookupSessionKey(scope)).Result()
	requireARCNoError(t, err)
	if ttl <= 590*time.Second || ttl > 600*time.Second {
		t.Fatalf("pin TTL = %s, want the lookup idle timeout", ttl)
	}
	if exists, _ := store.client.Exists(ctx, prefix+scope).Result(); exists != 0 {
		t.Fatal("pinning a lookup session wrote episode state")
	}
	if err := store.StoreLookupSession(ctx, scope, PolicySession{}, time.Second); err == nil {
		t.Fatal("an invalid session was pinned")
	}
}
