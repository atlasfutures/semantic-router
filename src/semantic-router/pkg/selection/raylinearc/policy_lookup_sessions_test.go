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
	"net"
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
	pinnedNow, err := store.StoreLookupSession(ctx, scope, nil, session, 600*time.Second)
	if err != nil || !pinnedNow {
		t.Fatalf("first pin: %v %v", pinnedNow, err)
	}
	other := PolicySession{ID: "sd-2", Token: "tok.2.secret"}
	if raced, err := store.StoreLookupSession(ctx, scope, nil, other, 600*time.Second); err != nil || raced {
		t.Fatalf("a racing first pin: %v %v", raced, err)
	}
	if refreshed, err := store.StoreLookupSession(ctx, scope, &session, session, 600*time.Second); err != nil || !refreshed {
		t.Fatalf("refresh: %v %v", refreshed, err)
	}
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
	if moved, err := store.StoreLookupSession(ctx, scope, &session, other, 600*time.Second); err != nil || !moved {
		t.Fatalf("move: %v %v", moved, err)
	}
	if stale, err := store.StoreLookupSession(ctx, scope, &session, session, 600*time.Second); err != nil || stale {
		t.Fatalf("a retired session was restored: %v %v", stale, err)
	}
	if pinned, _ := store.LoadLookupSession(ctx, scope); pinned == nil || *pinned != other {
		t.Fatalf("pin after the move = %v", pinned)
	}
	if _, err := store.StoreLookupSession(ctx, scope, nil, PolicySession{}, time.Second); err == nil {
		t.Fatal("an invalid session was pinned")
	}
}

// A pin call ends with its own context on a Redis that accepts and never
// answers: the store's client would hold it for its socket timeout
// (seconds). The episode store's own client is left as it was.
func TestRedisLookupSessionPinHonoursItsDeadline(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			defer conn.Close()
		}
	}()
	store, err := NewRedisEpisodeStore(RedisEpisodeStoreConfig{
		Address: listener.Addr().String(), KeyPrefix: "test:", LeaseTTL: time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	key := HashEpisodeID("lookup-session:scope")
	session := PolicySession{ID: "sd-1", Token: "tok.1.secret"}
	for name, call := range map[string]func(context.Context) error{
		"read": func(ctx context.Context) error {
			_, err := store.LoadLookupSession(ctx, key)
			return err
		},
		"write": func(ctx context.Context) error {
			_, err := store.StoreLookupSession(ctx, key, nil, session, time.Minute)
			return err
		},
	} {
		ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
		started := time.Now()
		err := call(ctx)
		cancel()
		if err == nil {
			t.Fatalf("%s: a silent Redis answered", name)
		}
		if elapsed := time.Since(started); elapsed > time.Second {
			t.Fatalf("%s: the pin call took %s past its 150ms deadline", name, elapsed)
		}
	}
	if store.client.Options().ContextTimeoutEnabled {
		t.Fatal("the episode store's own client was changed")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.LoadLookupSession(context.Background(), key); err == nil {
		t.Fatal("a closed store's pin client answered")
	}
}

// Close lets a pin call in flight finish on its own deadline rather than
// failing it with the client closed, and refuses calls after it.
func TestRedisLookupSessionPinCloseWaitsForCallsInFlight(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			defer conn.Close()
		}
	}()
	store, err := NewRedisEpisodeStore(RedisEpisodeStoreConfig{
		Address: listener.Addr().String(), KeyPrefix: "test:", LeaseTTL: time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	key := HashEpisodeID("lookup-session:scope")
	finished := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
		defer cancel()
		_, err := store.LoadLookupSession(ctx, key)
		finished <- err
	}()
	time.Sleep(50 * time.Millisecond)
	started := time.Now()
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if waited := time.Since(started); waited < 150*time.Millisecond {
		t.Fatalf("Close returned after %s, before the call in flight", waited)
	}
	if err := <-finished; err == nil {
		t.Fatal("a silent Redis answered")
	}
}
