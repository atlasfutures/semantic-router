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
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// PolicyLookupSessionIdleTimeout is how long a lookup scope's Modal session
// and its pin live after the scope's last lookup. It matches the policy
// service's default lookup-session idle TTL (ARC_LOOKUP_SESSION_IDLE_S,
// 600 s): a pin that outlived the service's encoder session would route the
// next lookup to a container that no longer holds it, and one that expired
// sooner would move it off a container that still does.
const PolicyLookupSessionIdleTimeout = 600 * time.Second

// PolicyLookupSessionPins keeps, per lookup scope, the Modal session its
// lookups are decided on (a sessioned policy Server), so each lookup on a
// conversation reaches the container that holds the scope's warm encoder
// session. It is not episode state: a lookup scope has no episode, and the
// pin holds nothing but the session. The token is a credential, as the one
// the episode state persists is; it is never logged.
type PolicyLookupSessionPins interface {
	// LoadLookupSession answers the scope's pinned session, or nil.
	LoadLookupSession(ctx context.Context, scopeHash string) (*PolicySession, error)
	// StoreLookupSession pins session for the scope until idle from now.
	StoreLookupSession(ctx context.Context, scopeHash string, session PolicySession, idle time.Duration) error
}

func (store *RedisEpisodeStore) lookupSessionKey(scopeHash string) string {
	return store.keyPrefix + scopeHash + ":lookup-session"
}

// LoadLookupSession reads the scope's pin, shared by every cell instance.
func (store *RedisEpisodeStore) LoadLookupSession(ctx context.Context, scopeHash string) (*PolicySession, error) {
	if store == nil || store.client == nil || !validEpisodeIDHash(scopeHash) {
		return nil, errors.New("invalid ARC Redis lookup session request")
	}
	fields, err := store.client.HMGet(ctx, store.lookupSessionKey(scopeHash), "id", "token").Result()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, boundedRedisEpisodeError("lookup session read", err)
	}
	id, _ := fields[0].(string)
	token, _ := fields[1].(string)
	if id == "" && token == "" {
		return nil, nil
	}
	session := PolicySession{ID: id, Token: token}
	if ValidatePolicySession(session) != nil {
		return nil, errors.New("ARC Redis lookup session is invalid")
	}
	return &session, nil
}

// StoreLookupSession writes the scope's pin with its idle expiry.
func (store *RedisEpisodeStore) StoreLookupSession(
	ctx context.Context,
	scopeHash string,
	session PolicySession,
	idle time.Duration,
) error {
	if store == nil || store.client == nil || !validEpisodeIDHash(scopeHash) ||
		ValidatePolicySession(session) != nil || idle <= 0 {
		return errors.New("invalid ARC Redis lookup session write")
	}
	key := store.lookupSessionKey(scopeHash)
	pipe := store.client.TxPipeline()
	pipe.HSet(ctx, key, "id", session.ID, "token", session.Token)
	pipe.PExpire(ctx, key, idle)
	if _, err := pipe.Exec(ctx); err != nil {
		return boundedRedisEpisodeError("lookup session write", err)
	}
	return nil
}

// MemoryPolicyLookupSessionPins is a process-local PolicyLookupSessionPins,
// for a cell whose episode store keeps nothing shared: each router instance
// pins its own session per scope.
type MemoryPolicyLookupSessionPins struct {
	mu      sync.Mutex
	entries map[string]memoryLookupSessionPin
	now     func() time.Time
}

type memoryLookupSessionPin struct {
	session PolicySession
	expiry  time.Time
}

// NewMemoryPolicyLookupSessionPins is an empty process-local pin table.
func NewMemoryPolicyLookupSessionPins() *MemoryPolicyLookupSessionPins {
	return &MemoryPolicyLookupSessionPins{entries: map[string]memoryLookupSessionPin{}, now: time.Now}
}

// LoadLookupSession answers the scope's live pin, or nil.
func (pins *MemoryPolicyLookupSessionPins) LoadLookupSession(_ context.Context, scopeHash string) (*PolicySession, error) {
	pins.mu.Lock()
	defer pins.mu.Unlock()
	pin, found := pins.entries[scopeHash]
	if !found {
		return nil, nil
	}
	if !pins.now().Before(pin.expiry) {
		delete(pins.entries, scopeHash)
		return nil, nil
	}
	session := pin.session
	return &session, nil
}

// StoreLookupSession pins session for the scope; expired pins are dropped
// on the way, so the table is bounded by the scopes live within one idle.
func (pins *MemoryPolicyLookupSessionPins) StoreLookupSession(
	_ context.Context,
	scopeHash string,
	session PolicySession,
	idle time.Duration,
) error {
	if ValidatePolicySession(session) != nil || idle <= 0 {
		return errors.New("invalid ARC lookup session write")
	}
	pins.mu.Lock()
	defer pins.mu.Unlock()
	now := pins.now()
	for scope, pin := range pins.entries {
		if !now.Before(pin.expiry) {
			delete(pins.entries, scope)
		}
	}
	pins.entries[scopeHash] = memoryLookupSessionPin{session: session, expiry: now.Add(idle)}
	return nil
}
