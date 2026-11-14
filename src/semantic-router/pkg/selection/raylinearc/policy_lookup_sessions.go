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

// PolicyLookupSessionPins keeps, per lookup scope and policy backend, the
// Modal session its lookups are decided on (a sessioned policy Server), so
// each lookup on a conversation reaches the container that holds the scope's
// warm encoder session. It is not episode state: a lookup scope has no
// episode, and the pin holds nothing but the session. The token is a
// credential, as the one the episode state persists is; it is never logged.
// pinKey is a 64-hex hash naming the scope on one backend.
type PolicyLookupSessionPins interface {
	// LoadLookupSession answers the pinned session, or nil.
	LoadLookupSession(ctx context.Context, pinKey string) (*PolicySession, error)
	// StoreLookupSession pins session until idle from now, but only when the
	// pin is still previous (nil: no pin) or already session (a refresh).
	// It reports whether it pinned: false means a concurrent lookup pinned
	// another session first, and this one's must not replace it.
	StoreLookupSession(ctx context.Context, pinKey string, previous *PolicySession, session PolicySession, idle time.Duration) (bool, error)
}

// lookupPinReplaces reports whether a pin holding current may be set to
// session by a lookup that read previous.
func lookupPinReplaces(current, previous *PolicySession, session PolicySession) bool {
	switch {
	case current == nil:
		return previous == nil
	case *current == session:
		return true
	default:
		return previous != nil && *current == *previous
	}
}

func (store *RedisEpisodeStore) lookupSessionKey(pinKey string) string {
	return store.keyPrefix + pinKey + ":lookup-session"
}

// LoadLookupSession reads the pin, shared by every cell instance.
func (store *RedisEpisodeStore) LoadLookupSession(ctx context.Context, pinKey string) (*PolicySession, error) {
	if store == nil || store.client == nil || !validEpisodeIDHash(pinKey) {
		return nil, errors.New("invalid ARC Redis lookup session request")
	}
	fields, err := store.client.HMGet(ctx, store.lookupSessionKey(pinKey), "id", "token").Result()
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

// redisLookupSessionStoreScript sets the pin (id, token) with its expiry
// only when it is absent and the writer read none, already the new session,
// or still the session the writer read (ARGV[1] "1" with ARGV[2], ARGV[3]).
var redisLookupSessionStoreScript = redis.NewScript(`
local id = redis.call("HGET", KEYS[1], "id")
local token = redis.call("HGET", KEYS[1], "token")
local allowed = false
if not id then
  allowed = ARGV[1] == "0"
elseif id == ARGV[4] and token == ARGV[5] then
  allowed = true
else
  allowed = ARGV[1] == "1" and id == ARGV[2] and token == ARGV[3]
end
if not allowed then
  return 0
end
redis.call("HSET", KEYS[1], "id", ARGV[4], "token", ARGV[5])
redis.call("PEXPIRE", KEYS[1], ARGV[6])
return 1
`)

// StoreLookupSession writes the pin with its idle expiry, conditionally.
func (store *RedisEpisodeStore) StoreLookupSession(
	ctx context.Context,
	pinKey string,
	previous *PolicySession,
	session PolicySession,
	idle time.Duration,
) (bool, error) {
	if store == nil || store.client == nil || !validEpisodeIDHash(pinKey) ||
		ValidatePolicySession(session) != nil || idle <= 0 {
		return false, errors.New("invalid ARC Redis lookup session write")
	}
	read, previousID, previousToken := "0", "", ""
	if previous != nil {
		read, previousID, previousToken = "1", previous.ID, previous.Token
	}
	pinned, err := redisLookupSessionStoreScript.Run(ctx, store.client, []string{store.lookupSessionKey(pinKey)},
		read, previousID, previousToken, session.ID, session.Token, idle.Milliseconds()).Int()
	if err != nil {
		return false, boundedRedisEpisodeError("lookup session write", err)
	}
	return pinned == 1, nil
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
func (pins *MemoryPolicyLookupSessionPins) LoadLookupSession(_ context.Context, pinKey string) (*PolicySession, error) {
	pins.mu.Lock()
	defer pins.mu.Unlock()
	pin, found := pins.entries[pinKey]
	if !found {
		return nil, nil
	}
	if !pins.now().Before(pin.expiry) {
		delete(pins.entries, pinKey)
		return nil, nil
	}
	session := pin.session
	return &session, nil
}

// StoreLookupSession pins session, conditionally; expired pins are dropped
// on the way, so the table is bounded by the scopes live within one idle.
func (pins *MemoryPolicyLookupSessionPins) StoreLookupSession(
	_ context.Context,
	pinKey string,
	previous *PolicySession,
	session PolicySession,
	idle time.Duration,
) (bool, error) {
	if ValidatePolicySession(session) != nil || idle <= 0 {
		return false, errors.New("invalid ARC lookup session write")
	}
	pins.mu.Lock()
	defer pins.mu.Unlock()
	now := pins.now()
	for key, pin := range pins.entries {
		if !now.Before(pin.expiry) {
			delete(pins.entries, key)
		}
	}
	var current *PolicySession
	if pin, found := pins.entries[pinKey]; found {
		current = &pin.session
	}
	if !lookupPinReplaces(current, previous, session) {
		return false, nil
	}
	pins.entries[pinKey] = memoryLookupSessionPin{session: session, expiry: now.Add(idle)}
	return true, nil
}
