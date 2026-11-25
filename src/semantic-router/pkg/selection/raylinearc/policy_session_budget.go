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
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// A sessioned policy Server holds a fixed number of sessions, shared by every
// caller. One caller fanning out parallel requests (each a fork of a busy
// parent episode, each with its own new session) can exhaust that pool, and
// then every caller's session start waits out its timeout and fails. The
// new-session budget bounds how many new sessions one caller opens in a
// rolling window; the cell decides a turn past it without opening one.

// PolicySessionBudgetWindow is the rolling window a caller's new sessions
// are counted over.
const PolicySessionBudgetWindow = 10 * time.Minute

// maxPolicySessionCallerBytes bounds the caller identity a gateway may send.
const maxPolicySessionCallerBytes = 256

// PolicySessionCallerKey is the key a caller's budget is counted under: a hex
// SHA-256 of the gateway's opaque caller id, so the id itself is never a
// Redis key or a log field. Empty (no budget) for an absent, oversized or
// non-printable id.
func PolicySessionCallerKey(caller string) string {
	caller = strings.TrimSpace(caller)
	if !visibleASCII(caller, maxPolicySessionCallerBytes) {
		return ""
	}
	digest := sha256.Sum256([]byte("rayline-arc-caller:" + caller))
	return hex.EncodeToString(digest[:])
}

// PolicySessionBudget counts the new policy sessions each caller opens.
type PolicySessionBudget interface {
	// AdmitPolicySessionStart records one new session for callerKey and
	// reports true when fewer than limit were recorded within window before
	// it; past the budget it records nothing and reports false. An error
	// means the count is unknown: the caller fails open.
	AdmitPolicySessionStart(ctx context.Context, callerKey string, limit int, window time.Duration) (bool, error)
}

func validPolicySessionBudgetRequest(callerKey string, limit int, window time.Duration) bool {
	return validEpisodeIDHash(callerKey) && limit > 0 && window > 0
}

func (store *RedisEpisodeStore) sessionStartsKey(callerKey string) string {
	return store.keyPrefix + callerKey + ":session-starts"
}

// redisSessionStartAdmitScript keeps a caller's session starts as a sorted
// set scored by unix milliseconds: it drops the ones older than the window,
// and records this one (ARGV[4]) only while fewer than the limit remain.
var redisSessionStartAdmitScript = redis.NewScript(`
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", now - window)
if redis.call("ZCARD", KEYS[1]) >= tonumber(ARGV[3]) then
  return 0
end
redis.call("ZADD", KEYS[1], now, ARGV[4])
redis.call("PEXPIRE", KEYS[1], window)
return 1
`)

// AdmitPolicySessionStart counts on the store's Redis, shared by every router
// instance, over the pins' client: each call ends with its context and none
// is retried.
func (store *RedisEpisodeStore) AdmitPolicySessionStart(
	ctx context.Context,
	callerKey string,
	limit int,
	window time.Duration,
) (bool, error) {
	if store == nil || store.client == nil || !validPolicySessionBudgetRequest(callerKey, limit, window) {
		return false, errors.New("invalid ARC Redis session budget request")
	}
	member, err := sessionStartMember(store.now())
	if err != nil {
		return false, err
	}
	client, done, err := store.lookupPinClient()
	if err != nil {
		return false, boundedRedisEpisodeError("session budget", err)
	}
	defer done()
	admitted, err := redisSessionStartAdmitScript.Run(ctx, client, []string{store.sessionStartsKey(callerKey)},
		store.now().UnixMilli(), window.Milliseconds(), limit, member).Int()
	if err != nil {
		return false, boundedRedisEpisodeError("session budget", err)
	}
	return admitted == 1, nil
}

// sessionStartMember is a unique sorted-set member for one start.
func sessionStartMember(now time.Time) (string, error) {
	raw := make([]byte, 8)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	return strconv.FormatInt(now.UnixNano(), 10) + ":" + hex.EncodeToString(raw), nil
}

// MemoryPolicySessionBudget is a process-local PolicySessionBudget, for a
// cell whose episode store keeps nothing shared: each router instance counts
// its own starts.
type MemoryPolicySessionBudget struct {
	mu     sync.Mutex
	starts map[string][]time.Time
	now    func() time.Time
}

// NewMemoryPolicySessionBudget is an empty process-local budget.
func NewMemoryPolicySessionBudget() *MemoryPolicySessionBudget {
	return &MemoryPolicySessionBudget{starts: map[string][]time.Time{}, now: time.Now}
}

// AdmitPolicySessionStart counts in process; callers whose starts all fell
// out of the window are dropped on the way, so the table stays bounded by the
// callers active within one window.
func (budget *MemoryPolicySessionBudget) AdmitPolicySessionStart(
	_ context.Context,
	callerKey string,
	limit int,
	window time.Duration,
) (bool, error) {
	if !validPolicySessionBudgetRequest(callerKey, limit, window) {
		return false, errors.New("invalid ARC session budget request")
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	now := budget.now()
	since := now.Add(-window)
	for key, starts := range budget.starts {
		kept := starts[:0]
		for _, start := range starts {
			if start.After(since) {
				kept = append(kept, start)
			}
		}
		if len(kept) == 0 {
			delete(budget.starts, key)
		} else {
			budget.starts[key] = kept
		}
	}
	if len(budget.starts[callerKey]) >= limit {
		return false, nil
	}
	budget.starts[callerKey] = append(budget.starts[callerKey], now)
	return true, nil
}
