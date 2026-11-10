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
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

// ErrRouteCallerOverBudget reports a lookup its caller had no budget left
// for. It is a capacity answer (429), never a failure of the lookup.
var ErrRouteCallerOverBudget = errors.New("route lookup caller budget: over budget")

// Route caller budget outcomes, as recorded by the caller.
const (
	RouteCallerAdmitted        = "admitted"
	RouteCallerRateLimited     = "rate_limited"
	RouteCallerInflightLimited = "inflight_limited"
	RouteCallerFailOpen        = "fail_open"
	// RouteCallerAnonymous is a lookup that named no caller: not counted.
	RouteCallerAnonymous = "anonymous"
)

const (
	routeCallerKeyNamespace = "rayline-arc:routes-caller:"
	// RouteCallerWindow is the fixed window lookups_per_minute counts in.
	RouteCallerWindow = time.Minute
	// routeCallerLocalMaxCallers bounds the per-process table. Past it the
	// table drops callers with nothing in flight and a past window; a table
	// still full of live callers admits the rest uncounted (fail open)
	// rather than grow without bound.
	routeCallerLocalMaxCallers = 65536
	// routeCallerInflightRetryAfter is the Retry-After on a caller that
	// already has max_inflight lookups holding the backend: those clear when
	// the backend answers, which for a busy backend takes about this long.
	routeCallerInflightRetryAfter = 2 * time.Second
)

var (
	// One round trip: count the lookup in its caller's window, then take a
	// slot in the caller's in-flight set. Returns 0 admitted, 1 over the
	// rate, 2 over the in-flight cap. A lookup over the rate takes no
	// in-flight slot. Expired holders (a crashed router) are swept first;
	// a token already holding a slot is admitted as it is.
	//
	// KEYS[1] rate counter for this window, KEYS[2] in-flight sorted set.
	// ARGV: rate limit, window ms, in-flight limit, now ms, token, lease ms.
	routeCallerAcquireScript = redis.NewScript(`
local count = redis.call("INCR", KEYS[1])
if count == 1 then redis.call("PEXPIRE", KEYS[1], 2 * tonumber(ARGV[2])) end
if count > tonumber(ARGV[1]) then return 1 end
local now = tonumber(ARGV[4])
redis.call("ZREMRANGEBYSCORE", KEYS[2], "-inf", now)
if redis.call("ZSCORE", KEYS[2], ARGV[5]) then return 0 end
if redis.call("ZCARD", KEYS[2]) >= tonumber(ARGV[3]) then return 2 end
redis.call("ZADD", KEYS[2], now + tonumber(ARGV[6]), ARGV[5])
redis.call("PEXPIRE", KEYS[2], tonumber(ARGV[6]))
return 0
`)
	routeCallerReleaseScript = redis.NewScript(`return redis.call("ZREM", KEYS[1], ARGV[1])`)
)

// RouteCallerBudgetConfig configures a per-caller lookup budget.
type RouteCallerBudgetConfig struct {
	// LookupsPerMinute is the per-caller count per window. Zero or less
	// disables the budget.
	LookupsPerMinute int
	// MaxInflight is the per-caller concurrency. Zero or less disables the
	// budget.
	MaxInflight int
	// LookupDeadline sizes the in-flight lease: a holder lives at most the
	// longer of it and RouteLookupDetachBound.
	LookupDeadline time.Duration
	// Pool names the backend the budget protects, as the lane's does.
	Pool string
	// Store is the recipe's episode store. A Redis store makes the budget
	// cell-wide; anything else leaves it per process.
	Store EpisodeStore
}

// RouteCallerBudget holds each caller to a number of lookups per minute and
// a number at once.
//
// It mirrors the lane's failure posture: with a Redis episode store the
// counts are cell-wide, and a Redis error falls back to the per-process
// counts for a short backoff (fail open), because the episode store is the
// same Redis and refusing every lookup on top of its outage would turn a
// store blip into a lookup outage. A nil budget admits everything.
type RouteCallerBudget struct {
	rate     int
	inflight int
	window   time.Duration
	leaseTTL time.Duration
	pool     string
	local    *routeCallerLocalBudget
	shared   routeCallerShared
	backoff  time.Duration
	// redisDownUntil is a UnixNano instant; until then the budget counts
	// locally only.
	redisDownUntil atomic.Int64
	now            func() time.Time
	releases       sync.WaitGroup
	onFailOpen     func(error)
}

// routeCallerShared is the cell-wide half of the budget.
type routeCallerShared interface {
	acquire(ctx context.Context, keys routeCallerKeys, rate, inflight int, window time.Duration,
		now time.Time, token string, ttl time.Duration) (int, error)
	release(ctx context.Context, keys routeCallerKeys, token string) error
}

type routeCallerKeys struct {
	rate     string
	inflight string
}

// RouteCallerTicket is an admitted lookup's hold on its caller's budget.
type RouteCallerTicket struct {
	// Outcome is RouteCallerAdmitted, RouteCallerFailOpen or
	// RouteCallerAnonymous.
	Outcome string
	// Backend is redis (cell-wide) or local (per process).
	Backend string
	release func()
}

// Held reports whether the ticket holds an in-flight slot to release.
func (ticket RouteCallerTicket) Held() bool { return ticket.release != nil }

// Release frees the caller's in-flight slot. Safe to call more than once and
// on a zero ticket.
func (ticket RouteCallerTicket) Release() {
	if ticket.release != nil {
		ticket.release()
	}
}

// RouteCallerRefusal is a lookup over its caller's budget.
type RouteCallerRefusal struct {
	// Outcome is RouteCallerRateLimited or RouteCallerInflightLimited.
	Outcome string
	Backend string
	// RetryAfter is when the caller has budget again: the window's end for
	// the rate, a short wait for the in-flight cap.
	RetryAfter time.Duration
}

func (refusal *RouteCallerRefusal) Error() string {
	return ErrRouteCallerOverBudget.Error() + ": " + refusal.Outcome
}
func (refusal *RouteCallerRefusal) Unwrap() error { return ErrRouteCallerOverBudget }

// NewRouteCallerBudget builds the budget, or returns nil when the config
// disables it.
func NewRouteCallerBudget(config RouteCallerBudgetConfig) *RouteCallerBudget {
	if config.LookupsPerMinute <= 0 || config.MaxInflight <= 0 {
		return nil
	}
	budget := &RouteCallerBudget{
		rate:     config.LookupsPerMinute,
		inflight: config.MaxInflight,
		window:   RouteCallerWindow,
		// The slot is taken before the lane's queue wait (at most the
		// lookup deadline), and the detached decision's own bound starts
		// after it, so the lease covers both.
		leaseTTL: config.LookupDeadline + max(config.LookupDeadline, RouteLookupDetachBound) + routeLookupLeaseMargin,
		pool:     config.Pool,
		local:    routeCallerLocalFor(config.Pool, config.LookupsPerMinute, config.MaxInflight),
		backoff:  routeLookupRedisBackoff,
		now:      time.Now,
	}
	if store := redisEpisodeStoreOf(config.Store); store != nil {
		budget.shared = &redisRouteCallerBudget{client: routeLookupRedisClient(store.client)}
	}
	return budget
}

// Distributed reports whether the budget is counted cell-wide.
func (budget *RouteCallerBudget) Distributed() bool { return budget != nil && budget.shared != nil }

// LookupsPerMinute is the configured rate, or zero for a disabled budget.
func (budget *RouteCallerBudget) LookupsPerMinute() int {
	if budget == nil {
		return 0
	}
	return budget.rate
}

// MaxInflight is the configured concurrency, or zero for a disabled budget.
func (budget *RouteCallerBudget) MaxInflight() int {
	if budget == nil {
		return 0
	}
	return budget.inflight
}

// SetFailOpenObserver installs the callback told about each Redis error that
// starts a backoff. Call it before the budget serves.
func (budget *RouteCallerBudget) SetFailOpenObserver(observe func(error)) {
	if budget != nil {
		budget.onFailOpen = observe
	}
}

// Close releases the budget's Redis connections once pending releases land.
func (budget *RouteCallerBudget) Close() error {
	if budget == nil || budget.shared == nil {
		return nil
	}
	budget.releases.Wait()
	if closer, ok := budget.shared.(interface{ close() error }); ok {
		return closer.close()
	}
	return nil
}

// Acquire counts one lookup against caller and takes one of its in-flight
// slots. On success the ticket must be released when the backend answers.
// Over budget the error is a *RouteCallerRefusal. A caller of "" is not
// counted. ctx bounds the one Redis call.
func (budget *RouteCallerBudget) Acquire(ctx context.Context, caller string) (RouteCallerTicket, error) {
	if budget == nil {
		return RouteCallerTicket{Outcome: RouteCallerAdmitted, Backend: RouteLookupBackendLocal}, nil
	}
	if caller == "" {
		return RouteCallerTicket{Outcome: RouteCallerAnonymous, Backend: RouteLookupBackendLocal}, nil
	}
	now := budget.now()
	token, err := newEpisodeOwnerToken()
	if err != nil {
		return budget.acquireLocal(caller, now, RouteCallerFailOpen)
	}
	if budget.shared == nil {
		return budget.acquireLocalToken(caller, token, now, RouteCallerAdmitted)
	}
	// Every admitted lookup also holds a per-process slot, whichever backend
	// admitted it, so switching between Redis and the fallback (a failure,
	// a recovery) never lets one process admit more than the cap for a
	// caller.
	if budget.redisBackingOff(now) {
		return budget.acquireLocalToken(caller, token, now, RouteCallerFailOpen)
	}
	local, err := budget.acquireLocalToken(caller, token, now, RouteCallerAdmitted)
	if err != nil {
		return RouteCallerTicket{}, err
	}
	keys := budget.keys(caller, now)
	callContext, cancel := context.WithTimeout(ctx, routeLookupRedisCallTimeout)
	verdict, err := budget.shared.acquire(callContext, keys, budget.rate, budget.inflight, budget.window,
		now, token, budget.leaseTTL)
	cancel()
	if err != nil {
		// The write may have landed with its reply lost: free the token's
		// slot off the path, and admit on the per-process count.
		budget.releaseShared(ctx, keys, token)
		if ctx.Err() == nil {
			budget.tripRedis(err)
		}
		local.Outcome = RouteCallerFailOpen
		return local, nil
	}
	switch verdict {
	case 1:
		local.Release()
		return RouteCallerTicket{}, &RouteCallerRefusal{
			Outcome: RouteCallerRateLimited, Backend: RouteLookupBackendRedis,
			RetryAfter: budget.windowRemaining(now),
		}
	case 2:
		local.Release()
		return RouteCallerTicket{}, &RouteCallerRefusal{
			Outcome: RouteCallerInflightLimited, Backend: RouteLookupBackendRedis,
			RetryAfter: routeCallerInflightRetryAfter,
		}
	}
	var once sync.Once
	return RouteCallerTicket{
		Outcome: RouteCallerAdmitted,
		Backend: RouteLookupBackendRedis,
		release: func() {
			once.Do(func() {
				local.Release()
				budget.releaseShared(ctx, keys, token)
			})
		},
	}, nil
}

func (budget *RouteCallerBudget) acquireLocal(caller string, now time.Time, outcome string) (RouteCallerTicket, error) {
	// No token could be minted: count the rate locally, hold no slot.
	return budget.acquireLocalToken(caller, "", now, outcome)
}

func (budget *RouteCallerBudget) acquireLocalToken(
	caller, token string,
	now time.Time,
	outcome string,
) (RouteCallerTicket, error) {
	verdict, tracked := budget.local.acquire(routeCallerDigest(caller), token, budget.windowIndex(now),
		now, now.Add(budget.leaseTTL))
	switch verdict {
	case 1:
		return RouteCallerTicket{}, &RouteCallerRefusal{
			Outcome: RouteCallerRateLimited, Backend: RouteLookupBackendLocal,
			RetryAfter: budget.windowRemaining(now),
		}
	case 2:
		return RouteCallerTicket{}, &RouteCallerRefusal{
			Outcome: RouteCallerInflightLimited, Backend: RouteLookupBackendLocal,
			RetryAfter: routeCallerInflightRetryAfter,
		}
	}
	if !tracked {
		// A full table or no token: admitted uncounted.
		return RouteCallerTicket{Outcome: RouteCallerFailOpen, Backend: RouteLookupBackendLocal}, nil
	}
	digest := routeCallerDigest(caller)
	var once sync.Once
	return RouteCallerTicket{
		Outcome: outcome,
		Backend: RouteLookupBackendLocal,
		release: func() { once.Do(func() { budget.local.release(digest, token) }) },
	}, nil
}

func (budget *RouteCallerBudget) releaseShared(ctx context.Context, keys routeCallerKeys, token string) {
	releaseContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), routeLookupRedisCallTimeout)
	budget.releases.Add(1)
	go func() {
		defer budget.releases.Done()
		defer cancel()
		_ = budget.shared.release(releaseContext, keys, token)
	}()
}

func (budget *RouteCallerBudget) windowIndex(now time.Time) int64 {
	return now.UnixMilli() / budget.window.Milliseconds()
}

// windowRemaining is how long until the caller's window turns over, at
// least a second so Retry-After is never zero.
func (budget *RouteCallerBudget) windowRemaining(now time.Time) time.Duration {
	windowMS := budget.window.Milliseconds()
	left := time.Duration(windowMS-now.UnixMilli()%windowMS) * time.Millisecond
	return max(left, time.Second)
}

func (budget *RouteCallerBudget) keys(caller string, now time.Time) routeCallerKeys {
	// The hash tag keeps a caller's two keys in one slot on a cluster.
	base := routeCallerKeyNamespace + "{" + budget.pool + ":" + routeCallerDigest(caller) + "}"
	return routeCallerKeys{
		rate:     base + ":rate:" + strconv.FormatInt(budget.windowIndex(now), 10),
		inflight: base + ":inflight",
	}
}

func (budget *RouteCallerBudget) redisBackingOff(now time.Time) bool {
	return now.UnixNano() < budget.redisDownUntil.Load()
}

// tripRedis starts the backoff; only the lookup that starts it reports it.
func (budget *RouteCallerBudget) tripRedis(err error) {
	now := budget.now()
	until := now.Add(budget.backoff).UnixNano()
	for {
		previous := budget.redisDownUntil.Load()
		if previous >= until {
			return
		}
		if budget.redisDownUntil.CompareAndSwap(previous, until) {
			if budget.onFailOpen != nil && previous < now.UnixNano() {
				budget.onFailOpen(err)
			}
			return
		}
	}
}

// routeCallerDigest keeps the caller (a conversation identity, which may name
// a user) out of Redis key names and process memory.
func routeCallerDigest(caller string) string {
	sum := sha256.Sum256([]byte(caller))
	return hex.EncodeToString(sum[:16])
}

// routeCallerLocalBudget is the per-process count, shared by every budget of
// one pool and limits in the process (as the lane's local semaphore is).
type routeCallerLocalBudget struct {
	mu       sync.Mutex
	rate     int
	inflight int
	callers  map[string]*routeCallerLocalEntry
}

type routeCallerLocalEntry struct {
	window  int64
	count   int
	holders map[string]time.Time
}

var routeCallerLocalBudgets sync.Map

func routeCallerLocalFor(pool string, rate, inflight int) *routeCallerLocalBudget {
	build := func() *routeCallerLocalBudget {
		return &routeCallerLocalBudget{rate: rate, inflight: inflight, callers: map[string]*routeCallerLocalEntry{}}
	}
	if pool == "" {
		return build()
	}
	key := pool + "/" + strconv.Itoa(rate) + "/" + strconv.Itoa(inflight)
	if existing, ok := routeCallerLocalBudgets.Load(key); ok {
		return existing.(*routeCallerLocalBudget)
	}
	actual, _ := routeCallerLocalBudgets.LoadOrStore(key, build())
	return actual.(*routeCallerLocalBudget)
}

// acquire mirrors the Redis script. It reports the verdict and whether the
// lookup holds a slot that must be released.
func (local *routeCallerLocalBudget) acquire(
	digest, token string,
	window int64,
	now, expires time.Time,
) (int, bool) {
	local.mu.Lock()
	defer local.mu.Unlock()
	entry := local.callers[digest]
	if entry == nil {
		if len(local.callers) >= routeCallerLocalMaxCallers {
			local.sweepLocked(window, now)
		}
		if len(local.callers) >= routeCallerLocalMaxCallers {
			return 0, false
		}
		entry = &routeCallerLocalEntry{window: window, holders: map[string]time.Time{}}
		local.callers[digest] = entry
	}
	if entry.window != window {
		entry.window, entry.count = window, 0
	}
	entry.count++
	if entry.count > local.rate {
		return 1, false
	}
	for holder, expiry := range entry.holders {
		if !now.Before(expiry) {
			delete(entry.holders, holder)
		}
	}
	if token == "" {
		return 0, false
	}
	if len(entry.holders) >= local.inflight {
		return 2, false
	}
	entry.holders[token] = expires
	return 0, true
}

func (local *routeCallerLocalBudget) release(digest, token string) {
	local.mu.Lock()
	defer local.mu.Unlock()
	if entry := local.callers[digest]; entry != nil {
		delete(entry.holders, token)
	}
}

// sweepLocked drops callers whose window is past and who hold nothing.
func (local *routeCallerLocalBudget) sweepLocked(window int64, now time.Time) {
	for digest, entry := range local.callers {
		for holder, expiry := range entry.holders {
			if !now.Before(expiry) {
				delete(entry.holders, holder)
			}
		}
		if entry.window != window && len(entry.holders) == 0 {
			delete(local.callers, digest)
		}
	}
}

type redisRouteCallerBudget struct {
	client *redis.Client
}

func (shared *redisRouteCallerBudget) acquire(
	ctx context.Context,
	keys routeCallerKeys,
	rate, inflight int,
	window time.Duration,
	now time.Time,
	token string,
	ttl time.Duration,
) (int, error) {
	return routeCallerAcquireScript.Run(ctx, shared.client, []string{keys.rate, keys.inflight},
		rate, window.Milliseconds(), inflight, now.UnixMilli(), token, ttl.Milliseconds()).Int()
}

func (shared *redisRouteCallerBudget) release(ctx context.Context, keys routeCallerKeys, token string) error {
	return routeCallerReleaseScript.Run(ctx, shared.client, []string{keys.inflight}, token).Err()
}

func (shared *redisRouteCallerBudget) close() error { return shared.client.Close() }
