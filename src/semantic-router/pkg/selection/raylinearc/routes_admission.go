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
	"math/rand/v2"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
)

// ErrRouteLookupShed reports a lookup that found no admission slot within its
// queue wait. It is a capacity answer (429), never a failure of the lookup.
var ErrRouteLookupShed = errors.New("route lookup admission: no slot within the queue wait")

// Route lookup admission outcomes, as recorded by the caller.
const (
	RouteLookupAdmitted  = "admitted"
	RouteLookupQueued    = "queued"
	RouteLookupShed      = "shed"
	RouteLookupCancelled = "cancelled"
	RouteLookupFailOpen  = "fail_open"
)

// Route lookup gate backends.
const (
	RouteLookupBackendRedis = "redis"
	RouteLookupBackendLocal = "local"
)

const (
	// routeLookupKeyNamespace is not under an episode key_prefix on purpose:
	// the slots protect the backend, not a recipe's episodes, so two recipes
	// that call one backend through one Redis share them.
	routeLookupKeyNamespace = "rayline-arc:routes-admission:"
	// routeLookupLocalQueue bounds how many lookups one process parks at the
	// gate. Past it a lookup is shed at once rather than queued.
	routeLookupLocalQueue = 256
	routeLookupPollMin    = 25 * time.Millisecond
	// routeLookupQueuedAfter separates a lookup that found a slot free from
	// one that waited for it: a free slot costs one Redis round trip, a
	// waited one at least a poll interval.
	routeLookupQueuedAfter = 20 * time.Millisecond
	routeLookupPollMax     = 200 * time.Millisecond
	// routeLookupRedisCallTimeout bounds one Redis round trip, so that a
	// stalled store costs a lookup this much once and not its queue wait.
	routeLookupRedisCallTimeout = 250 * time.Millisecond
	// routeLookupRedisMinCall is the least time one call is given, so a
	// call that fails within it is a Redis failure. A lookup may overrun its
	// queue wait by at most this much.
	routeLookupRedisMinCall = 50 * time.Millisecond
	// routeLookupRedisPoolSize bounds the gate's own connections. A process
	// has at most max_inflight lookups polling (the rest queue locally), so
	// a handful is plenty.
	routeLookupRedisPoolSize = 4
	// routeLookupRedisBackoff is how long the gate stops consulting Redis
	// after a Redis error. Lookups admitted meanwhile hold the per-process
	// cap only.
	routeLookupRedisBackoff = 5 * time.Second
	// routeLookupLeaseMargin is added to the longest a holder can live to
	// make the lease TTL. A holder releases on return; the TTL only matters
	// for one that crashed, and must outlast any live holder.
	routeLookupLeaseMargin = 5 * time.Second
)

// RouteLookupDetachBound is the longest an admitted lookup's decision may run
// after its caller stopped waiting.
//
// The backend keeps encoding after the lookup's deadline answers 504: the
// policy service has no cancel, and a large transcript encodes for longer than
// any deadline a caller waits through. A slot freed at the 504 would admit the
// next lookup onto a backend that is still busy, so the handler runs an
// admitted ephemeral decision on its own context, bounded here, and releases
// the slot when the backend answers. The lease TTL outlasts this bound.
const RouteLookupDetachBound = 120 * time.Second

var (
	// Takes the first free slot, in order, in one round trip. Returns the
	// 1-based slot index, or 0 when every slot is held. Idempotent per
	// token: a slot this token already holds is returned as it is, so a
	// replayed call (a reply lost after the write) neither reports the
	// token's own slot as busy nor takes a second one.
	routeLookupAcquireScript = redis.NewScript(`
for index, key in ipairs(KEYS) do
  if redis.call("GET", key) == ARGV[1] then return index end
end
for index, key in ipairs(KEYS) do
  if redis.call("SET", key, ARGV[1], "NX", "PX", ARGV[2]) then return index end
end
return 0
`)
	// Releases whatever slot this token holds, and only that: a slot whose
	// lease expired and was taken by another lookup is left alone.
	routeLookupReleaseScript = redis.NewScript(`
local released = 0
for _, key in ipairs(KEYS) do
  if redis.call("GET", key) == ARGV[1] then
    redis.call("DEL", key)
    released = released + 1
  end
end
return released
`)
)

// RouteLookupGateConfig configures a route lookup admission gate.
type RouteLookupGateConfig struct {
	// MaxInflight is the number of lookups admitted at once. Zero or less
	// disables the gate.
	MaxInflight int
	// QueueWait is how long a lookup waits for a slot before it is shed.
	QueueWait time.Duration
	// LookupDeadline is the lookup's own end-to-end bound. The lease TTL
	// covers the longer of it and RouteLookupDetachBound.
	LookupDeadline time.Duration
	// Pool names the backend the slots protect. Every gate built with the
	// same pool on the same Redis shares the same slots.
	Pool string
	// Store is the recipe's episode store. A Redis store makes the gate
	// cell-wide over its client; anything else leaves it per process.
	Store EpisodeStore
}

// routeLookupLeases is the shared half of the gate.
type routeLookupLeases interface {
	acquire(ctx context.Context, token string, ttl time.Duration) (bool, error)
	release(ctx context.Context, token string) error
}

// RouteLookupGate admits at most MaxInflight route lookups at once.
//
// It has two stages. The first is a process-local semaphore with MaxInflight
// slots: it bounds this process on its own and queues this process's waiters
// without touching Redis. The second, when the episode store is Redis, is one
// of MaxInflight cell-wide leases. A lookup holds both until it answers.
//
// Redis failures fail open to the first stage. The episode store is the same
// Redis, so an outage already degrades the cell; refusing every lookup on top
// would turn a store blip into a lookup outage, while admitting on the local
// cap still bounds the cell at MaxInflight per instance. The gate stops
// consulting Redis for a short backoff after an error, so a dead store does
// not cost every lookup a round-trip timeout.
//
// A nil gate is a disabled gate; Acquire on it admits at once.
type RouteLookupGate struct {
	limit     int
	queueWait time.Duration
	leaseTTL  time.Duration
	local     *admission.Semaphore
	leases    routeLookupLeases
	pollMin   time.Duration
	pollMax   time.Duration
	backoff   time.Duration
	// redisDownUntil is a UnixNano instant; until then the gate does not
	// consult Redis.
	redisDownUntil atomic.Int64
	now            func() time.Time
	// releases tracks background lease releases, so tests (and Close) can
	// wait for them.
	releases sync.WaitGroup
	// onFailOpen is told when a Redis error trips the backoff, so the caller
	// can log it once per trip rather than once per lookup.
	onFailOpen func(error)
}

// RouteLookupTicket is an admitted lookup's hold on the gate.
type RouteLookupTicket struct {
	// Outcome is RouteLookupAdmitted, RouteLookupQueued or RouteLookupFailOpen.
	Outcome string
	// Backend is the gate stage that bounded this lookup.
	Backend string
	// Wait is how long the lookup waited for its slot.
	Wait    time.Duration
	release func()
	// leaseExpires is when the slot's cell-wide lease runs out, for a slot
	// that has one (the redis backend); zero otherwise. Taken once the
	// lease was granted, so it is at most one Redis call late.
	leaseExpires time.Time
}

// LeaseRemaining is how long the slot's cell-wide lease has left at now, and
// whether it has one: a per-process slot never expires.
func (ticket RouteLookupTicket) LeaseRemaining(now time.Time) (time.Duration, bool) {
	if ticket.leaseExpires.IsZero() {
		return 0, false
	}
	return ticket.leaseExpires.Sub(now), true
}

// Held reports whether the ticket holds a slot, so Release frees something.
func (ticket RouteLookupTicket) Held() bool { return ticket.release != nil }

// Release frees the lookup's slot. It is safe to call more than once and on
// a zero ticket.
func (ticket RouteLookupTicket) Release() {
	if ticket.release != nil {
		ticket.release()
	}
}

// RouteLookupRefusal is a lookup the gate did not admit.
type RouteLookupRefusal struct {
	// Outcome is RouteLookupShed or RouteLookupCancelled.
	Outcome string
	Backend string
	Wait    time.Duration
	Err     error
}

func (refusal *RouteLookupRefusal) Error() string { return refusal.Err.Error() }
func (refusal *RouteLookupRefusal) Unwrap() error { return refusal.Err }

// NewRouteLookupGate builds the gate, or returns nil when MaxInflight
// disables it.
func NewRouteLookupGate(config RouteLookupGateConfig) *RouteLookupGate {
	if config.MaxInflight <= 0 {
		return nil
	}
	gate := &RouteLookupGate{
		limit:     config.MaxInflight,
		queueWait: config.QueueWait,
		leaseTTL:  max(config.LookupDeadline, RouteLookupDetachBound) + routeLookupLeaseMargin,
		local:     routeLookupLocalSlots(config.Pool, config.MaxInflight),
		pollMin:   routeLookupPollMin,
		pollMax:   routeLookupPollMax,
		backoff:   routeLookupRedisBackoff,
		now:       time.Now,
	}
	if store := redisEpisodeStoreOf(config.Store); store != nil {
		gate.leases = newRedisRouteLookupLeases(routeLookupRedisClient(store.client), config.Pool, config.MaxInflight)
	}
	return gate
}

// routeLookupRedisClient opens the gate's own small pool to the episode
// store's Redis, with the store's address, credentials and TLS.
//
// It is not the store's client because that one is configured for the
// episode lease: socket timeouts of seconds, context deadlines not applied
// to socket I/O, and automatic retries. On the gate those would let a stalled
// Redis hold a lookup past its queue wait (or its deadline) before failing
// open, and a retried acquire could replay a write whose reply was lost.
// Here every call is bounded by its context and by short socket timeouts,
// and nothing is retried: the poll loop is the retry.
func routeLookupRedisClient(shared *redis.Client) *redis.Client {
	options := *shared.Options()
	options.ContextTimeoutEnabled = true
	options.DialTimeout = routeLookupRedisCallTimeout
	options.ReadTimeout = routeLookupRedisCallTimeout
	options.WriteTimeout = routeLookupRedisCallTimeout
	options.MaxRetries = -1
	options.PoolSize = routeLookupRedisPoolSize
	options.MinIdleConns = 0
	return redis.NewClient(&options)
}

// Close releases the gate's Redis connections. Lookups still holding a slot
// are left to their lease TTL.
func (gate *RouteLookupGate) Close() error {
	if gate == nil || gate.leases == nil {
		return nil
	}
	gate.releases.Wait()
	if closer, ok := gate.leases.(interface{ close() error }); ok {
		return closer.close()
	}
	return nil
}

// routeLookupLocalSemaphores holds one per-process semaphore per pool and
// limit, so every recipe and every router generation in the process that
// calls one backend shares the local cap, the way they share its Redis
// slots. Entries are never removed: there is one per configured backend.
var routeLookupLocalSemaphores sync.Map

// routeLookupLocalSlots is the process's local semaphore for pool. A gate
// with no pool (tests, a backend with no identity) gets its own.
func routeLookupLocalSlots(pool string, limit int) *admission.Semaphore {
	build := func() *admission.Semaphore {
		return admission.NewSemaphore(limit, routeLookupLocalQueue, 0, admission.OverflowShed)
	}
	if pool == "" {
		return build()
	}
	key := pool + "/" + strconv.Itoa(limit)
	if existing, ok := routeLookupLocalSemaphores.Load(key); ok {
		return existing.(*admission.Semaphore)
	}
	actual, _ := routeLookupLocalSemaphores.LoadOrStore(key, build())
	return actual.(*admission.Semaphore)
}

// redisEpisodeStoreOf unwraps the Redis episode store a recipe runs on, or
// nil when it runs on another backend.
func redisEpisodeStoreOf(store EpisodeStore) *RedisEpisodeStore {
	for store != nil {
		switch typed := store.(type) {
		case *RedisEpisodeStore:
			if typed == nil || typed.client == nil {
				return nil
			}
			return typed
		case interface{ UnwrapEpisodeStore() EpisodeStore }:
			store = typed.UnwrapEpisodeStore()
		default:
			return nil
		}
	}
	return nil
}

// Distributed reports whether the gate holds cell-wide leases.
func (gate *RouteLookupGate) Distributed() bool {
	return gate != nil && gate.leases != nil
}

// Limit is the configured cap, or zero for a disabled gate.
func (gate *RouteLookupGate) Limit() int {
	if gate == nil {
		return 0
	}
	return gate.limit
}

// QueueWait is the configured wait.
func (gate *RouteLookupGate) QueueWait() time.Duration {
	if gate == nil {
		return 0
	}
	return gate.queueWait
}

// SetFailOpenObserver installs the callback told about each Redis error that
// opens the gate. Call it before the gate serves.
func (gate *RouteLookupGate) SetFailOpenObserver(observe func(error)) {
	if gate != nil {
		gate.onFailOpen = observe
	}
}

// Acquire waits up to the queue wait for a slot. ctx is the lookup's own
// context: its cancellation or deadline ends the wait at once.
//
// On success the ticket must be released when the lookup answers. On refusal
// the error is a *RouteLookupRefusal wrapping ErrRouteLookupShed when the
// queue wait ran out, or ctx's error when ctx ended first.
func (gate *RouteLookupGate) Acquire(ctx context.Context) (RouteLookupTicket, error) {
	if gate == nil {
		return RouteLookupTicket{Outcome: RouteLookupAdmitted, Backend: RouteLookupBackendLocal}, nil
	}
	started := gate.now()
	waitContext, cancel := context.WithTimeout(ctx, gate.queueWait)
	defer cancel()

	localRelease, err := gate.local.Acquire(waitContext)
	if err != nil {
		return RouteLookupTicket{}, gate.refusal(ctx, RouteLookupBackendLocal, started)
	}
	if gate.leases == nil || gate.redisBackingOff() {
		return gate.admit(started, RouteLookupBackendLocal, gate.leases != nil, localRelease), nil
	}

	token, err := newEpisodeOwnerToken()
	if err != nil {
		// No token, no lease to take; the local cap still holds.
		gate.tripRedis(err)
		return gate.admit(started, RouteLookupBackendLocal, true, localRelease), nil
	}
	delay := gate.pollMin
	for {
		acquired, err := gate.tryLease(ctx, waitContext, token)
		if err != nil {
			if ctx.Err() != nil {
				// The lookup itself ended during the call.
				localRelease()
				return RouteLookupTicket{}, gate.refusal(ctx, RouteLookupBackendRedis, started)
			}
			// A healthy Redis answers in about a millisecond, and every call
			// is given at least routeLookupRedisMinCall, so an error here is
			// a Redis failure even when the queue wait also ran out.
			gate.tripRedis(err)
			if waitContext.Err() != nil {
				localRelease()
				return RouteLookupTicket{}, gate.refusal(ctx, RouteLookupBackendRedis, started)
			}
			return gate.admit(started, RouteLookupBackendLocal, true, localRelease), nil
		}
		if acquired {
			if ctx.Err() != nil {
				gate.releaseLeaseAsync(ctx, token)
				localRelease()
				return RouteLookupTicket{}, gate.refusal(ctx, RouteLookupBackendRedis, started)
			}
			var once sync.Once
			release := func() {
				once.Do(func() {
					// The Redis release runs off the response path; the
					// local slot frees at once. Until the delete lands the
					// next local waiter just polls a slot still held.
					gate.releaseLeaseAsync(ctx, token)
					localRelease()
				})
			}
			return gate.admit(started, RouteLookupBackendRedis, false, release), nil
		}
		timer := time.NewTimer(jitterRouteLookupPoll(delay))
		select {
		case <-waitContext.Done():
			timer.Stop()
			localRelease()
			return RouteLookupTicket{}, gate.refusal(ctx, RouteLookupBackendRedis, started)
		case <-timer.C:
		}
		delay = min(2*delay, gate.pollMax)
	}
}

// tryLease makes one attempt at a slot, bounded by what is left of the queue
// wait -- but never by less than routeLookupRedisMinCall, so that a call
// that fails is a Redis that did not answer in time, not a wait that ran out
// as the call began -- and never by more than one call timeout or the
// lookup's own context. A call that errors may still have taken a slot (the
// reply was lost, not the write), so the token's slot is released, off the
// lookup's path, before the error is reported.
func (gate *RouteLookupGate) tryLease(ctx, waitContext context.Context, token string) (bool, error) {
	budget := routeLookupRedisCallTimeout
	if deadline, ok := waitContext.Deadline(); ok {
		budget = min(max(time.Until(deadline), routeLookupRedisMinCall), routeLookupRedisCallTimeout)
	}
	callContext, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	acquired, err := gate.leases.acquire(callContext, token, gate.leaseTTL)
	if err != nil {
		gate.releaseLeaseAsync(ctx, token)
		return false, err
	}
	return acquired, nil
}

// releaseLeaseAsync frees the token's slot in the background, on a context
// detached from the lookup (which may be done) and bounded by one call
// timeout. The release compares the token, so a late one never frees a slot
// another lookup took after this lease expired. A failed release is left to
// the TTL.
func (gate *RouteLookupGate) releaseLeaseAsync(ctx context.Context, token string) {
	releaseContext, cancel := context.WithTimeout(
		context.WithoutCancel(ctx), routeLookupRedisCallTimeout,
	)
	gate.releases.Add(1)
	go func() {
		defer gate.releases.Done()
		defer cancel()
		_ = gate.leases.release(releaseContext, token)
	}()
}

func (gate *RouteLookupGate) admit(
	started time.Time,
	backend string,
	failOpen bool,
	release func(),
) RouteLookupTicket {
	wait := gate.now().Sub(started)
	outcome := RouteLookupAdmitted
	switch {
	case failOpen:
		outcome = RouteLookupFailOpen
	case wait >= routeLookupQueuedAfter:
		outcome = RouteLookupQueued
	}
	ticket := RouteLookupTicket{Outcome: outcome, Backend: backend, Wait: wait, release: release}
	if backend == RouteLookupBackendRedis && !failOpen {
		ticket.leaseExpires = gate.now().Add(gate.leaseTTL)
	}
	return ticket
}

// refusal says why the wait ended. ctx is the lookup's own context; when it
// ended the wait the lookup was cancelled or ran out of deadline, which is
// not a capacity answer.
func (gate *RouteLookupGate) refusal(ctx context.Context, backend string, started time.Time) error {
	refusal := &RouteLookupRefusal{
		Outcome: RouteLookupShed,
		Backend: backend,
		Wait:    gate.now().Sub(started),
		Err:     ErrRouteLookupShed,
	}
	if err := ctx.Err(); err != nil {
		refusal.Outcome = RouteLookupCancelled
		refusal.Err = err
	}
	return refusal
}

func (gate *RouteLookupGate) redisBackingOff() bool {
	return gate.now().UnixNano() < gate.redisDownUntil.Load()
}

// tripRedis starts the backoff. Only the lookup that trips it reports it.
func (gate *RouteLookupGate) tripRedis(err error) {
	until := gate.now().Add(gate.backoff).UnixNano()
	for {
		previous := gate.redisDownUntil.Load()
		if previous >= until {
			return
		}
		if gate.redisDownUntil.CompareAndSwap(previous, until) {
			if gate.onFailOpen != nil && previous < gate.now().UnixNano() {
				gate.onFailOpen(err)
			}
			return
		}
	}
}

func jitterRouteLookupPoll(delay time.Duration) time.Duration {
	// 50..100% of the delay, so instances polling one slot do not align.
	half := delay / 2
	if half <= 0 {
		return delay
	}
	return half + rand.N(half+1)
}

// RouteLookupPoolID derives the pool name for a backend from its URLs, so
// every gate that calls the same backend computes the same slots. The hash
// keeps URLs (which may carry deployment detail) out of Redis key names.
func RouteLookupPoolID(urls ...string) string {
	digest := sha256.New()
	for _, url := range urls {
		// The clients call a base URL with its trailing slashes trimmed, so
		// two spellings of one backend are one pool.
		url = strings.TrimRight(strings.TrimSpace(url), "/")
		digest.Write([]byte(strconv.Itoa(len(url))))
		digest.Write([]byte{':'})
		digest.Write([]byte(url))
	}
	return hex.EncodeToString(digest.Sum(nil))[:16]
}

type redisRouteLookupLeases struct {
	client *redis.Client
	keys   []string
}

func newRedisRouteLookupLeases(client *redis.Client, pool string, slots int) *redisRouteLookupLeases {
	keys := make([]string, slots)
	for index := range keys {
		keys[index] = routeLookupKeyNamespace + pool + ":slot:" + strconv.Itoa(index)
	}
	return &redisRouteLookupLeases{client: client, keys: keys}
}

func (leases *redisRouteLookupLeases) acquire(
	ctx context.Context,
	token string,
	ttl time.Duration,
) (bool, error) {
	slot, err := routeLookupAcquireScript.Run(ctx, leases.client, leases.keys, token, ttl.Milliseconds()).Int()
	if err != nil {
		return false, err
	}
	return slot > 0, nil
}

func (leases *redisRouteLookupLeases) release(ctx context.Context, token string) error {
	return routeLookupReleaseScript.Run(ctx, leases.client, leases.keys, token).Err()
}

func (leases *redisRouteLookupLeases) close() error {
	return leases.client.Close()
}
