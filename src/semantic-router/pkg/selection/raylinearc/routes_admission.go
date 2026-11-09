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
	// routeLookupRedisBackoff is how long the gate stops consulting Redis
	// after a Redis error. Lookups admitted meanwhile hold the per-process
	// cap only.
	routeLookupRedisBackoff = 5 * time.Second
	// routeLookupLeaseMargin is added to the lookup deadline to make the
	// lease TTL. A holder releases on return; the TTL only matters for one
	// that crashed, and must outlast any live holder.
	routeLookupLeaseMargin = 5 * time.Second
)

var (
	// Takes the first free slot, in order, in one round trip. Returns the
	// 1-based slot index, or 0 when every slot is held.
	routeLookupAcquireScript = redis.NewScript(`
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
	// LookupDeadline is the lookup's own end-to-end bound; the lease TTL is
	// derived from it.
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
}

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
		leaseTTL:  config.LookupDeadline + routeLookupLeaseMargin,
		local: admission.NewSemaphore(
			config.MaxInflight, routeLookupLocalQueue, 0, admission.OverflowShed,
		),
		pollMin: routeLookupPollMin,
		pollMax: routeLookupPollMax,
		backoff: routeLookupRedisBackoff,
		now:     time.Now,
	}
	if store := redisEpisodeStoreOf(config.Store); store != nil {
		gate.leases = newRedisRouteLookupLeases(store.client, config.Pool, config.MaxInflight)
	}
	return gate
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
		acquired, err := gate.tryLease(ctx, token)
		if err != nil {
			if ctx.Err() != nil {
				localRelease()
				return RouteLookupTicket{}, gate.refusal(ctx, RouteLookupBackendRedis, started)
			}
			gate.tripRedis(err)
			return gate.admit(started, RouteLookupBackendLocal, true, localRelease), nil
		}
		if acquired {
			var once sync.Once
			release := func() {
				once.Do(func() {
					gate.releaseLease(ctx, token)
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

// tryLease makes one bounded attempt at a slot. A call that errors may still
// have taken a slot (the reply was lost, not the write), so it releases the
// token's slot before reporting the error.
func (gate *RouteLookupGate) tryLease(ctx context.Context, token string) (bool, error) {
	callContext, cancel := context.WithTimeout(ctx, routeLookupRedisCallTimeout)
	defer cancel()
	acquired, err := gate.leases.acquire(callContext, token, gate.leaseTTL)
	if err != nil {
		gate.releaseLease(ctx, token)
		return false, err
	}
	return acquired, nil
}

// releaseLease frees the token's slot on a context detached from the
// lookup, which may already be done. A failed release is left to the TTL.
func (gate *RouteLookupGate) releaseLease(ctx context.Context, token string) {
	releaseContext, cancel := context.WithTimeout(
		context.WithoutCancel(ctx), routeLookupRedisCallTimeout,
	)
	defer cancel()
	_ = gate.leases.release(releaseContext, token)
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
	return RouteLookupTicket{Outcome: outcome, Backend: backend, Wait: wait, release: release}
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
