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

package extproc

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// raylineRoutesLookupShedRetrySeconds is the Retry-After on a lookup shed by
// the lookup gate. Longer than the contended 1 s, because the slot it waited
// on is held by a whole-transcript encode, which runs for seconds to tens of
// seconds; telling the caller 1 s would only bring it back into the same
// queue.
const raylineRoutesLookupShedRetrySeconds = 5

// raylineRoutesLookupQueueDepth bounds how many lookups may wait for a slot at
// once. Each waiter holds its parsed body, which can be megabytes, so a burst
// beyond this is shed at once rather than queued.
const raylineRoutesLookupQueueDepth = 4

// raylineRoutesLookupDecisionMargin is the part of a lookup's deadline the
// slot wait never takes, so an admitted lookup still has time to decide and
// a wait that runs out answers 429 rather than racing the deadline's 504.
const raylineRoutesLookupDecisionMargin = 500 * time.Millisecond

// raylineRoutesAbandonedLookupCeiling bounds how long an abandoned lookup
// keeps its slot while the policy service finishes the encode. Above the
// slowest whole-transcript encode measured (about 45 s for a 1M-token
// rebuild), and far below the encoder's own total_timeout_seconds.
const raylineRoutesAbandonedLookupCeiling = 120 * time.Second

// errRaylineRoutesLookupShed reports that no lookup slot freed up within the
// bounded wait.
var errRaylineRoutesLookupShed = errors.New("route lookup shed: another lookup holds this cell's lookup slot")

// raylineRoutesLookupGate bounds the lookups one router process has in flight
// against the policy service. Live turns never pass through it.
type raylineRoutesLookupGate struct {
	once  sync.Once
	slots *admission.Semaphore // nil: unbounded
	wait  time.Duration
}

// enabled reports whether the gate bounds anything.
func (gate *raylineRoutesLookupGate) enabled() bool {
	return gate != nil && gate.slots != nil
}

// acquire takes a lookup slot. It waits at most the configured queue time,
// and never into the last raylineRoutesLookupDecisionMargin of ctx's
// deadline. A wait that runs out is errRaylineRoutesLookupShed; ctx ending
// first returns ctx's own error. The release function must be called exactly
// once. An unconfigured gate admits at once.
func (gate *raylineRoutesLookupGate) acquire(ctx context.Context) (func(), time.Duration, error) {
	if !gate.enabled() {
		return func() {}, 0, nil
	}
	wait := gate.wait
	if deadline, ok := ctx.Deadline(); ok {
		if remaining := time.Until(deadline) - raylineRoutesLookupDecisionMargin; remaining < wait {
			wait = remaining
		}
	}
	started := time.Now()
	if wait <= 0 {
		// No time to queue: admit only a free slot.
		wait = time.Nanosecond
	}
	waitContext, cancel := context.WithTimeout(ctx, wait)
	defer cancel()
	ticket, err := gate.slots.Acquire(waitContext)
	waited := time.Since(started)
	if err == nil {
		return func() { ticket() }, waited, nil
	}
	if ctx.Err() != nil {
		return nil, waited, ctx.Err()
	}
	if errors.Is(err, admission.ErrQueueFull) || errors.Is(err, context.DeadlineExceeded) {
		return nil, waited, errRaylineRoutesLookupShed
	}
	return nil, waited, err
}

// decide runs one admitted lookup's decision and releases the slot when the
// policy service has answered, not when the caller stops waiting.
//
// The policy container does not stop an encode whose caller has gone, so a
// slot released at the lookup's deadline would admit the next lookup onto a
// container still busy with this one. The decision therefore runs on a
// context detached from the caller's cancellation (bounded by
// raylineRoutesAbandonedLookupCeiling), and the caller gets lookupContext's
// error as soon as lookupContext ends. Only ephemeral lookups are detached:
// a tracked lookup commits its episode, and one its caller abandoned must not.
func (gate *raylineRoutesLookupGate) decide(
	lookupContext context.Context,
	release func(),
	ephemeral bool,
	run func(context.Context) (routerruntime.RouteDecision, error),
) (routerruntime.RouteDecision, error) {
	if !gate.enabled() || !ephemeral {
		defer release()
		return run(lookupContext)
	}
	type outcome struct {
		decision routerruntime.RouteDecision
		err      error
	}
	done := make(chan outcome, 1)
	decisionContext, cancel := context.WithTimeout(context.WithoutCancel(lookupContext), raylineRoutesAbandonedLookupCeiling)
	go func() {
		defer cancel()
		defer release()
		decision, err := run(decisionContext)
		done <- outcome{decision, err}
	}()
	select {
	case result := <-done:
		return result.decision, result.err
	case <-lookupContext.Done():
		return routerruntime.RouteDecision{}, lookupContext.Err()
	}
}

// raylineRoutesLookups returns this process's lookup gate, built on first use
// from the default recipe's routes_api. Every recipe's lookups share it,
// because every recipe's lookups share the encoder.
func (r *OpenAIRouter) raylineRoutesLookups() *raylineRoutesLookupGate {
	gate := &r.raylineRoutesLookupGate
	gate.once.Do(func() {
		settings := r.raylineRoutesConfig()
		if settings == nil || settings.MaxInflightLookups <= 0 {
			return
		}
		gate.wait = settings.EffectiveLookupQueue()
		// The wait is bounded by acquire's own context, so the semaphore
		// needs no timer of its own.
		gate.slots = admission.NewSemaphore(
			settings.MaxInflightLookups,
			raylineRoutesLookupQueueDepth,
			0,
			admission.OverflowShed,
		)
	})
	return gate
}
