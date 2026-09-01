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
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
)

// EncoderFailureAdmission marks a decision the router refused to send to the
// encoder because the configured in-flight limit was already reached.
//
// This is a deliberate capacity decision, never an encoder error. It keeps its
// own class so that a shed rate can never be read as an encoder failure rate,
// and so that the API answers a shed with 429 and a Retry-After rather than
// the 503 every real encoder failure earns.
const EncoderFailureAdmission EncoderFailureClass = "admission"

const admissionStage = "admission"

// AdmissionGate bounds how many encoder calls one router process may have in
// flight at the same time.
//
// The gate itself is the fixed-slot semaphore in pkg/admission, configured
// with no wait queue so that a caller which cannot get a slot is shed on the
// spot. A bounded wait would absorb microbursts but would also blur the shed
// counter, which is the measurement this gate exists to produce. Add a wait
// only if the recorded shed rate shows microbursts matter.
//
// This type adds the two things the semaphore does not carry: the bounded
// *EncoderFailure a shed must answer with, and the occupancy counters an
// operator reads to see whether the cap has ever bound.
//
// The gate is deliberately process-local and holds no distributed state. ARC
// episode state is already process-local, so a deployment that serves retained
// sessions runs exactly one router instance. Under that constraint a plain
// counter is an exact global gate. Revisit this the moment more than one
// router instance can serve the same episodes.
//
// A nil gate is a disabled gate, and Acquire always succeeds on one. That
// keeps the call site free of a "is the gate configured" branch.
type AdmissionGate struct {
	slots     *admission.Semaphore
	limit     int
	inflight  atomic.Int64
	highWater atomic.Int64
}

// NewAdmissionGate returns a gate that admits at most limit concurrent encoder
// calls. A limit of zero or less disables admission control and returns nil,
// which every method on this type accepts.
func NewAdmissionGate(limit int) *AdmissionGate {
	if limit <= 0 {
		return nil
	}
	return &AdmissionGate{
		// No queue and no queue timeout: overflow sheds immediately.
		slots: admission.NewSemaphore(limit, 0, 0, admission.OverflowShed),
		limit: limit,
	}
}

// Acquire takes one in-flight slot without waiting.
//
// On success it returns a release function that the caller must invoke exactly
// once, normally through defer, so that a panic on the encoder path cannot
// strand a slot. Releasing more than once is harmless.
//
// On refusal it returns a bounded *EncoderFailure carrying
// EncoderFailureAdmission, so the existing encoder failure plumbing maps a
// shed to its own class with no extra wiring.
func (gate *AdmissionGate) Acquire() (func(), error) {
	if gate == nil {
		return func() {}, nil
	}
	// The gate never waits, so it needs no deadline of its own. Passing the
	// caller's context would only let a cancelled request report a
	// cancellation where the answer is always immediate.
	ticket, err := gate.slots.Acquire(context.Background())
	if err != nil {
		if !errors.Is(err, admission.ErrQueueFull) {
			return nil, err
		}
		return nil, &EncoderFailure{
			Class: EncoderFailureAdmission,
			Stage: admissionStage,
		}
	}
	gate.raiseHighWater(gate.inflight.Add(1))
	var released sync.Once
	return func() {
		released.Do(func() {
			gate.inflight.Add(-1)
			ticket()
		})
	}, nil
}

// Inflight reports how many encoder calls hold a slot right now.
func (gate *AdmissionGate) Inflight() int {
	if gate == nil {
		return 0
	}
	return int(gate.inflight.Load())
}

// HighWater reports the largest in-flight count seen since start. It only ever
// rises. Compare it against Limit to see whether the cap has ever bound.
func (gate *AdmissionGate) HighWater() int {
	if gate == nil {
		return 0
	}
	return int(gate.highWater.Load())
}

// Limit reports the configured cap, or zero when admission control is off.
func (gate *AdmissionGate) Limit() int {
	if gate == nil {
		return 0
	}
	return gate.limit
}

func (gate *AdmissionGate) raiseHighWater(current int64) {
	for {
		previous := gate.highWater.Load()
		if current <= previous {
			return
		}
		if gate.highWater.CompareAndSwap(previous, current) {
			return
		}
	}
}
