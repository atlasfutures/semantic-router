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

// acquire takes a lookup slot, waiting at most the configured queue time and
// never past ctx. The release function must be called exactly once. A nil
// receiver or an unconfigured gate admits at once.
func (gate *raylineRoutesLookupGate) acquire(ctx context.Context) (func(), time.Duration, error) {
	if gate == nil || gate.slots == nil {
		return func() {}, 0, nil
	}
	started := time.Now()
	ticket, err := gate.slots.Acquire(ctx)
	waited := time.Since(started)
	if err != nil {
		if errors.Is(err, admission.ErrQueueFull) {
			return nil, waited, errRaylineRoutesLookupShed
		}
		return nil, waited, err
	}
	return func() { ticket() }, waited, nil
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
		gate.slots = admission.NewSemaphore(
			settings.MaxInflightLookups,
			raylineRoutesLookupQueueDepth,
			gate.wait,
			admission.OverflowShed,
		)
	})
	return gate
}
