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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A snapshot cell's readiness is lazy past its default. Readiness verifies
// the default snapshot, the package every unpinned turn decides with; every
// other row is loaded and verified on the first turn that pins it, so a
// table of many dated snapshots neither makes the store app hold every
// package resident nor delays the cell's readiness by one load per row.

const (
	// arcFailureSnapshotLoading: the pinned snapshot's package is still
	// loading and was not verified within the turn's wait. Public class
	// not_ready, 503 with Retry-After.
	arcFailureSnapshotLoading = "arc_snapshot_loading"
	// arcFailureSnapshotUnavailable: the pinned snapshot's verification
	// failed; it is refused, never served from another package, until a
	// later verification succeeds. Public class unavailable, 503.
	arcFailureSnapshotUnavailable = "arc_snapshot_unavailable"
	// snapshotLoadingRetryAfterSeconds is the Retry-After of a turn that
	// found its snapshot still loading: a cold package load takes 14-20 s.
	snapshotLoadingRetryAfterSeconds = 5
	// snapshotVerifyAttempt bounds one verification of a snapshot.
	snapshotVerifyAttempt = 2 * time.Minute
	// snapshotVerifyRetry is how long a failed verification is answered
	// unavailable before a turn pinning it verifies again.
	snapshotVerifyRetry = 30 * time.Second
)

// raylineARCSnapshotReadiness is the lazy readiness of a snapshot cell's
// non-default snapshots: one gate per row, verified once, by the first turn
// that pins it, while later turns pinning it wait on that one verification.
type raylineARCSnapshotReadiness struct {
	mu sync.Mutex
	// verify loads and checks one snapshot's package on the service, as
	// readiness checks the default's.
	verify   func(ctx context.Context, id string) error
	lifetime context.Context
	attempt  time.Duration
	retry    time.Duration
	now      func() time.Time
	gates    map[string]*raylineARCSnapshotGate
}

type raylineARCSnapshotGate struct {
	ready bool
	// loading is open while a verification runs, closed when it ends.
	loading  chan struct{}
	err      error
	failedAt time.Time
}

// newRaylineARCSnapshotReadiness gates every snapshot id but the default.
// lifetime bounds every verification: the cell's, so a disarmed cell stops
// verifying.
func newRaylineARCSnapshotReadiness(
	lifetime context.Context,
	lazy []string,
	verify func(ctx context.Context, id string) error,
) *raylineARCSnapshotReadiness {
	gates := make(map[string]*raylineARCSnapshotGate, len(lazy))
	for _, id := range lazy {
		gates[id] = &raylineARCSnapshotGate{}
	}
	return &raylineARCSnapshotReadiness{
		verify: verify, lifetime: lifetime, attempt: snapshotVerifyAttempt, retry: snapshotVerifyRetry,
		now: time.Now, gates: gates,
	}
}

// ensure answers "" once snapshot id is verified, starting its one
// verification if none runs and waiting for it up to wait (or ctx). A
// snapshot still loading answers arcFailureSnapshotLoading; one whose last
// verification failed within the retry window, or failed now, answers
// arcFailureSnapshotUnavailable. An id with no gate (the default, or a cell
// without lazy rows) is ready.
func (readiness *raylineARCSnapshotReadiness) ensure(ctx context.Context, id string, wait time.Duration) string {
	if readiness == nil {
		return ""
	}
	readiness.mu.Lock()
	gate := readiness.gates[id]
	if gate == nil || gate.ready {
		readiness.mu.Unlock()
		return ""
	}
	if gate.loading == nil {
		if gate.err != nil && readiness.now().Sub(gate.failedAt) < readiness.retry {
			readiness.mu.Unlock()
			return arcFailureSnapshotUnavailable
		}
		gate.loading, gate.err = make(chan struct{}), nil
		go readiness.run(id, gate, gate.loading)
	}
	loading := gate.loading
	readiness.mu.Unlock()
	if wait > 0 {
		timer := time.NewTimer(wait)
		select {
		case <-loading:
		case <-timer.C:
		case <-ctx.Done():
		}
		timer.Stop()
	}
	readiness.mu.Lock()
	defer readiness.mu.Unlock()
	switch {
	case gate.ready:
		return ""
	case gate.loading != nil:
		return arcFailureSnapshotLoading
	default:
		return arcFailureSnapshotUnavailable
	}
}

func (readiness *raylineARCSnapshotReadiness) run(id string, gate *raylineARCSnapshotGate, done chan struct{}) {
	ctx, cancel := context.WithTimeout(readiness.lifetime, readiness.attempt)
	defer cancel()
	started := time.Now()
	err := readiness.verify(ctx, id)
	readiness.mu.Lock()
	gate.loading = nil
	if err == nil {
		gate.ready = true
	} else {
		gate.err, gate.failedAt = err, readiness.now()
	}
	readiness.mu.Unlock()
	close(done)
	fields := map[string]interface{}{"snapshot": id, "elapsed_ms": time.Since(started).Milliseconds(), "ready": err == nil}
	if err != nil {
		fields["class"] = "transport"
		var failure *raylinearc.PolicyServiceError
		if errors.As(err, &failure) {
			fields["class"] = failure.Class
		}
		logging.ComponentErrorEvent("extproc", "rayline_arc_snapshot_verification", fields)
		return
	}
	logging.ComponentEvent("extproc", "rayline_arc_snapshot_verification", fields)
}
