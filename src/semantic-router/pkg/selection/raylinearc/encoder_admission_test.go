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
	"errors"
	"strings"
	"sync"
	"testing"
)

func TestAdmissionGateShedsBeyondConfiguredLimit(t *testing.T) {
	gate := NewAdmissionGate(2)

	firstRelease, err := gate.Acquire()
	if err != nil {
		t.Fatalf("first Acquire() error = %v", err)
	}
	secondRelease, err := gate.Acquire()
	if err != nil {
		t.Fatalf("second Acquire() error = %v", err)
	}

	release, err := gate.Acquire()
	if err == nil {
		t.Fatalf("third Acquire() succeeded past the configured limit of 2")
	}
	if release != nil {
		t.Fatalf("a shed Acquire() must not hand back a release function")
	}
	if gate.Inflight() != 2 {
		t.Fatalf("Inflight() = %d, want 2", gate.Inflight())
	}

	firstRelease()
	secondRelease()
	if gate.Inflight() != 0 {
		t.Fatalf("Inflight() after release = %d, want 0", gate.Inflight())
	}
}

// The shed must never look like an encoder error. Selection maps failures by
// EncoderFailure.Class, so a distinct class is what keeps shed rate out of the
// encoder failure rate.
func TestAdmissionGateShedCarriesItsOwnBoundedFailureClass(t *testing.T) {
	gate := NewAdmissionGate(1)
	release, err := gate.Acquire()
	if err != nil {
		t.Fatalf("Acquire() error = %v", err)
	}
	defer release()

	_, shedErr := gate.Acquire()
	var failure *EncoderFailure
	if !errors.As(shedErr, &failure) {
		t.Fatalf("shed error = %T, want *EncoderFailure", shedErr)
	}
	if failure.Class != EncoderFailureAdmission {
		t.Fatalf("Class = %q, want %q", failure.Class, EncoderFailureAdmission)
	}
	for _, reserved := range []EncoderFailureClass{
		EncoderFailureRequest,
		EncoderFailureTransport,
		EncoderFailureTimeout,
		EncoderFailureStatus,
		EncoderFailureDecode,
		EncoderFailureContract,
	} {
		if failure.Class == reserved {
			t.Fatalf("admission shed reused encoder failure class %q", reserved)
		}
	}
	if failure.StatusCode != 0 {
		t.Fatalf("StatusCode = %d, want 0: no encoder call was made", failure.StatusCode)
	}
	// The gate never sees request text or episode identity, so its message
	// must stay a bounded class and stage string.
	if message := failure.Error(); !strings.Contains(message, "class=admission") {
		t.Fatalf("Error() = %q, want a bounded admission class", message)
	}
}

func TestAdmissionGateReusesReleasedSlots(t *testing.T) {
	gate := NewAdmissionGate(1)
	for attempt := range 5 {
		release, err := gate.Acquire()
		if err != nil {
			t.Fatalf("Acquire() attempt %d error = %v", attempt, err)
		}
		release()
	}
	if gate.Inflight() != 0 {
		t.Fatalf("Inflight() = %d, want 0", gate.Inflight())
	}
	if gate.HighWater() != 1 {
		t.Fatalf("HighWater() = %d, want 1", gate.HighWater())
	}
}

// Release runs from a defer on a path that can panic. A repeated release must
// not corrupt the counter, or the gate would slowly leak capacity.
func TestAdmissionGateToleratesRepeatedRelease(t *testing.T) {
	gate := NewAdmissionGate(1)
	release, err := gate.Acquire()
	if err != nil {
		t.Fatalf("Acquire() error = %v", err)
	}
	release()
	release()
	release()
	if gate.Inflight() != 0 {
		t.Fatalf("Inflight() = %d, want 0", gate.Inflight())
	}
	if _, err := gate.Acquire(); err != nil {
		t.Fatalf("Acquire() after repeated release error = %v", err)
	}
}

// A zero or negative cap means admission control is off, not that every
// request is shed. An unset integer in a config file must never start
// rejecting traffic.
func TestAdmissionGateDisabledWhenLimitIsNotPositive(t *testing.T) {
	for _, limit := range []int{0, -1, -32} {
		gate := NewAdmissionGate(limit)
		if gate != nil {
			t.Fatalf("NewAdmissionGate(%d) = %v, want nil", limit, gate)
		}
		if gate.Limit() != 0 {
			t.Fatalf("Limit() = %d, want 0", gate.Limit())
		}
		for attempt := range 64 {
			release, err := gate.Acquire()
			if err != nil {
				t.Fatalf("disabled Acquire() attempt %d error = %v", attempt, err)
			}
			if release == nil {
				t.Fatalf("disabled Acquire() returned a nil release function")
			}
			defer release()
		}
		if gate.Inflight() != 0 || gate.HighWater() != 0 {
			t.Fatalf("a disabled gate must not report occupancy")
		}
	}
}

func TestAdmissionGateNeverExceedsLimitUnderConcurrency(t *testing.T) {
	const (
		limit   = 4
		callers = 128
	)
	gate := NewAdmissionGate(limit)

	var (
		wait     sync.WaitGroup
		mu       sync.Mutex
		admitted int
		shed     int
	)
	// Hold every admitted slot until all callers have tried, so the observed
	// concurrency is real rather than an artefact of fast release.
	gateOpen := make(chan struct{})
	tried := make(chan struct{}, callers)

	wait.Add(callers)
	for range callers {
		go func() {
			defer wait.Done()
			release, err := gate.Acquire()
			mu.Lock()
			if err != nil {
				shed++
			} else {
				admitted++
			}
			mu.Unlock()
			tried <- struct{}{}
			if err == nil {
				<-gateOpen
				release()
			}
		}()
	}
	for range callers {
		<-tried
	}
	if gate.Inflight() > limit {
		t.Fatalf("Inflight() = %d, exceeds limit %d", gate.Inflight(), limit)
	}
	close(gateOpen)
	wait.Wait()

	if admitted != limit {
		t.Fatalf("admitted = %d, want exactly the limit %d", admitted, limit)
	}
	if shed != callers-limit {
		t.Fatalf("shed = %d, want %d", shed, callers-limit)
	}
	// admitted + shed is the offered rate. It is the whole reason the gate
	// carries counters, so it must account for every caller.
	if admitted+shed != callers {
		t.Fatalf("admitted+shed = %d, want %d", admitted+shed, callers)
	}
	if gate.HighWater() != limit {
		t.Fatalf("HighWater() = %d, want %d", gate.HighWater(), limit)
	}
	if gate.Inflight() != 0 {
		t.Fatalf("Inflight() after drain = %d, want 0", gate.Inflight())
	}
}

// High water only ever rises, so an operator can read it once and know whether
// the cap has ever bound.
func TestAdmissionGateHighWaterOnlyRises(t *testing.T) {
	gate := NewAdmissionGate(3)
	firstRelease, _ := gate.Acquire()
	secondRelease, _ := gate.Acquire()
	thirdRelease, _ := gate.Acquire()
	if gate.HighWater() != 3 {
		t.Fatalf("HighWater() = %d, want 3", gate.HighWater())
	}
	firstRelease()
	secondRelease()
	thirdRelease()
	if gate.HighWater() != 3 {
		t.Fatalf("HighWater() fell to %d after release", gate.HighWater())
	}
	if gate.Limit() != 3 {
		t.Fatalf("Limit() = %d, want 3", gate.Limit())
	}
}
