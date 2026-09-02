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
	"reflect"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// A transient encoder failure at startup used to pin the instance at ready=0
// for its whole lifetime, because the readiness probe ran exactly once. The
// recovery loop must keep probing on a bounded schedule and arm the selector
// the moment the encoder answers.
func TestRaylineARCReadinessRecoversFromTransientProbeFailures(t *testing.T) {
	setRaylineARCEncoderReadyGauge(t, false)
	probe := &fakeARCProbe{failures: 6}
	clock := &fakeARCProbeClock{}
	selector := newRaylineARCSelector(nil, nil, nil, "artifact-revision")

	recovered := raylineARCRecoverReadiness(
		context.Background(),
		selector,
		armedARCComponentsFixture(),
		probe.run,
		raylineARCProbeBackoff{
			initial: 5 * time.Second,
			max:     60 * time.Second,
		},
		clock.wait,
	)

	if !recovered {
		t.Fatal("readiness recovery gave up on a recoverable encoder")
	}
	wantSchedule := []time.Duration{
		5 * time.Second,
		10 * time.Second,
		20 * time.Second,
		40 * time.Second,
		60 * time.Second,
		60 * time.Second,
		60 * time.Second,
	}
	if !reflect.DeepEqual(clock.waits, wantSchedule) {
		t.Fatalf("backoff schedule = %v, want %v", clock.waits, wantSchedule)
	}
	if probe.calls != len(wantSchedule) {
		t.Fatalf("probe calls = %d, want %d", probe.calls, len(wantSchedule))
	}
	if value := raylineARCEncoderReadyGauge(); value != 1 {
		t.Fatalf("readiness gauge = %v, want 1 after recovery", value)
	}
	if selector.armedComponents() == nil {
		t.Fatal("recovered readiness left the selector unarmed")
	}
}

// Shutdown is the loop's only exit other than a good probe, so a cancelled
// wait must stop it without ever claiming readiness.
func TestRaylineARCReadinessRecoveryStopsOnShutdown(t *testing.T) {
	setRaylineARCEncoderReadyGauge(t, false)
	probe := &fakeARCProbe{failures: 100}
	clock := &fakeARCProbeClock{cancelAfter: 3}
	selector := newRaylineARCSelector(nil, nil, nil, "artifact-revision")

	if raylineARCRecoverReadiness(
		context.Background(),
		selector,
		armedARCComponentsFixture(),
		probe.run,
		raylineARCProbeBackoff{
			initial: 5 * time.Second,
			max:     60 * time.Second,
		},
		clock.wait,
	) {
		t.Fatal("a cancelled recovery reported readiness")
	}
	if len(clock.waits) != 3 {
		t.Fatalf("waits after shutdown = %d, want 3", len(clock.waits))
	}
	if value := raylineARCEncoderReadyGauge(); value != 0 {
		t.Fatalf("readiness gauge = %v, want 0 after shutdown", value)
	}
	if selector.armedComponents() != nil {
		t.Fatal("a cancelled recovery armed the selector")
	}
}

// The schedule is config-driven, and an unset knob keeps the shipped default
// rather than a zero-second hot loop.
func TestRaylineARCProbeBackoffFromConfig(t *testing.T) {
	tests := []struct {
		name    string
		encoder config.RaylineARCEncoderConfig
		want    raylineARCProbeBackoff
	}{
		{
			name: "unset falls back to the defaults",
			want: raylineARCProbeBackoff{
				initial: defaultRaylineARCProbeRetryInitial,
				max:     defaultRaylineARCProbeRetryMax,
			},
		},
		{
			name: "configured seconds are honoured",
			encoder: config.RaylineARCEncoderConfig{
				ProbeRetryInitialSeconds: 2,
				ProbeRetryMaxSeconds:     30,
			},
			want: raylineARCProbeBackoff{
				initial: 2 * time.Second,
				max:     30 * time.Second,
			},
		},
		{
			name: "an initial above the ceiling never shrinks it",
			encoder: config.RaylineARCEncoderConfig{
				ProbeRetryInitialSeconds: 120,
			},
			want: raylineARCProbeBackoff{
				initial: 120 * time.Second,
				max:     120 * time.Second,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := raylineARCProbeBackoffFromConfig(test.encoder)
			if got != test.want {
				t.Fatalf("backoff = %+v, want %+v", got, test.want)
			}
		})
	}
}

func setRaylineARCEncoderReadyGauge(t *testing.T, ready bool) {
	t.Helper()
	metrics.SetRaylineARCComponentReady(ready)
}

func raylineARCEncoderReadyGauge() float64 {
	return testutil.ToFloat64(
		metrics.RaylineARCComponentReady.WithLabelValues(
			"artifact_head_encoder",
		),
	)
}

// fakeARCProbe fails its first `failures` calls, which is the transient
// encoder rejection the recovery loop exists to survive.
type fakeARCProbe struct {
	failures int
	calls    int
}

func (probe *fakeARCProbe) run(context.Context) error {
	probe.calls++
	if probe.calls <= probe.failures {
		return errors.New("encoder probe rejected")
	}
	return nil
}

// fakeARCProbeClock records the schedule instead of sleeping it, and can cut
// a wait short to stand in for shutdown.
type fakeARCProbeClock struct {
	waits       []time.Duration
	cancelAfter int
}

func (clock *fakeARCProbeClock) wait(
	_ context.Context,
	delay time.Duration,
) bool {
	clock.waits = append(clock.waits, delay)
	return clock.cancelAfter == 0 || len(clock.waits) < clock.cancelAfter
}
