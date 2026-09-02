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
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
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

// The startup probe is synchronous and the router does not open its port
// until it returns, so an unbounded probe against a cold encoder starves
// Cloud Run's own startup probe and the instance is killed before the
// recovery loop is ever entered. Deploy 00008-7dm failed exactly that way.
// A startup probe that outruns its bound must be treated as a failed probe:
// unarmed, live, and recovering.
func TestRaylineARCStartupProbeTimeoutHandsOverToRecovery(t *testing.T) {
	setRaylineARCEncoderReadyGauge(t, false)
	probe := &gatedARCProbe{release: make(chan struct{})}
	clock := &fakeARCProbeClock{}
	selector := newRaylineARCSelector(nil, nil, nil, "artifact-revision")
	components := armedARCComponentsFixture()

	failureClass := raylineARCArmOrRecover(
		context.Background(),
		selector,
		components,
		probe.run,
		20*time.Millisecond,
		raylineARCProbeBackoff{initial: time.Second, max: time.Second},
		clock.wait,
	)

	if failureClass != raylineARCEncoderProbeFailureClass {
		t.Fatalf("failure class = %q, want %q", failureClass, raylineARCEncoderProbeFailureClass)
	}
	// Fail closed for the whole window: live selector, no selection served.
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	_, err = selector.Select(context.Background(), validARCSelectionContext(state))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "not_ready" {
		t.Fatalf("selection during recovery = %v, want class not_ready", err)
	}
	if value := raylineARCEncoderReadyGauge(); value != 0 {
		t.Fatalf("readiness gauge = %v, want 0 while recovering", value)
	}

	// The encoder warms. The next recovery attempt must arm the selector.
	close(probe.release)
	deadline := time.Now().Add(5 * time.Second)
	for selector.armedComponents() == nil {
		if time.Now().After(deadline) {
			t.Fatal("recovery never armed the selector after the encoder warmed")
		}
		time.Sleep(time.Millisecond)
	}
	if value := raylineARCEncoderReadyGauge(); value != 1 {
		t.Fatalf("readiness gauge = %v, want 1 after recovery", value)
	}
	if probe.calls.Load() < 2 {
		t.Fatalf("probe calls = %d, want the startup probe plus a recovery attempt", probe.calls.Load())
	}
}

// A startup probe that answers inside its bound arms straight away and never
// starts a recovery loop.
func TestRaylineARCStartupProbeArmsWithoutRecovery(t *testing.T) {
	setRaylineARCEncoderReadyGauge(t, false)
	probe := &fakeARCProbe{}
	clock := &fakeARCProbeClock{}
	selector := newRaylineARCSelector(nil, nil, nil, "artifact-revision")

	failureClass := raylineARCArmOrRecover(
		context.Background(),
		selector,
		armedARCComponentsFixture(),
		probe.run,
		time.Minute,
		raylineARCProbeBackoff{initial: time.Second, max: time.Second},
		clock.wait,
	)

	if failureClass != "" {
		t.Fatalf("failure class = %q, want ready", failureClass)
	}
	if selector.armedComponents() == nil {
		t.Fatal("a good startup probe left the selector unarmed")
	}
	if len(clock.waits) != 0 {
		t.Fatalf("recovery ran after a good startup probe: waits = %v", clock.waits)
	}
}

// The startup bound is config-driven and must stay well under Cloud Run's
// 180 s TCP startup budget, so an unset knob keeps the shipped 30 s.
func TestRaylineARCStartupProbeTimeoutFromConfig(t *testing.T) {
	tests := []struct {
		name    string
		encoder config.RaylineARCEncoderConfig
		want    time.Duration
	}{
		{
			name: "unset falls back to the default",
			want: defaultRaylineARCProbeStartupTimeout,
		},
		{
			name: "configured seconds are honoured",
			encoder: config.RaylineARCEncoderConfig{
				ProbeStartupTimeoutSeconds: 45,
			},
			want: 45 * time.Second,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := raylineARCStartupProbeTimeoutFromConfig(test.encoder)
			if got != test.want {
				t.Fatalf("startup timeout = %v, want %v", got, test.want)
			}
		})
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

// gatedARCProbe blocks until the encoder "warms", which is how a cold Modal
// boot looks to the router: not an error, just an answer that does not come.
type gatedARCProbe struct {
	release chan struct{}
	calls   atomic.Int32
}

func (probe *gatedARCProbe) run(ctx context.Context) error {
	probe.calls.Add(1)
	select {
	case <-probe.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
