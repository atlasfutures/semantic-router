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
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// The shipped schedule. Five seconds is short enough that a Modal cold start
// or a momentary credential rejection costs one request window, and a minute
// is long enough that a genuinely down encoder is not hammered.
const (
	defaultRaylineARCProbeRetryInitial = 5 * time.Second
	defaultRaylineARCProbeRetryMax     = 60 * time.Second
	// The startup probe holds the router's port shut while it runs, and Cloud
	// Run allows 180 s on its own TCP startup probe. Thirty seconds answers a
	// warm encoder many times over and leaves the platform budget intact when
	// the encoder is cold.
	defaultRaylineARCProbeStartupTimeout = 30 * time.Second
)

// raylineARCEncoderProbeFailureClass is the readiness class a failed or
// timed-out encoder probe reports. It is produced in two places and matched in
// a third, so it lives here rather than as three string literals.
const raylineARCEncoderProbeFailureClass = "encoder_probe"

// raylineARCStartupProbeAttempt numbers the one synchronous probe, so the
// recovery loop's attempts continue the same count in the logs.
const raylineARCStartupProbeAttempt = 1

// raylineARCReadinessProbeName is the session identity every readiness probe
// uses, so the encoder sees one long-lived readiness session rather than one
// per attempt.
const raylineARCReadinessProbeName = "semantic-router-startup-readiness"

// raylineARCProbeBackoff bounds the readiness re-probe schedule.
type raylineARCProbeBackoff struct {
	initial time.Duration
	max     time.Duration
}

// raylineARCStartupProbeTimeoutFromConfig bounds the synchronous startup
// probe, defaulting when the deployment leaves the knob unset.
func raylineARCStartupProbeTimeoutFromConfig(
	encoder config.RaylineARCEncoderConfig,
) time.Duration {
	if encoder.ProbeStartupTimeoutSeconds <= 0 {
		return defaultRaylineARCProbeStartupTimeout
	}
	return time.Duration(encoder.ProbeStartupTimeoutSeconds) * time.Second
}

// raylineARCProbeBackoffFromConfig reads the operator's schedule, filling in
// the shipped defaults for the knobs a deployment leaves unset. A configured
// initial delay above the ceiling raises the ceiling rather than shrinking the
// delay, so the schedule can never be faster than the operator asked for.
func raylineARCProbeBackoffFromConfig(
	encoder config.RaylineARCEncoderConfig,
) raylineARCProbeBackoff {
	backoff := raylineARCProbeBackoff{
		initial: time.Duration(encoder.ProbeRetryInitialSeconds) * time.Second,
		max:     time.Duration(encoder.ProbeRetryMaxSeconds) * time.Second,
	}
	if backoff.initial <= 0 {
		backoff.initial = defaultRaylineARCProbeRetryInitial
	}
	if backoff.max <= 0 {
		backoff.max = defaultRaylineARCProbeRetryMax
	}
	if backoff.max < backoff.initial {
		backoff.max = backoff.initial
	}
	return backoff
}

// raylineARCWait sleeps for delay unless ctx ends first, and reports whether
// the wait ran to completion. A false answer means the router generation is
// shutting down, which is the recovery loop's only exit other than readiness.
func raylineARCWait(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// raylineARCReprobe re-runs probe on a bounded exponential schedule until it
// answers or ctx ends.
//
// Attempts are deliberately uncapped. While the encoder is unreachable the
// instance can only fail closed, so the two sensible end states are "ready"
// and "shut down"; giving up would just reproduce the one-shot probe this
// loop exists to replace.
func raylineARCReprobe(
	ctx context.Context,
	probe func(context.Context) error,
	backoff raylineARCProbeBackoff,
	wait func(context.Context, time.Duration) bool,
) bool {
	delay := backoff.initial
	attempt := raylineARCStartupProbeAttempt
	for {
		if !wait(ctx, delay) {
			return false
		}
		attempt++
		if probe(ctx) == nil {
			return true
		}
		delay *= 2
		if delay > backoff.max {
			delay = backoff.max
		}
		logRaylineARCProbeFailure(attempt, delay, false)
	}
}

// raylineARCArmOrRecover runs the one synchronous readiness probe under a
// bound and decides which way readiness goes.
//
// The bound is the point of this function. The router does not open its port
// until readiness returns, so an unbounded probe against a cold encoder holds
// the port shut past the platform's own startup budget and the instance is
// killed before it can recover. A probe that outruns the bound is therefore
// treated exactly as a failed probe: the selector stays unarmed, the process
// stays up and fails closed, and the recovery loop takes over. Recovery
// attempts are not bounded this way, so a cold encoder is answered by the
// first recovery attempt and arms the selector.
//
// It returns the readiness failure class, empty when the probe armed the
// selector on the spot.
func raylineARCArmOrRecover(
	ctx context.Context,
	selector *raylineARCSelector,
	armed *raylineARCArmedComponents,
	probe func(context.Context) error,
	startupTimeout time.Duration,
	backoff raylineARCProbeBackoff,
	wait func(context.Context, time.Duration) bool,
) string {
	logging.ComponentEvent(
		"extproc",
		"rayline_arc_readiness_schedule",
		map[string]interface{}{
			"startup_timeout_seconds": startupTimeout.Seconds(),
			"initial_backoff_seconds": backoff.initial.Seconds(),
			"max_backoff_seconds":     backoff.max.Seconds(),
		},
	)
	startupContext, cancelStartup := context.WithTimeout(ctx, startupTimeout)
	probeErr := probe(startupContext)
	timedOut := errors.Is(startupContext.Err(), context.DeadlineExceeded)
	cancelStartup()
	if probeErr == nil {
		selector.arm(armed)
		return ""
	}
	logRaylineARCProbeFailure(
		raylineARCStartupProbeAttempt,
		backoff.initial,
		timedOut,
	)
	go raylineARCRecoverReadiness(ctx, selector, armed, probe, backoff, wait)
	return raylineARCEncoderProbeFailureClass
}

// logRaylineARCProbeFailure is the one place a readiness attempt is reported,
// so an operator can read the schedule off the logs rather than infer it.
func logRaylineARCProbeFailure(
	attempt int,
	nextDelay time.Duration,
	timedOut bool,
) {
	logging.ComponentErrorEvent(
		"extproc",
		"rayline_arc_readiness_probe_failed",
		map[string]interface{}{
			"attempt":            attempt,
			"next_delay_seconds": nextDelay.Seconds(),
			"timed_out":          timedOut,
			"startup":            attempt == raylineARCStartupProbeAttempt,
		},
	)
}

// raylineARCRecoverReadiness turns a failed startup probe into a recoverable
// state. It keeps probing, and the first good answer arms the selector and
// flips the readiness gauge. Until then the selector stays unarmed and every
// selection fails closed.
func raylineARCRecoverReadiness(
	ctx context.Context,
	selector *raylineARCSelector,
	armed *raylineARCArmedComponents,
	probe func(context.Context) error,
	backoff raylineARCProbeBackoff,
	wait func(context.Context, time.Duration) bool,
) bool {
	if !raylineARCReprobe(ctx, probe, backoff, wait) {
		return false
	}
	// Arm before the gauge: an operator who sees ready=1 must never find a
	// selector that still refuses requests.
	selector.arm(armed)
	metrics.SetRaylineARCComponentReady(true)
	logging.ComponentEvent(
		"extproc",
		"rayline_arc_component_readiness",
		map[string]interface{}{
			"ready":     true,
			"recovered": true,
		},
	)
	return true
}
