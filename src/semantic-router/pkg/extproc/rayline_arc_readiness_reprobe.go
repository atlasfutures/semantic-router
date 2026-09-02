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
)

// raylineARCReadinessPendingClass is the state every instance now starts in:
// the selector is registered and unarmed, and a background probe decides when
// it arms. It is not a failure, so readiness reports it as an ordinary event.
const raylineARCReadinessPendingClass = "encoder_probe_pending"

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
		logRaylineARCProbeFailure(attempt, delay)
	}
}

// raylineARCArmInBackground starts readiness without blocking the caller.
//
// This is the whole point of the function. The router does not open its gRPC
// port until every component is constructed, so any encoder call made here
// holds the port shut, and a cold encoder holds it shut past the platform's
// own startup budget. Nothing on the construction path may wait on the
// encoder. The selector is therefore registered unarmed, every selection
// fails closed on not_ready, and the first probe runs on this goroutine
// exactly like the retries that may follow it.
//
// Probes are not deadlined beyond the encoder's own timeout. The bound that
// used to exist here protected the port, and the port is no longer waiting.
func raylineARCArmInBackground(
	ctx context.Context,
	selector *raylineARCSelector,
	armed *raylineARCArmedComponents,
	probe func(context.Context) error,
	backoff raylineARCProbeBackoff,
	wait func(context.Context, time.Duration) bool,
) {
	logging.ComponentEvent(
		"extproc",
		"rayline_arc_readiness_schedule",
		map[string]interface{}{
			"initial_backoff_seconds": backoff.initial.Seconds(),
			"max_backoff_seconds":     backoff.max.Seconds(),
		},
	)
	go func() {
		if probe(ctx) == nil {
			raylineARCPublishReady(selector, armed, false)
			return
		}
		logRaylineARCProbeFailure(raylineARCStartupProbeAttempt, backoff.initial)
		raylineARCRecoverReadiness(ctx, selector, armed, probe, backoff, wait)
	}()
}

// logRaylineARCProbeFailure is the one place a readiness attempt is reported,
// so an operator can read the schedule off the logs rather than infer it.
func logRaylineARCProbeFailure(attempt int, nextDelay time.Duration) {
	logging.ComponentErrorEvent(
		"extproc",
		"rayline_arc_readiness_probe_failed",
		map[string]interface{}{
			"attempt":            attempt,
			"next_delay_seconds": nextDelay.Seconds(),
			"startup":            attempt == raylineARCStartupProbeAttempt,
		},
	)
}

// raylineARCRecoverReadiness keeps probing until the encoder answers or the
// generation shuts down. The first good answer arms the selector. Until then
// every selection fails closed.
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
	raylineARCPublishReady(selector, armed, true)
	return true
}

// raylineARCPublishReady is the one place readiness becomes true. It arms
// before it reports, so an operator who sees ready=1 never finds a selector
// that still refuses requests.
func raylineARCPublishReady(
	selector *raylineARCSelector,
	armed *raylineARCArmedComponents,
	recovered bool,
) {
	selector.arm(armed)
	metrics.SetRaylineARCComponentReady(true)
	logging.ComponentEvent(
		"extproc",
		"rayline_arc_component_readiness",
		map[string]interface{}{
			"ready":     true,
			"recovered": recovered,
		},
	)
}
