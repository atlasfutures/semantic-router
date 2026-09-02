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

// raylineARCReadinessProbeName is the session identity every readiness probe
// uses, so the encoder sees one long-lived readiness session rather than one
// per attempt.
const raylineARCReadinessProbeName = "semantic-router-startup-readiness"

// raylineARCProbeBackoff bounds the readiness re-probe schedule.
type raylineARCProbeBackoff struct {
	initial time.Duration
	max     time.Duration
}

// raylineARCDefaultProbeBackoff is the shipped schedule.
func raylineARCDefaultProbeBackoff() raylineARCProbeBackoff {
	return raylineARCProbeBackoff{
		initial: defaultRaylineARCProbeRetryInitial,
		max:     defaultRaylineARCProbeRetryMax,
	}
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
	for {
		if !wait(ctx, delay) {
			return false
		}
		if probe(ctx) == nil {
			return true
		}
		delay *= 2
		if delay > backoff.max {
			delay = backoff.max
		}
	}
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
