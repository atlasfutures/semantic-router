package extproc

import (
	"context"
	"errors"
	"math/rand/v2"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A decide that starts an episode (transition start) and is answered 503
// backend_unavailable is asked once more, after about a second: the service's
// edge sheds new episodes in short bursts before they reach a handler, and the
// episode has nothing committed that a second ask could duplicate. Only the
// start is retried. An extending turn, a delta, an append or a release may
// already have advanced the service's state, so each fails closed as before;
// so does the start's second failure.
const (
	policyStartRetryDelayMin = 750 * time.Millisecond
	policyStartRetryDelayMax = 1250 * time.Millisecond
	// policyStartRetryMinAttempt is the least time the retry must have left
	// once the backoff is over; a turn with less fails closed at once.
	policyStartRetryMinAttempt = 500 * time.Millisecond
)

// policyStartRetryDelay is the backoff before the one retry, jittered so the
// starts a burst shed do not all come back together. Tests shorten it.
var policyStartRetryDelay = func() time.Duration {
	return policyStartRetryDelayMin + rand.N(policyStartRetryDelayMax-policyStartRetryDelayMin+1)
}

// policyStartRetryable reports the failure a start is retried on: the
// service's 503 backend_unavailable.
func policyStartRetryable(err error) bool {
	var failure *raylinearc.PolicyServiceError
	return errors.As(err, &failure) && failure.Class == "backend_unavailable" &&
		failure.Status == http.StatusServiceUnavailable
}

// retryPolicyStartOnce asks a failed start once more through decide, within
// ctx's deadline. It returns the first answer when the failure is not a
// start's 503 backend_unavailable, or when the turn has too little time left.
func retryPolicyStartOnce(
	ctx context.Context,
	transition string,
	episodeIDHash string,
	response *raylinearc.PolicyDecisionResponse,
	err error,
	decide func() (*raylinearc.PolicyDecisionResponse, error),
) (*raylinearc.PolicyDecisionResponse, error) {
	if transition != raylinearc.PolicyTransitionStart || !policyStartRetryable(err) || ctx.Err() != nil {
		return response, err
	}
	delay := policyStartRetryDelay()
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < delay+policyStartRetryMinAttempt {
		metrics.RecordRaylineARCPolicySessionStartRetry("skipped")
		logging.ComponentWarnEvent("extproc", "rayline_arc_policy_session_start_retry", map[string]interface{}{
			"episode_id_hash": episodeIDHash, "outcome": "skipped", "delay_ms": delay.Milliseconds(),
		})
		return response, err
	}
	timer := time.NewTimer(delay)
	select {
	case <-ctx.Done():
		timer.Stop()
		return response, err
	case <-timer.C:
	}
	response, err = decide()
	outcome := "recovered"
	fields := map[string]interface{}{"episode_id_hash": episodeIDHash, "delay_ms": delay.Milliseconds()}
	if err != nil {
		outcome = "failed"
		var failure *raylinearc.PolicyServiceError
		if errors.As(err, &failure) {
			fields["class"], fields["status"] = failure.Class, failure.Status
		}
	}
	fields["outcome"] = outcome
	metrics.RecordRaylineARCPolicySessionStartRetry(outcome)
	logging.ComponentWarnEvent("extproc", "rayline_arc_policy_session_start_retry", fields)
	return response, err
}

// recordPolicyModalSessionStart counts and logs how a Modal session start
// that got no answer at first ended (raylinearc.PolicySessionBackend.Start
// retries it once under its idempotency key and reaps one it gave up on).
// The episode it served is not known there; the session token never is.
func recordPolicyModalSessionStart(outcome string) {
	metrics.RecordRaylineARCPolicyModalSessionStart(outcome)
	logging.ComponentWarnEvent("extproc", "rayline_arc_policy_modal_session_start", map[string]interface{}{
		"outcome": outcome,
	})
}
