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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type raylineARCSessionCloseFunc func(
	context.Context,
	string,
	[]string,
) (raylinearc.EncoderCloseReport, error)

func (transaction *raylineARCEpisodeTransaction) finalizeTimeout() time.Duration {
	if transaction == nil || !transaction.closeRequested ||
		transaction.sessionCloseWait <= 0 {
		return episodeFinalizeTimeout
	}
	return episodeFinalizeTimeout + transaction.sessionCloseWait
}

func (transaction *raylineARCEpisodeTransaction) closeRetainedEncoderSession(
	ctx context.Context,
	requestContext *RequestContext,
	nextState *raylinearc.EpisodeState,
) {
	if transaction == nil || !transaction.closeRequested || nextState == nil {
		return
	}
	report := raylinearc.EncoderCloseReport{Failed: 1}
	var closeErr error
	if transaction.sessionCloser == nil ||
		len(nextState.EncoderVisitedOwners) == 0 {
		closeErr = errors.New("ARC encoder session closer is unavailable")
	} else {
		report, closeErr = transaction.sessionCloser(
			ctx,
			transaction.episodeIDHash,
			nextState.EncoderVisitedOwners,
		)
	}
	metrics.RecordRaylineARCEncoderSessionClose(
		report.Closed,
		report.Unavailable,
		report.Failed,
	)
	fields := map[string]interface{}{
		"outcome":     "closed",
		"attempted":   report.Attempted,
		"closed":      report.Closed,
		"unavailable": report.Unavailable,
		"failed":      report.Failed,
	}
	if requestContext != nil {
		fields["request_id"] = requestContext.RequestID
	}
	if closeErr != nil {
		fields["outcome"] = "partial_failure"
		fields["failure_class"] = boundedARCSessionCloseFailure(closeErr)
		logging.ComponentErrorEvent(
			"extproc",
			"rayline_arc_encoder_session_close",
			fields,
		)
		return
	}
	// Every visited owner either confirmed deletion or was explicitly
	// unavailable. Keep policy history but release encoder affinity so a future
	// reuse of the episode ID starts a new retained session assignment.
	nextState.EncoderOwner = ""
	nextState.EncoderVisitedOwners = nil
	logging.ComponentEvent(
		"extproc",
		"rayline_arc_encoder_session_close",
		fields,
	)
}

func boundedARCSessionCloseFailure(err error) string {
	var encoderFailure *raylinearc.EncoderFailure
	switch {
	case err == nil:
		return ""
	case errors.Is(err, context.Canceled),
		errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.As(err, &encoderFailure):
		return string(encoderFailure.Class)
	default:
		return "close"
	}
}
