package extproc

import (
	"errors"
	"net/http"
	"strconv"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// selectionContendedRetryAfterSeconds is the shortest honest wait. Both
// contention sources clear on the order of one consult: an episode lease is
// held for one request, and the encoder admission gate releases as soon as an
// in-flight call returns.
const selectionContendedRetryAfterSeconds = 1

// selectionFailureIsContended separates back-pressure from breakage. A
// contended episode lease and a spent encoder admission budget both mean the
// router is healthy and the request is well formed, so waiting fixes them.
// Every other class means waiting will not help.
//
// This is the one classifier for that split. The decision-only API adapter
// derives its 429 from the same set, so a class added here cannot answer 429
// on one entrypoint and 503 on the other.
func selectionFailureIsContended(class string) bool {
	switch class {
	case "episode_timeout",
		"episode_capacity",
		arcEncoderFailureClassAdmission:
		return true
	default:
		return false
	}
}

// authoritativeSelectionFailureResponse maps a fail-closed selector's bounded
// failure onto the admission contract. It answers nil for every other error so
// the caller keeps its existing classification.
func (r *OpenAIRouter) authoritativeSelectionFailureResponse(
	err error,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	var failure *modelSelectionFailure
	if !errors.As(err, &failure) {
		return nil
	}
	recordSelectionLifecycleFailure(ctx, "selection", err)
	if !selectionFailureIsContended(failure.class) {
		return r.createErrorResponse(
			http.StatusServiceUnavailable,
			selectionUnavailableMessage(ctx),
		)
	}
	response := r.createErrorResponse(
		http.StatusTooManyRequests,
		selectionContendedMessage(ctx),
	)
	appendRetryAfterHeader(response, selectionContendedRetryAfterSeconds)
	return response
}

// selectionDispatchGateResponse fails a request closed when the prepared
// selection may no longer dispatch. A lost lease means another consult owns
// the episode, so this request's state is stale.
func (r *OpenAIRouter) selectionDispatchGateResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	err := selectionDispatchAllowed(ctx)
	if err == nil {
		return nil
	}
	recordSelectionLifecycleFailure(ctx, "dispatch", err)
	return r.createErrorResponse(
		http.StatusServiceUnavailable,
		selectionUnavailableMessage(ctx),
	)
}

func appendRetryAfterHeader(
	response *ext_proc.ProcessingResponse,
	seconds int,
) {
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return
	}
	if immediate.Headers == nil {
		immediate.Headers = &ext_proc.HeaderMutation{}
	}
	immediate.Headers.SetHeaders = append(
		immediate.Headers.SetHeaders,
		&core.HeaderValueOption{Header: &core.HeaderValue{
			Key:      "retry-after",
			RawValue: []byte(strconv.Itoa(seconds)),
		}},
	)
}
