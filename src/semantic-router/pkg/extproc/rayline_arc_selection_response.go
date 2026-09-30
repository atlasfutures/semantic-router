package extproc

import (
	"errors"
	"net/http"
	"strconv"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
		arcEncoderFailureClassAdmission,
		// The policy service's own back-pressure: another call holds the
		// session, or its session table is full. Both clear like a lease.
		"policy_service_session_busy",
		"policy_service_session_capacity":
		return true
	default:
		return false
	}
}

// selectionFailureHeader tells the caller WHY a selection failed, in a small
// public vocabulary, so it can choose between waiting for its own in-flight
// turn, backing off, warming up and giving up. The status and body alone
// cannot: every contended class shares one 429 and one message, and every
// other class shares one 503. The value never carries the internal class,
// which names private components.
const selectionFailureHeader = headers.VSRFailureClass

const (
	// Another request holds this session: wait for it, do not resend it.
	selectionFailureSessionBusy = "session_busy"
	// The router or a decision service is at its concurrency limit: back off.
	selectionFailureCapacity = "capacity"
	// A decision dependency is still starting: retry after it warms.
	selectionFailureNotReady = "not_ready"
	// The request named no session: retrying the same request cannot succeed.
	selectionFailureMissingSession = "missing_session"
	// Anything else: waiting is not known to help.
	selectionFailureUnavailable = "unavailable"
)

// publicSelectionFailureClass maps an internal failure class onto the public
// vocabulary. Every contended class lands on session_busy or capacity, so the
// header never contradicts the 429 it rides on.
func publicSelectionFailureClass(class string) string {
	switch class {
	case "episode_timeout", "policy_service_session_busy":
		return selectionFailureSessionBusy
	case "episode_capacity",
		arcEncoderFailureClassAdmission,
		"policy_service_session_capacity":
		return selectionFailureCapacity
	case "not_ready":
		return selectionFailureNotReady
	case arcFailureMissingEpisodeID:
		return selectionFailureMissingSession
	default:
		return selectionFailureUnavailable
	}
}

// selectionFailureIsCallerError separates the caller's omission from the
// router's breakage. A request that names no episode is not going to succeed
// on retry, so answering 503 sends the caller round a loop that cannot end.
// Every other class is the router's to answer for.
func selectionFailureIsCallerError(class string) bool {
	return class == arcFailureMissingEpisodeID
}

// missingEpisodeHeaderMessage names the header the caller left out. The header
// name is configuration, not request content, so it is safe to return. The
// article is "the" rather than "a" or "an" because the name is configured and
// either article would be wrong for some spelling of it.
func missingEpisodeHeaderMessage(ctx *RequestContext) string {
	if ctx == nil || ctx.RaylineARCEpisodeIDHeader == "" {
		return "This request needs a session header."
	}
	return "This request needs the " + ctx.RaylineARCEpisodeIDHeader + " header."
}

// missingEpisodeHeaderResponse refuses a request that named no episode. The
// immediate response is re-encoded into the client's wire format before it
// leaves, and that step uses a canned message per status unless the request
// carries the protocol error to use, so the refusal is recorded there too.
// Otherwise the caller is told only that something was invalid.
func (r *OpenAIRouter) missingEpisodeHeaderResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	message := missingEpisodeHeaderMessage(ctx)
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			arcFailureMissingEpisodeID,
			message,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusBadRequest, message)
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
	var response *ext_proc.ProcessingResponse
	switch {
	case selectionFailureIsCallerError(failure.class):
		// The bounded class is already counted where it is constructed, which
		// is the one place that sees both entrypoints. Counting it again here
		// would count the ExtProc path twice.
		response = r.missingEpisodeHeaderResponse(ctx)
	case !selectionFailureIsContended(failure.class):
		response = r.createErrorResponse(
			http.StatusServiceUnavailable,
			selectionUnavailableMessage(ctx),
		)
	default:
		response = r.createErrorResponse(
			http.StatusTooManyRequests,
			selectionContendedMessage(ctx),
		)
		appendRetryAfterHeader(response, selectionContendedRetryAfterSeconds)
	}
	appendImmediateHeader(response, selectionFailureHeader, publicSelectionFailureClass(failure.class))
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
	response := r.createErrorResponse(
		http.StatusServiceUnavailable,
		selectionUnavailableMessage(ctx),
	)
	// A lost lease means another request now owns this session.
	appendImmediateHeader(response, selectionFailureHeader, selectionFailureSessionBusy)
	return response
}

func appendRetryAfterHeader(
	response *ext_proc.ProcessingResponse,
	seconds int,
) {
	appendImmediateHeader(response, "retry-after", strconv.Itoa(seconds))
}

func appendImmediateHeader(
	response *ext_proc.ProcessingResponse,
	key string,
	value string,
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
			Key:      key,
			RawValue: []byte(value),
		}},
	)
}
