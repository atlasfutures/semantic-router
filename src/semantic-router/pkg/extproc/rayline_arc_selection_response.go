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
// This vocabulary is published in three places: the constants below, the
// header's own contract (pkg/headers/headers.go) and the client reference
// (website/docs/troubleshooting/vsr-headers.md). A new value changes all
// three, or a caller reading one of them handles it wrongly.
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
	// A decision service cannot represent something this request carries:
	// resending it unchanged cannot succeed either, and no arm was involved.
	selectionFailureRequestShape = "request_shape"
	// The policy would not decide this turn and nothing of this
	// conversation's own decisions could hold it, so no model serves it:
	// resending it unchanged cannot succeed, and no backend was called.
	selectionFailureUndecidable = "undecidable"
	// The request's data mode allows none of the arms that could serve this
	// turn: resending it unchanged cannot succeed, and no backend was called.
	selectionFailureNoPrivateArm = arcFailureNoPrivateArm
	// The turn asked to continue a stored conversation on a path that does
	// not do stateful continuation: resend the items instead of the id.
	selectionFailureStatelessPath = "stateless_path"
	// The request pinned an ARC snapshot this route does not serve:
	// resending it unchanged cannot succeed; name a served snapshot or none.
	selectionFailureUnknownSnapshot = "unknown_snapshot"
	// Anything else: waiting is not known to help.
	selectionFailureUnavailable = "unavailable"
)

// selectionShapeRefusalClasses are the internal classes that say a decision
// service refused what the request carried, not that it was unwell. They are
// the caller's to fix and nobody's outage: the arm is never dialled, so
// reporting them as a model-service failure blames a component that was not
// involved and invites a retry that cannot succeed.
//
// Only policy_service_unsupported_request qualifies: the contract's "the
// projection cannot represent this envelope" (its own status is 422), which
// is a statement about the conversation the caller sent.
//
// invalid_request is deliberately NOT here. The service returns it for a
// malformed PolicyDecisionRequest as a whole, and the router builds most of
// that envelope itself -- schema version, package ref, selection, attribution
// and episode fields (rayline_arc_policy_service.go). A service-version skew
// or a router-generated field the service refuses would then be reported to
// the caller as its own 400, blaming it for an integration fault it cannot
// fix and hiding a router bug. It stays "unavailable".
var selectionShapeRefusalClasses = map[string]bool{
	"policy_service_unsupported_request": true,
}

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
	case arcFailurePolicyUndecidable:
		return selectionFailureUndecidable
	case arcFailureResponsesPreviousID:
		return selectionFailureStatelessPath
	case arcFailureNoPrivateArm:
		return selectionFailureNoPrivateArm
	case arcFailureUnknownSnapshot:
		return selectionFailureUnknownSnapshot
	default:
		if selectionShapeRefusalClasses[class] {
			return selectionFailureRequestShape
		}
		return selectionFailureUnavailable
	}
}

// selectionFailureIsCallerError separates the caller's omission from the
// router's breakage. A request that names no episode, or that carries
// something a decision service cannot represent, is not going to succeed on
// retry, so answering 503 sends the caller round a loop that cannot end.
// Every other class is the router's to answer for.
func selectionFailureIsCallerError(class string) bool {
	return class == arcFailureMissingEpisodeID ||
		class == arcFailureResponsesPreviousID ||
		selectionShapeRefusalClasses[class]
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

// requestShapeRefusalMessage says the request carried something the routing
// decision cannot represent, without naming the member or the component: the
// refusal detail is a decision service's own text, and the class vocabulary is
// deliberately free of private component names.
const requestShapeRefusalMessage = "This request carries a member the routing decision cannot represent."

// undecidableMessage says the router could not decide the turn, without naming
// the component or the package.
const undecidableMessage = "The router could not decide a model for this turn, and serves none in its place."

// undecidableResponse refuses a turn the policy would not decide. It is a 422,
// not a 503: no backend was called and the router is healthy, and the same
// request is refused identically every time, so a 503 would both read as an
// outage and invite a retry that cannot succeed.
func (r *OpenAIRouter) undecidableResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			selectionFailureUndecidable,
			undecidableMessage,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusUnprocessableEntity, undecidableMessage)
}

// noPrivateArmMessage says the request's data mode allows no arm that could
// serve the turn, without naming the arms or their providers.
const noPrivateArmMessage = "No model this route may use under the request's privacy mode can serve this turn, and none is served in its place."

// noPrivateArmResponse refuses a private turn none of whose allowed arms can
// serve it. It is a 422 for the reason undecidable is: no backend was called,
// the router is healthy, and the same request is refused identically every
// time, so a 503 would read as an outage and invite a replay.
func (r *OpenAIRouter) noPrivateArmResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			selectionFailureNoPrivateArm,
			noPrivateArmMessage,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusUnprocessableEntity, noPrivateArmMessage)
}

// requestShapeRefusalResponse refuses a request a decision service could not
// represent. It is a 400 and not a 503 for the reason the status is read: no
// arm was dialled, so no model service was unavailable, and a 503 both blames
// a component that was not involved and asks the caller to retry a request
// that will be refused identically every time. Monitoring reads a 503 as
// provider downtime, so the wrong status also reports an outage that did not
// happen.
func (r *OpenAIRouter) requestShapeRefusalResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			selectionFailureRequestShape,
			requestShapeRefusalMessage,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusBadRequest, requestShapeRefusalMessage)
}

// statelessPathMessage tells the caller both halves of what it needs: that
// this route does not continue a stored conversation, and what to send
// instead. Naming the remedy is the point of the ticket -- the old 503 said
// only that something was unavailable, so a client retried the same request,
// or recreated the conversation, forever (memex-desktop#7484). It names
// previous_response_id because the caller sent it; it names no component.
const statelessPathMessage = "This route does not continue a stored conversation: " +
	"previous_response_id cannot be resolved here. Resend the conversation's items in input."

// statelessPathResponse refuses a Responses turn that names
// previous_response_id. It is a 400 and not a 503 because nothing was
// unavailable: no arm was dialled, the router is healthy, and this same
// request is refused identically every time, so a 503 both reports an outage
// that did not happen and invites a retry that cannot succeed. A client told
// 400 with this message can fall back to replaying items, which is what the
// Codex CLI and our own SDK harness already do.
func (r *OpenAIRouter) statelessPathResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			selectionFailureStatelessPath,
			statelessPathMessage,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusBadRequest, statelessPathMessage)
}

// unknownSnapshotMessage says the request named an ARC snapshot this route
// does not serve, without listing the ones it does.
const unknownSnapshotMessage = "This route does not serve the ARC snapshot the x-rayline-arc-snapshot header names, and serves no other in its place."

// unknownSnapshotResponse refuses a request pinned to a snapshot the cell
// does not serve. It is a 400: the caller's pin is wrong, the router is
// healthy, and the same request is refused identically every time. Serving
// another snapshot instead would bill and decide under a release the caller
// did not ask for.
func (r *OpenAIRouter) unknownSnapshotResponse(
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx != nil {
		ctx.ImmediateProtocolError = llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest,
			selectionFailureUnknownSnapshot,
			unknownSnapshotMessage,
			nil,
		)
	}
	return r.createErrorResponse(http.StatusBadRequest, unknownSnapshotMessage)
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
	recordTurnFailure(ctx, selectionTurnFailureClass(failure.class), false)
	var response *ext_proc.ProcessingResponse
	switch {
	case selectionShapeRefusalClasses[failure.class]:
		response = r.requestShapeRefusalResponse(ctx)
	case failure.class == arcFailurePolicyUndecidable:
		response = r.undecidableResponse(ctx)
	case failure.class == arcFailureNoPrivateArm:
		response = r.noPrivateArmResponse(ctx)
	case failure.class == arcFailureResponsesPreviousID:
		response = r.statelessPathResponse(ctx)
	case failure.class == arcFailureUnknownSnapshot:
		response = r.unknownSnapshotResponse(ctx)
	case failure.class == arcFailureSnapshotPackageMismatch:
		// The policy service answered for another package than the
		// snapshot pins: an upstream fault, never served.
		response = r.createErrorResponse(http.StatusBadGateway, selectionUnavailableMessage(ctx))
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
	recordTurnFailure(ctx, selectionFailureUnavailable, false)
	response := r.createErrorResponse(
		http.StatusServiceUnavailable,
		selectionUnavailableMessage(ctx),
	)
	// Not session_busy: the gate cannot tell a competing owner from a renewal
	// that failed on transport or timeout, and a caller told session_busy
	// would wait for an in-flight turn that may not exist.
	appendImmediateHeader(response, selectionFailureHeader, selectionFailureUnavailable)
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
