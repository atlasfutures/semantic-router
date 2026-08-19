package routerruntime

import (
	"context"
	"errors"
)

// RouteDecisionRequest is one decision-only route consult.
//
// It carries transport facts the management API has already validated, not
// routing inputs. Everything a routing algorithm needs is derived from these
// fields by the implementation, so the API server never has to reach into
// selection internals and the algorithm never has to parse HTTP.
type RouteDecisionRequest struct {
	// Body is the client's request body, unmutated. The caller sends an
	// Anthropic Messages payload verbatim; the implementation normalizes it.
	Body []byte
	// DecisionID is the caller-stamped route id, or a minted id when the
	// caller sent none. The adapter echoes it back so one id spans the
	// caller's durable row and this router's decision record.
	DecisionID string
	// SessionID is the caller's session identity, or empty. The
	// implementation maps it onto the algorithm's configured episode
	// identity; it is never used as an episode id directly.
	SessionID string
	// ExecutedModel is what the caller actually ran last turn, or empty when
	// it did not report one.
	//
	// Record-only. It must reach logs and traces and must never reach
	// episode state: a decision-only consult stays a self-consistent
	// hypothetical trajectory, so feeding execution feedback back into the
	// previous-arm state would silently change later selections. The
	// reference decision server holds the same rule.
	ExecutedModel string
}

// RouteDecision is the bounded set of selection facts a decision-only consult
// may publish.
//
// Optional fields are empty when the runtime has no real source for them. The
// adapter omits empty optional fields from the wire response rather than
// emitting a zero value, because a caller joining these rows offline cannot
// tell an invented value from a measured one.
type RouteDecision struct {
	// SelectedWorker is the chosen worker's identifier. Required.
	SelectedWorker string
	// WorkerModel is the model that worker serves. Required.
	WorkerModel string
	// Provider names the worker's provider, or is empty when the worker
	// manifest declares none.
	Provider string
}

// RouteDecisionRuntime is the narrow API-server seam for decision-only
// routing. The implementation lives with the router runtime; the API server
// only needs one selection answer without depending on extproc internals.
//
// Episode lifecycle: RouteDecision commits its episode transaction before it
// returns. A decision-only consult has no dispatch phase to commit against, so
// the selected arm becomes the episode's previous arm optimistically, at
// decision time. There is no third terminal state for the caller to report and
// no lease left pending after the response.
//
// Failure policy: an error is fail-closed. There is no fallback worker, so the
// adapter must surface the failure rather than answer with a default.
type RouteDecisionRuntime interface {
	RouteDecision(context.Context, RouteDecisionRequest) (RouteDecision, error)
}

// ErrRouteDecisionContended marks a consult that could not start because the
// router was already busy with this session's episode, or because the episode
// store was at capacity. It is contention, not a fault: the request was well
// formed and the router is healthy, so the caller may retry.
//
// Implementations wrap it; the adapter matches with errors.Is and answers 429
// instead of 503. Every other failure stays 503, because the caller cannot
// fix it by waiting.
var ErrRouteDecisionContended = errors.New("route decision contended")
