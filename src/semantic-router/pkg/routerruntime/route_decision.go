package routerruntime

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	// Surface names the endpoint asking. It exists because the two adapters
	// share this method and must not share every rule: the route lookup
	// resolves its target by which decision enables routes_api, and applying
	// that to the legacy consult would silently move it off its fail-closed
	// ambiguity and onto a policy its caller never selected.
	Surface RouteDecisionSurface
	// Ephemeral asks for a decision that joins no trajectory: no episode
	// lease, no episode store read or write, and nothing left behind.
	//
	// It is separate from an empty SessionID because the two are different
	// questions. An empty SessionID says the caller named no conversation;
	// this says the deployment does not keep episodes for these calls at all.
	// A caller that names a conversation on a cell with episodes switched off
	// still gets an ephemeral decision, and is told so.
	Ephemeral bool
	// WireFormat names the protocol Body is written in.
	//
	// Empty selects Anthropic Messages, which is what the legacy consult
	// contract fixed and what its shipped caller sends. A caller that relays
	// an OpenAI Responses body names it here instead, so the same runtime
	// normalizes both without the wire format being guessed from the body
	// twice, once by the adapter and once by the router.
	WireFormat llmprotocol.WireFormat
}

// RouteDecisionSurface names which endpoint a consult came from.
type RouteDecisionSurface string

const (
	// RouteDecisionSurfaceConsult is the legacy management POST /v1/route.
	// Its zero value, so an adapter that says nothing keeps the older rules.
	RouteDecisionSurfaceConsult RouteDecisionSurface = ""
	// RouteDecisionSurfaceRoutes is the public POST /v1/routes.
	RouteDecisionSurfaceRoutes RouteDecisionSurface = "routes"
)

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
	// Thinking is the reasoning configuration the chosen arm was scored
	// under.
	//
	// The budget travels with the mode on purpose. The selector requires a
	// positive reasoning budget for every thinking-on arm, so it already
	// knows the number it scored against; publishing the mode alone leaves
	// the caller to invent one, and decision and execution then diverge on
	// the exact axis the choice was made on.
	Thinking RouteThinking
	// Checkpoint identifies the artifact revision that scored this route. It
	// is the already-hashed deployment identity, never the raw pin.
	Checkpoint string
	// Alternatives are the other arms and their adjusted scores, ordered best
	// first and excluding the selected one.
	//
	// It explains a choice; it is not a failover list. A caller that cannot
	// serve WorkerModel has no supported way to pick from here, because these
	// arms were scored and rejected for this turn.
	Alternatives []RouteAlternative
	// SelectedPricing and Baseline let a caller compute its own savings from
	// its own token counts, with no reporting call back to this router.
	// Baseline is the artifact's declared reference worker, and is empty when
	// the artifact declares none.
	SelectedPricing RoutePricing
	Baseline        RouteBaseline
	// Episode reports the trajectory this decision advanced, and is nil for a
	// consult that joined no episode.
	Episode *RouteEpisode
	// Usage counts what the encoder actually read.
	Usage RouteUsage
	// Warnings names every way this router silently did something other than
	// what the request literally asked for. Always non-nil on a successful
	// decision, and usually empty.
	Warnings []string
}

// RouteThinking is the reasoning configuration of the selected arm.
type RouteThinking struct {
	// Mode is "on" or "off", the only two modes the selector's manifest
	// validates.
	Mode string
	// BudgetTokens is the reasoning budget the arm was scored under. Zero
	// when Mode is "off".
	BudgetTokens uint64
}

// RouteAlternative is one arm the selector considered and did not choose.
type RouteAlternative struct {
	Model string
	Score float64
}

// RoutePricing is a worker's rate card, per million tokens.
//
// Per million rather than per token because that is the unit every provider
// publishes and every caller reasons in; a per-token float here would be read
// back out of JSON as a number with more zeros than significant digits.
type RoutePricing struct {
	InputPerMTok  float64
	OutputPerMTok float64
	// Cache rates are part of the card, not decoration: a caller reproducing
	// their own cost from their own token counts cannot do it without them
	// wherever reads are discounted or writes are charged.
	CacheReadPerMTok  float64
	CacheWritePerMTok float64
}

// RouteBaseline is the artifact's reference worker and its rate card. It is
// the counterfactual a caller measures savings against.
type RouteBaseline struct {
	Model   string
	Pricing RoutePricing
}

// RouteEpisode reports the trajectory state this decision advanced.
type RouteEpisode struct {
	// TurnIndex is the episode revision this decision wrote.
	TurnIndex int
	// Stayed reports whether the selector kept the previous turn's arm.
	//
	// It is the field an operator watches to catch an external gateway whose
	// episode ids are not stable per conversation: continuity that silently
	// degrades shows up here as a trajectory that never stays.
	Stayed bool
}

// RouteUsage counts what the encoder read to reach this decision. It is the
// basis this call is metered on, which is why it is published rather than
// only logged.
type RouteUsage struct {
	EncodedInputTokens int
	CacheReadTokens    int
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

// ErrRouteDecisionInvalidBody marks a request the codec refused: a bad role, a
// malformed content block, a missing required field. The caller can fix it and
// retrying unchanged cannot, so an adapter answers 400 rather than 503 --
// reporting it as unavailability would blame a healthy router for a request
// that will never succeed.
var ErrRouteDecisionInvalidBody = errors.New("route decision request is invalid")
