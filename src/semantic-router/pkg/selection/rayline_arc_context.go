package selection

import (
	"encoding/json"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// RaylineARCSelectionContext carries only the structured, privacy-safe inputs
// required by the artifact-owned ARC selector. It is nil for every other
// algorithm.
type RaylineARCSelectionContext struct {
	EpisodeIDHash string
	// RequestID is the request's correlation id, which joins a decision's
	// logs to the turn's llm_usage line.
	RequestID string
	// Coalesced is the decision an identical in-flight request on this
	// episode already made. When set, the selector returns it instead of
	// deciding again: the request is a resend of a turn being decided, not a
	// new turn.
	Coalesced *SelectionResult
	// EncoderVisitedReplicaIDs names the replicas this turn's encode touched,
	// recorded as soon as the encode returns rather than when a result is
	// built.
	//
	// It exists because closing a retained session needs those identities and
	// every step after the encode can fail: scoring, policy, validation, the
	// manifest mapping. A cleanup that reads them from the finished trace
	// therefore finds nothing on exactly the paths where the session was
	// retained but the lookup did not complete.
	EncoderVisitedReplicaIDs []string
	Turns                    []raylinearc.Turn
	State                    *raylinearc.EpisodeState
	InputTokens              int
	// ImageBearing reports that this turn carries image input. The turn
	// projection drops image blocks, so the encoder never sees them, but the
	// provider request does: an arm that rejects image input answers 404 and
	// not a degraded completion.
	ImageBearing bool
	// ServedModels are the models that last served this conversation as
	// the episode store remembers them past the episode (its own record
	// only; a parent session never holds it), read only when the episode holds no
	// previous arm. They name a held model for a two-stage policy package's
	// stage_one_held_unknown refusal; nil otherwise.
	ServedModels []RaylineARCServedModel
	// NonVisionArms marks, by arm ordinal, the candidates whose model card
	// declares no image input. It is indexed like CandidateModels and is nil
	// when every arm is vision-capable, which is the unmarked default.
	NonVisionArms []bool
	// DisabledArms marks, by arm ordinal, the candidates an operator has taken
	// out of service. Unlike NonVisionArms it is not a fact about this turn:
	// it holds for every turn until the card changes. Nil when no arm is
	// marked, which is the unmarked default.
	DisabledArms []bool
	// PrivacyExcludedArms marks, by arm ordinal, the candidates outside a
	// private request's allowed set, as its trusted gateway headers state it
	// (extproc/rayline_arc_privacy.go). Nil for a standard request, which
	// leaves selection exactly as it was.
	PrivacyExcludedArms []bool
	// PrivacyArmEndpoints is, by arm ordinal, each OpenRouter arm's endpoint
	// constraint on a private request whose gateway listed the endpoints
	// each allowed arm may use (x-rayline-privacy-allowed-endpoints); nil
	// otherwise, and nil at an arm that is masked already or not on
	// OpenRouter. The selector masks an arm none of whose reachable
	// endpoints the gateway allows, once it holds the routing-price feed
	// that says what the arm reaches.
	PrivacyArmEndpoints []*raylinearc.ArmEndpoints
	// EndpointFeed is the routing-price feed snapshot the selector read for
	// the endpoint mask, nil when the cell has none; the decision carries it
	// on (RaylineARCTrace.EndpointFeed) so the endpoint pick reads the same.
	EndpointFeed *raylinearc.RoutingPriceSettlement
	// ContextTokens is the request's conservative token estimate, the same
	// number the generic context filter compares a card's window against.
	// Zero when no estimate was made, which excludes nothing.
	ContextTokens int
	// OverContextArms marks, by arm ordinal, the candidates whose model card
	// declares a context_window_size smaller than ContextTokens. It is nil
	// when no arm is over its window, which leaves selection exactly as it
	// was, and it replaces the generic context filter for an ARC decision:
	// that filter removes a candidate from the list, and the arm list is
	// positional, so the arm has to be masked where it stands instead.
	OverContextArms []bool
	// RequiredCapabilities names what this turn needs an arm to hold, from
	// llmprotocol.RequiredRoutingCapabilities. It is empty for almost every
	// turn: only a server tool or an image inside a tool result puts a name
	// here.
	RequiredCapabilities []string
	// IncapableArms marks, by arm ordinal, the candidates whose model card
	// does not claim every capability this turn requires. It is nil when the
	// turn requires none, which leaves selection exactly as it was.
	IncapableArms []bool
	// ToolLoopForeignArms marks, by arm ordinal, the candidates outside the
	// model family of ToolLoopArm, the episode's previous arm, when this turn
	// answers a tool call that arm made and the hold is configured. Nil
	// otherwise, which leaves selection exactly as it was. Unlike the other
	// masks it yields: when it would leave no eligible arm it is not applied.
	ToolLoopForeignArms []bool
	ToolLoopArm         int
	ToolLoopFamily      string
	PreparationFailure  string
	// RawRequest and RequestFormat are the client body as received and its
	// policy-service request format (anthropic_messages or openai_chat), set
	// only for the policy-service mode.
	RawRequest    []byte
	RequestFormat string
	// PolicyInput and PolicyInstructions are a Responses request materialized
	// for the policy service; set only for request format openai_responses.
	PolicyInput        []json.RawMessage
	PolicyInstructions *string
	// PolicySignalHeaders holds the request's explicit turn-signal headers
	// (call kind, compaction ordinal), keyed by lower-case name.
	PolicySignalHeaders map[string]string
	// RouteControls are a POST /v1/routes lookup's public routing controls
	// (pathfinder#5664): its result points, caller prices and exploration.
	// Nil on every routed turn and on a lookup that names none.
	RouteControls *RaylineARCRouteControls
	// TurnCostWeight is a generation turn's own cost weight (configurable
	// ARC's routing.value_usd, 1 / V), sent in place of the cell's; nil
	// keeps the cell's.
	TurnCostWeight *float64
	// TurnSignals is a generation turn's routing.signals, forwarded as the
	// decide request's signals; nil keeps the request without them.
	TurnSignals []string
	// PolicyCallKind and PolicyCallKindSource are the request's call kind as
	// classified before the episode was read; selection serves the request as
	// that kind. Empty when the request was not classified ahead of selection.
	PolicyCallKind       raylinearc.PolicyCallKind
	PolicyCallKindSource string
}

// RaylineARCTrace records bounded, privacy-safe artifact policy diagnostics.
// RaylineARCRouteResult is one checked result pick: the arm its action
// dispatches and that action's level.
type RaylineARCRouteResult struct {
	Pick          raylinearc.PolicyTierPick
	Arm           int
	ThinkingLevel string
}

// RaylineARCRouteControls is what a Decisions lookup adds to its decide
// request. Points are asked as selection.tier_points beside any the cell
// configures, each a result the lookup returns; Exploration replaces the
// request's exploration (surface decisions, never training data); Prices
// replace the live feed's rates; with Exclusive only the actions of
// PricedModels are offered.
type RaylineARCRouteControls struct {
	Points       []raylinearc.PolicyTierPoint
	Exploration  *raylinearc.PolicyExplorationRequest
	Prices       *raylinearc.PolicyRoutingPrices
	PricedModels []string
	Exclusive    bool
	// PriceSHA256 is the caller price world's public hash, which the
	// service's caller-mode echo must restate.
	PriceSHA256 string
	// Predictions asks the service for its stage-one prediction table
	// (routing.response_fields: predictions, pathfinder#5717), sent only to
	// a service that advertises the capability.
	Predictions bool
	// Execution is the caller's own execution state (routing.execution,
	// pathfinder #5874); nil without one.
	Execution *RaylineARCRouteExecution
	// Policy is routing.policy as sent (hybrid routing v2, pathfinder
	// #5904) and ExecutionV2 the v2 envelope; each nil without one.
	Policy      *raylinearc.PolicyMember
	ExecutionV2 *RaylineARCRouteExecutionV2
	// Signals is routing.signals as sent (pathfinder #5943), forwarded as
	// the decide request's signals; nil without one.
	Signals []string
}

// RaylineARCRouteExecutionV2 is a v2 routing.execution as decide-request
// members (arc_execution_v2.decide_inputs_v2) plus what the package-side
// checks need (arc_execution_v2.bind_v2). Never written back to any state.
type RaylineARCRouteExecutionV2 struct {
	ContextEpoch string
	Attribution  []raylinearc.PolicyAttribution
	// LastServedActionID (and the message it names, nil when none) is the
	// executor's truthful record of what last ran: accepted whenever its
	// action is in the package, offered or not.
	LastServedActionID *string
	LastServedMessage  *int
	// HoldActionID is a mandatory action hold, nil without one.
	HoldActionID *string
	CacheContext *cachecontext.PolicyCacheContext
	Event        raylinearc.PolicyExecutorEvent
	Capacity     []raylinearc.PolicyActionCapacity
	// Infeasible are the targets the request's own capacity marks fits:
	// false (any provenance).
	Infeasible []string
	TurnKey    string
	Reconsult  *string
	// ActionIDs are every action the envelope names, checked against the
	// serving package's catalog.
	ActionIDs                                          []string
	ExpectPackage, ExpectThinkingControls, ExpectOffer *string
}

// RaylineARCRouteExecution is a Decisions caller's execution context as
// decide-request members. It replaces what VSR would derive from its own
// episode (epoch, attribution, held model, cache evidence) for this lookup
// only, and is never written back to any state.
type RaylineARCRouteExecution struct {
	ContextEpoch string
	Attribution  []raylinearc.PolicyAttribution
	// HoldScope is "", "model" (attested_held_action_id) or "action"
	// (held_action_id); HoldActionID names the action.
	HoldScope    string
	HoldActionID string
	// CacheContext is the admitted cache evidence, nil when the caller sent
	// no workers. The service decides the cache mode itself (its cell's
	// ARC_CALLER_CACHE_MULTIWARM switch); the caller never forces it.
	CacheContext *cachecontext.PolicyCacheContext
	// ActionIDs are every action the context names (attribution, hold,
	// workers), each checked against the serving package's catalog.
	ActionIDs []string
	// Expect* are the identity the caller pinned (nil: not pinned).
	ExpectPackage, ExpectThinkingControls, ExpectOffer *string
}

// RaylineARCRouteIdentity is what a Decisions answer states about the
// identity it was decided under (pathfinder #5874 "Result identity").
type RaylineARCRouteIdentity struct {
	PackageSHA256          string
	ThinkingControlsSHA256 string
	OfferSHA256            string
}

// RaylineARCArmAction is the action an arm's score comes from: its best
// available action, and that action's thinking level.
type RaylineARCArmAction struct {
	ActionID      string
	ThinkingLevel string
}

type RaylineARCTrace struct {
	// PolicyActionID, PolicyArmID and ThinkingLevel are the policy-service
	// decision: the package action, the trained arm the service scored, and
	// the level the action binds. Empty outside the policy-service mode.
	PolicyActionID string
	PolicyArmID    string
	ThinkingLevel  string
	// PolicyActionModel is the chosen action's trained model and
	// WorkerProviderModel the provider model its worker serves; they differ
	// when an action is remapped to another provider.
	PolicyActionModel   string
	WorkerProviderModel string
	// WorkerRoute names the worker's route (worker, backend endpoint and
	// provider model) for the fallback's route exclusions.
	WorkerRoute string
	// PolicyLatency is the decide call's round trip as the router timed it;
	// EncoderLatency stays the encode alone.
	PolicyLatency time.Duration
	// PolicyDecisionMode is how a delta cell's staged turn was decided (ADR
	// 0155): delta (on a turn_delta) or sync (sent in full). Empty on a sync
	// cell, whose records are as before.
	PolicyDecisionMode string
	// PolicyStaging is what a delta cell's strict main turn hands its
	// episode transaction: the append to post before the lease is released.
	// Opaque here (extproc's *policyStaging); never copied to a coalesced
	// resend.
	PolicyStaging any
	// EncoderLatencyUnknown marks an encode time the policy service did not
	// report: EncoderLatency is then not a measurement, and is neither
	// logged nor observed.
	EncoderLatencyUnknown bool
	// TokenCountsUnknown marks a decision whose serialized and full-history
	// token counts no one measured (an encoder capacity hold whose refusal
	// reported no count): the counts are then not zeros, and are logged as
	// null.
	TokenCountsUnknown bool
	// PolicyNextState is the ledger and epoch to commit with this turn.
	PolicyNextState *raylinearc.PolicyEpisodeState
	// PolicyTurnState is the context this turn was decided in, before its
	// reply: the state a refusal excludes the refusing model in (ADR 0120).
	PolicyTurnState *raylinearc.PolicyEpisodeState
	// PolicySideCall marks a call outside the main conversation: it commits
	// nothing to the episode, neither a turn nor a ledger entry.
	PolicySideCall bool
	// PolicyBoundary is the model decision this turn took at a schedule
	// boundary, to be stored before dispatch; nil off a boundary, for a side
	// call, and for a retry that reused a stored one.
	PolicyBoundary *raylinearc.PolicyBoundaryDecision
	// PolicyPackageSHA256 is the package that decided, and
	// PolicyPredictedCostUSD the chosen action's predicted cost when the
	// service returned one.
	PolicyPackageSHA256    string
	PolicyPredictedCostUSD *float64
	// RouteResults are a Decisions lookup's results, keyed by result name
	// (swift, sage, custom): the service's tier pick at each requested
	// point, each checked as the decision's own action is. Nil off a lookup
	// with route controls.
	RouteResults map[string]RaylineARCRouteResult
	// RouteExploration is the service's exploration record for the
	// lookup's own decision under route controls' exploration. Kept apart
	// from PolicyExploration, which feeds llm_usage and the episode's
	// boundary origin: Decisions draws are never training data.
	RouteExploration *raylinearc.PolicyExplorationRecord
	// TurnCostWeightApplied: the decision was taken, and echoed, at the
	// turn's own cost weight (configurable ARC's routing.value_usd). False
	// on a held turn, which decides nothing.
	TurnCostWeightApplied bool
	// RoutePredictionsApplied: route controls asked for predictions and the
	// service answered them (RoutePredictions, checked against every tier
	// pick's row). False when the service does not advertise them.
	RoutePredictionsApplied bool
	RoutePredictions        *raylinearc.PolicyDecisionPredictions
	// PolicyArmActions is, per arm, the action its score came from; empty
	// for an arm the decision did not score.
	PolicyArmActions []RaylineARCArmAction
	// PolicyArmOffered is, per arm, whether the package's stage one offers
	// its model; nil when the package states no by-model offer. An arm it
	// does not offer is never published as an alternative.
	PolicyArmOffered []bool
	// PolicyCostWeight is the stage-one cost weight the service echoed for
	// the decision, on every policy-service cell; nil when it echoed none,
	// never read as 0.
	PolicyCostWeight *float64
	// RouteIdentity is a Decisions lookup's identity (package, thinking
	// controls, offer), set on every policy-service lookup with route
	// controls; RouteExecutionApplied marks a decision taken with the
	// caller's routing.execution, and RouteCacheTrace is the service's
	// decision.cache_trace for it (nil when the service sent none).
	RouteIdentity         *RaylineARCRouteIdentity
	RouteExecutionApplied bool
	RouteCacheTrace       *raylinearc.PolicyCacheTrace
	// RouteReceipt is the service's decision.receipt for a lookup that sent
	// routing.policy or a v2 routing.execution (nil when the service sent
	// none).
	RouteReceipt *raylinearc.PolicyDecisionReceipt
	// Signals is the service's signals block for a decision whose request
	// named signals, as the service wrote it; nil otherwise (and on a turn
	// the service did not decide, such as a hold).
	Signals json.RawMessage
	// PolicyOperatingPoint is the operating point the service decided at, as
	// it echoed it (point_id names its cost weight), and PolicyTierPicks the
	// decision at each configured tier point. Both are set only for a cell
	// with decisions_v2; the tier picks are recorded, never served.
	PolicyOperatingPoint *raylinearc.PolicyResolvedOperatingPoint
	PolicyTierPicks      map[string]raylinearc.PolicyTierPick
	// PolicyOperatingPointName is the decisions_v2 name the cell sent, when
	// it sets one; the echoed name is a record label and is not checked.
	PolicyOperatingPointName string
	// PolicyTierPicksMissing names the configured tier points the response
	// carried no pick for, so a partial set is never read as complete.
	PolicyTierPicksMissing []string
	// PolicyTierPicksInvalid names the configured tier points whose pick
	// names an action the cell cannot dispatch or did not offer, or a model
	// its binding does not serve; such a pick is not kept.
	PolicyTierPicksInvalid []string
	// PolicyExploration is the service's record of the turn's Stage-1
	// exploration draw (decision.stage_one.exploration, pathfinder#5436), nil
	// when the request carried no exploration or the service returned none.
	// PolicyExplorationSent marks a request that carried it, and
	// PolicyExplorationExcluded names why an exploration cell's turn did not
	// (privacy mode, training-data opt-out, an eval cell, ...).
	PolicyExploration         *raylinearc.PolicyExplorationRecord
	PolicyExplorationSent     bool
	PolicyExplorationExcluded string
	// PolicyExplorationHeldKey is, on a relaxed held turn the service stated
	// no origin for, the decision key of the episode's latest Stage-1
	// boundary from VSR's own record, when the held model is the one that
	// boundary chose; nil otherwise.
	PolicyExplorationHeldKey *string
	// RoutingPricesStatus is the routing price cache status the request
	// read (fresh, stale, expired, absent); empty when live prices are not
	// configured. RoutingPricesUsed is the service's echo, nil when it sent
	// none. Together they are the decision's price audit.
	RoutingPricesStatus string
	RoutingPricesUsed   *raylinearc.PolicyRoutingPricesUsed
	// SettlementPrices is the routing price cache the call is settled at,
	// set on every policy-service decision of a cell with routing_prices,
	// a capacity hold included: the feed then prices the call's usage
	// record where it can (raylinearc.RoutingPriceSettlement).
	SettlementPrices *raylinearc.RoutingPriceCache
	// EndpointFeed is the feed snapshot the decision's endpoint mask read
	// (RaylineARCSelectionContext.EndpointFeed): the endpoint pick, its
	// re-sends and x-vsr-served-endpoint read this one, so a refresh between
	// the mask and the dispatch cannot give the two different endpoints.
	EndpointFeed *raylinearc.RoutingPriceSettlement
	// ArtifactID and ArtifactRevision hold SHA256-derived hashes of the
	// deployment-private artifact identity, never the raw pins.
	ArtifactID          string
	ArtifactRevision    string
	EncoderRevision     string
	EpisodeIDHash       string
	SelectedArm         int
	PreviousArm         *int
	RawScores           []float32
	AdjustedScores      []float32
	SwitchCostUSD       []float64
	CacheMissTokens     []int
	Stayed              bool
	UpgradeExemptions   []bool
	StayUpgradeExempted bool
	// ExcludedArms marks the arms that were not scored. In artifact mode
	// that is the hard constraint; in policy mode it is also every arm a
	// held or retained turn did not offer, so read MaskedArms for the masks.
	ExcludedArms []bool
	// OfferKind says why a policy-mode turn offered what it did: "fresh" (a
	// schedule boundary, every unmasked arm), "held" (between boundaries, or
	// a side call: the held arm only), "retained" (a retry at a boundary
	// already decided), "fallback" (a held arm an exclusion released, decided
	// again among the rest), "derived_hold" (a cold turn narrowed to a
	// derived model), "capacity_hold" (the service's encoder could not hold
	// the context, so the held action stood) or "media_hold" (the service's
	// projection refused media it cannot read, so the held action stood).
	// Empty outside policy mode.
	OfferKind string
	// MaskedArms marks only the arms a real mask removed from a policy-mode
	// offer: the hard mask (vision, capability, context, operator), the
	// tool-loop family hold, and the fallback's model and route exclusions.
	// Nil outside policy mode.
	MaskedArms           []bool
	SerializedTokens     int
	FullHistoryTokens    int
	TruncatedTokens      int
	CachedPrefixTokens   int
	RetainedPrefixTokens int
	AppendedTokens       int
	SessionAction        string
	SessionRevision      int
	EncoderLatency       time.Duration
	EncoderReplicaIndex  int
	EncoderAttempts      int
	EncoderFailover      bool
	// EncoderReplicaID and EncoderVisitedReplicaIDs are internal transaction
	// state. They must never be logged or exported as metric labels.
	EncoderReplicaID         string
	EncoderVisitedReplicaIDs []string
	// PolicySession is the Modal sticky session a sessioned policy Server
	// decided this turn on, which the episode commit persists; nil when the
	// turn changes none. Internal transaction state: never logged.
	PolicySession *raylinearc.PolicySession
	// PolicySessionReplaced is the turn's session binding when the turn moved
	// off the session the episode records: the episode commit ends that one
	// once it has recorded PolicySession (TerminateReplaced), never before.
	// Internal transaction state.
	PolicySessionReplaced *raylinearc.PolicySessionBinding
	// PolicySessions is how many sessions the turn's calls used, and
	// PolicySessionMoved whether the turn left a session that was gone.
	PolicySessions     int
	PolicySessionMoved bool
	// PolicySessionCleared drops a session the episode state holds from a
	// cell that no longer decides through sessions, so a router that predates
	// them can read the record again.
	PolicySessionCleared bool
	// TurnIndex is the episode position this selection was made at: the
	// number of turns the episode had committed before it.
	TurnIndex uint64
}

// RaylineARCServedModel is one remembered serving model and where the
// record came from: "session_record".
type RaylineARCServedModel struct {
	Source string
	Model  string
}
