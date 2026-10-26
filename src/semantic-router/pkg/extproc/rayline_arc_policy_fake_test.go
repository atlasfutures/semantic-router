package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fakePolicyService is a contract-faithful policy service: it serves the
// packages listing and answers each decide call with the action choose names,
// scoring every catalog action. Responses are built from the shared v1
// fixture, so they pass the router's strict decoder.
type fakePolicyService struct {
	t       *testing.T
	server  *httptest.Server
	alias   string
	sha256  string
	catalog []string

	mu sync.Mutex
	// answerAs, when set, is the package the decide responses name.
	answerAs *raylinearc.PolicyPackageRef
	// encodeUnreported answers timing_ms.encode null, as the reference
	// service does.
	encodeUnreported bool
	// cold lists no loaded package, as a service still loading its package.
	cold   atomic.Bool
	choose func(raylinearc.PolicyDecisionRequest) string
	// listedFallbackActionID and listedFallbackModel are the fallback pair
	// the packages listing names (pathfinder#3677); empty lists none.
	listedFallbackActionID, listedFallbackModel string
	// bodies are the decide request bodies as received, in order.
	bodies [][]byte
	// echo, when set, answers routing_prices_used.
	echo     func(raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed
	failWith string
	// failDetail is the detail failWith answers with ({} when nil).
	failDetail map[string]any
	// failFirst answers the next failFirstLeft decide calls with this code.
	failFirst     string
	failFirstLeft int
	// refuseCold, when set, answers 422 selection_refused /
	// stage_one_held_unknown for every decide it holds true of, as a
	// two-stage package does for a turn it would decide cold.
	refuseCold func(raylinearc.PolicyDecisionRequest) bool
	// coldReason replaces refuseCold's stage_one_held_unknown reason.
	coldReason string
	// rejectAttested, when set, answers every decide that carries
	// attested_held_action_id with 400 invalid_request, as a service that
	// rejects the field.
	rejectAttested bool
	// dropAttested, when set, closes the connection of every decide that
	// carries attested_held_action_id without answering, as a service lost
	// mid-call: the client reports the transport class.
	dropAttested bool
	// refuseAttestedWith, when set, answers every decide that carries
	// attested_held_action_id with 422 selection_refused and this reason.
	refuseAttestedWith string
	requests           []raylinearc.PolicyDecisionRequest
	// relaxedUnsupported, when set, answers every relaxed decide with 422
	// unsupported_request and this detail.reason, as a service whose
	// package cannot serve relaxed calls.
	relaxedUnsupported string
	// nullRevision answers every decide with session_revision null, as only
	// a relaxed call may.
	nullRevision bool
	// ignoreEpisodeMode answers a relaxed decide as a strict one, with a
	// session revision, as a service that predates episode_mode.
	ignoreEpisodeMode bool
	// verification, when set, is reported on the packages listing and on
	// every decide response, as a service from pathfinder#3120 on.
	verification raylinearc.PolicyVerification
	// barrier, when set, holds each decide call until that many are in
	// flight at once (or it times out), and inflight/maxInflight count them.
	barrier     int
	inflight    int
	maxInflight int
	arrived     chan struct{}
	// echoCostWeight, when set, echoes this cost weight instead of the one a
	// decide sent, as a service that decided at another point; echoNoCostWeight
	// echoes null.
	echoCostWeight   *float64
	echoNoCostWeight bool
	// omitTierPick, when set, leaves this tier point without a pick, and
	// tierPickAction, when set, picks this action at every tier point.
	// tierPickModel names a pick's model ("model-<action>" when nil).
	omitTierPick string
	// tierPickExploration gives every tier pick its own temperature-shape
	// record, as a service that draws per point (pathfinder#5664) does.
	tierPickExploration bool
	// predictionsRow, when set, answers response_fields: predictions with a
	// two-row stage-one table (an unoffered row, then each pick's model)
	// and points every tier pick at row predictionsRow.
	predictionsRow *int
	// cacheTrace, when set, answers response_fields cache_trace with it.
	cacheTrace *raylinearc.PolicyCacheTrace
	// receipt, when set, answers a request carrying a v2 member with it.
	receipt        *raylinearc.PolicyDecisionReceipt
	tierPickAction string
	tierPickModel  func(actionID string) string
	// A decide that carries selection.exploration with tau > 0 gets the
	// exploration fixture's record, with the request's tau, basis and salt
	// (pathfinder#5436). explorationHeld answers it as a held turn's record;
	// omitExploration answers with none. A relaxed held turn's record states
	// no origin, as the service keeps no memory for it. explorationChosen,
	// when set, is the record's chosen candidate.
	explorationHeld   bool
	omitExploration   bool
	explorationChosen raylinearc.PolicyStageOneCandidate
	// A listing asked for detail=capabilities advertises every capability
	// and bounds of [0, 1000], as a current service (pathfinder#5454), unless
	// a test overrides them: legacyListing answers as a service that ignores
	// the detail, advertised replaces the capabilities, and bounds replaces
	// the package's adjustable_cost_weight (Stated with nil Bounds is null).
	legacyListing bool
	advertised    []string
	bounds        *raylinearc.PolicyAdjustableCostWeight
	// listingQueries are the packages listings' queries, in order.
	listingQueries []url.Values
	// Staged turns (ADR 0155, pathfinder#5919): a decide with stage or
	// turn_delta stages the episode's turn, an append extends it once, and a
	// delta must name the staged prefix digest (session_mismatch otherwise).
	// Like the service, a delta is decided as the full request it stands
	// for: the staged messages then the delta's, under the delta's shell
	// (resolved records each decide so). refuseDelta answers every delta
	// session_mismatch; appendStatus, when set, answers every append with
	// it; appendGate, when set, holds each append until it is closed;
	// contentScores scores each action from the resolved request's content,
	// so a parity test sees any difference in what was decided on.
	staged        map[string]*fakeStagedTurn
	refuseDelta   bool
	appendStatus  int
	appendGate    chan struct{}
	appends       []raylinearc.PolicyAppendRequest
	resolved      []raylinearc.PolicyDecisionRequest
	contentScores bool
	// A decide naming signals (pathfinder #5943) is refused as the service
	// refuses it: a name outside knownSignals is signal_not_supported, one
	// outside servedSignals signal_not_admitted, and signalHeadsBroken
	// refuses every one with signal_heads_unavailable. Otherwise the answer
	// carries signalBlock(names).
	knownSignals      []string
	servedSignals     []string
	signalHeadsBroken bool
	// omitSignals answers a decide naming signals without the block.
	omitSignals bool
}

type fakeStagedTurn struct {
	messages     []json.RawMessage
	awaitsAppend bool
}

func fakePrefixDigest(episode string, count int) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", episode, count)))
	return hex.EncodeToString(sum[:])
}

// stageDecide applies a decide's staging and returns the full request it is
// decided as, or false for a delta it refuses as session_mismatch.
func (fake *fakePolicyService) stageDecide(decide raylinearc.PolicyDecisionRequest) (raylinearc.PolicyDecisionRequest, bool) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.staged == nil {
		fake.staged = map[string]*fakeStagedTurn{}
	}
	resolved := decide
	var messages []json.RawMessage
	if delta := decide.TurnDelta; delta != nil {
		base := fake.staged[decide.EpisodeIDHash]
		if fake.refuseDelta || base == nil || fakePrefixDigest(decide.EpisodeIDHash, len(base.messages)) != delta.BasePrefixDigest {
			return resolved, false
		}
		messages = append(append([]json.RawMessage{}, base.messages...), delta.Messages...)
		shell := decide.Request
		resolved.TurnDelta = nil
		if shell.Shell == raylinearc.PolicyShellInput {
			resolved.Request = raylinearc.PolicyClientRequest{Input: messages, Instructions: shell.Instructions, Tools: shell.Tools}
		} else {
			joined, err := json.Marshal(messages)
			if err != nil {
				fake.t.Errorf("fake policy service: %v", err)
			}
			resolved.Request = raylinearc.PolicyClientRequest{System: shell.System, Tools: shell.Tools, Messages: joined}
		}
	} else if decide.Request.Input != nil {
		messages = append([]json.RawMessage{}, decide.Request.Input...)
	} else {
		_ = json.Unmarshal(decide.Request.Messages, &messages)
	}
	if (decide.Stage != nil && *decide.Stage) || decide.TurnDelta != nil {
		fake.staged[decide.EpisodeIDHash] = &fakeStagedTurn{messages: messages, awaitsAppend: true}
	}
	fake.resolved = append(fake.resolved, resolved)
	return resolved, true
}

func (fake *fakePolicyService) receivedAppends() []raylinearc.PolicyAppendRequest {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([]raylinearc.PolicyAppendRequest(nil), fake.appends...)
}

// resolvedDecides are the full requests the fake decided, deltas resolved.
func (fake *fakePolicyService) resolvedDecides() []raylinearc.PolicyDecisionRequest {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([]raylinearc.PolicyDecisionRequest(nil), fake.resolved...)
}

func (fake *fakePolicyService) serveAppend(writer http.ResponseWriter, request *http.Request) {
	body, err := io.ReadAll(request.Body)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	var appended raylinearc.PolicyAppendRequest
	if err := json.Unmarshal(body, &appended); err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	fake.mu.Lock()
	fake.appends = append(fake.appends, appended)
	gate, status := fake.appendGate, fake.appendStatus
	fake.mu.Unlock()
	if gate != nil {
		<-gate
	}
	if status != 0 {
		fake.writeJSON(writer, status, map[string]any{"error": "backend_unavailable", "detail": map[string]any{}})
		return
	}
	fake.mu.Lock()
	staged := fake.staged[appended.EpisodeIDHash]
	if staged == nil || !staged.awaitsAppend {
		fake.mu.Unlock()
		fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{
			"error": "unsupported_request", "detail": map[string]any{"reason": "session_mismatch"},
		})
		return
	}
	staged.messages = append(staged.messages, appended.Messages...)
	staged.awaitsAppend = false
	count := len(staged.messages)
	fake.mu.Unlock()
	var response raylinearc.PolicyAppendResponse
	fake.readFixture("append_response.v1.json", &response)
	response.Package = raylinearc.PolicyPackageRef{Alias: fake.alias, PackageSHA256: fake.sha256}
	response.PrefixDigest = fakePrefixDigest(appended.EpisodeIDHash, count)
	response.Messages = count
	// The fixture's hold stays: VSR never reads it (ADR 0155's amendment).
	fake.writeJSON(writer, http.StatusOK, response)
}

// fakeSignalBlock is the fake's signals block: one opaque result per name.
func fakeSignalBlock(names []string) json.RawMessage {
	block := map[string]any{}
	for _, name := range names {
		block[name] = map[string]any{
			"schema": "rayline.signal." + name + ".v1", "taxonomy": "fake-v1", "scopes": map[string]any{},
			"unavailable": "no_stage_one_readout", "latency_ms": 0.01,
		}
	}
	encoded, _ := json.Marshal(block)
	return encoded
}

// signalRefusal is the service's refusal of a decide's signals, or nil.
func (fake *fakePolicyService) signalRefusal(names []string) map[string]any {
	if len(names) == 0 {
		return nil
	}
	fake.mu.Lock()
	known, served, broken := fake.knownSignals, fake.servedSignals, fake.signalHeadsBroken
	fake.mu.Unlock()
	var unknown, refused []string
	for _, name := range names {
		if !slices.Contains(known, name) {
			unknown = append(unknown, name)
		} else if !slices.Contains(served, name) {
			refused = append(refused, name)
		}
	}
	switch {
	case len(unknown) > 0:
		return map[string]any{"reason": "signal_not_supported", "signals": unknown}
	case broken:
		return map[string]any{"reason": "signal_heads_unavailable"}
	case len(refused) > 0:
		return map[string]any{"reason": "signal_not_admitted", "signals": refused}
	}
	return nil
}

func newFakePolicyService(t *testing.T, alias, sha256 string, catalog []string) *fakePolicyService {
	t.Helper()
	fake := &fakePolicyService{t: t, alias: alias, sha256: sha256, catalog: catalog}
	fake.server = httptest.NewServer(http.HandlerFunc(fake.serve))
	t.Cleanup(fake.server.Close)
	return fake
}

func (fake *fakePolicyService) URL() string { return fake.server.URL }

func (fake *fakePolicyService) chooseWith(choose func(raylinearc.PolicyDecisionRequest) string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.choose = choose
}

// failNext makes every later decide call answer with this contract error.
func (fake *fakePolicyService) failNext(code string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.failWith = code
}

// failFirstCalls makes the next n decide calls answer with this contract
// error and later ones succeed.
func (fake *fakePolicyService) failFirstCalls(code string, n int) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.failFirst, fake.failFirstLeft = code, n
}

func (fake *fakePolicyService) received() []raylinearc.PolicyDecisionRequest {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([]raylinearc.PolicyDecisionRequest(nil), fake.requests...)
}

// receivedBodies are the decide request bodies byte for byte, so a test can
// tell an omitted member from a null one.
func (fake *fakePolicyService) receivedBodies() [][]byte {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([][]byte(nil), fake.bodies...)
}

func (fake *fakePolicyService) serve(writer http.ResponseWriter, request *http.Request) {
	switch request.URL.Path {
	case "/v1/rayline/arc/policy/packages":
		// Like pathfinder's store-mode service, the package is listed only
		// for the pin that names it; a bare listing loads nothing.
		query := request.URL.Query()
		listing := fake.packages()
		if query.Get("alias") != fake.alias || query.Get("package_sha256") != fake.sha256 {
			listing.Packages = listing.Packages[:0]
		}
		fake.mu.Lock()
		fake.listingQueries = append(fake.listingQueries, query)
		if query.Get("detail") == "capabilities" && !fake.legacyListing {
			listing.Capabilities = []string{
				raylinearc.PolicyCapabilityDecisionsV2, raylinearc.PolicyCapabilityTierPoints,
				raylinearc.PolicyCapabilityResponseFields, raylinearc.PolicyCapabilityExploration,
				raylinearc.PolicyCapabilityStagedTurns,
			}
			if fake.advertised != nil {
				listing.Capabilities = append([]string{}, fake.advertised...)
			}
			bounds := raylinearc.PolicyAdjustableCostWeight{
				Stated: true, Bounds: &raylinearc.PolicyCostWeightBounds{Min: 0, Max: 1000},
			}
			if fake.bounds != nil {
				bounds = *fake.bounds
			}
			for index := range listing.Packages {
				listing.Packages[index].AdjustableCostWeight = bounds
			}
		}
		fake.mu.Unlock()
		fake.writeJSON(writer, http.StatusOK, listing)
	case "/v1/rayline/arc/policy/decide":
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		var decide raylinearc.PolicyDecisionRequest
		if err := json.Unmarshal(body, &decide); err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		fake.mu.Lock()
		fake.requests = append(fake.requests, decide)
		fake.bodies = append(fake.bodies, body)
		choose, failWith, failDetail, relaxedUnsupported := fake.choose, fake.failWith, fake.failDetail, fake.relaxedUnsupported
		refuseCold, rejectAttested, refuseAttestedWith := fake.refuseCold, fake.rejectAttested, fake.refuseAttestedWith
		dropAttested := fake.dropAttested
		if fake.failFirstLeft > 0 {
			fake.failFirstLeft--
			failWith = fake.failFirst
		}
		fake.mu.Unlock()
		if relaxedUnsupported != "" && decide.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed {
			fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{
				"error": "unsupported_request", "detail": map[string]any{"reason": relaxedUnsupported},
			})
			return
		}
		if dropAttested && decide.Selection.AttestedHeldActionID != nil {
			if hijacker, ok := writer.(http.Hijacker); ok {
				if conn, _, hijackErr := hijacker.Hijack(); hijackErr == nil {
					_ = conn.Close()
					return
				}
			}
			panic("fake policy service: connection cannot be dropped")
		}
		if rejectAttested && decide.Selection.AttestedHeldActionID != nil {
			fake.writeJSON(writer, http.StatusBadRequest, map[string]any{
				"error": "invalid_request", "detail": map[string]any{"field": "selection.attested_held_action_id"},
			})
			return
		}
		if refuseAttestedWith != "" && decide.Selection.AttestedHeldActionID != nil {
			fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{
				"error": "selection_refused", "detail": map[string]any{"reason": refuseAttestedWith},
			})
			return
		}
		if refuseCold != nil && refuseCold(decide) {
			reason := "stage_one_held_unknown"
			if fake.coldReason != "" {
				reason = fake.coldReason
			}
			fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{
				"error": "selection_refused", "detail": map[string]any{"reason": reason},
			})
			return
		}
		if refusal := fake.signalRefusal(decide.Signals); refusal != nil {
			fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{"error": "unsupported_request", "detail": refusal})
			return
		}
		resolved, ok := fake.stageDecide(decide)
		if !ok {
			fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{
				"error": "unsupported_request", "detail": map[string]any{"reason": "session_mismatch"},
			})
			return
		}
		decide = resolved
		fake.awaitBarrier()
		if failWith != "" {
			if failDetail == nil {
				failDetail = map[string]any{}
			}
			fake.writeJSON(writer, http.StatusServiceUnavailable, map[string]any{"error": failWith, "detail": failDetail})
			return
		}
		fake.writeJSON(writer, http.StatusOK, fake.decision(decide, choose(decide)))
	case "/v1/rayline/arc/policy/append":
		fake.serveAppend(writer, request)
	default:
		http.NotFound(writer, request)
	}
}

// refuseRelaxed answers relaxed decide calls with this relaxed_* reason, or
// serves them again when reason is empty.
func (fake *fakePolicyService) refuseRelaxed(reason string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.relaxedUnsupported = reason
}

// holdUntilConcurrent makes each later decide call wait until n are in flight.
func (fake *fakePolicyService) holdUntilConcurrent(n int) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.barrier, fake.arrived = n, make(chan struct{})
}

func (fake *fakePolicyService) peakConcurrency() int {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return fake.maxInflight
}

func (fake *fakePolicyService) awaitBarrier() {
	fake.mu.Lock()
	if fake.barrier == 0 {
		fake.mu.Unlock()
		return
	}
	fake.inflight++
	fake.maxInflight = max(fake.maxInflight, fake.inflight)
	arrived := fake.arrived
	if fake.inflight == fake.barrier {
		close(arrived)
	}
	fake.mu.Unlock()
	select {
	case <-arrived:
	case <-time.After(3 * time.Second):
	}
	fake.mu.Lock()
	fake.inflight--
	fake.mu.Unlock()
}

// listFallback makes the packages listing name this fallback pair.
func (fake *fakePolicyService) listFallback(actionID, model string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.listedFallbackActionID, fake.listedFallbackModel = actionID, model
}

func (fake *fakePolicyService) packages() raylinearc.PolicyPackagesResponse {
	var listing raylinearc.PolicyPackagesResponse
	fake.readFixture("packages_response.v1.json", &listing)
	listing.Packages = listing.Packages[:1]
	if fake.cold.Load() {
		listing.Packages = listing.Packages[:0]
		return listing
	}
	listing.Packages[0].Alias, listing.Packages[0].PackageSHA256 = fake.alias, fake.sha256
	fake.mu.Lock()
	listing.Packages[0].Verification = fake.verification
	// The fixture's entry names a fallback; the fake lists one only when a
	// test sets it.
	listing.Packages[0].FallbackActionID, listing.Packages[0].FallbackModel = fake.listedFallbackActionID, fake.listedFallbackModel
	fake.mu.Unlock()
	return listing
}

func (fake *fakePolicyService) decision(request raylinearc.PolicyDecisionRequest, selected string) raylinearc.PolicyDecisionResponse {
	var response raylinearc.PolicyDecisionResponse
	fake.readFixture("decision_response.v1.json", &response)
	response.Package = raylinearc.PolicyPackageRef{Alias: fake.alias, PackageSHA256: fake.sha256}
	fake.mu.Lock()
	if fake.answerAs != nil {
		response.Package = *fake.answerAs
	}
	if fake.encodeUnreported {
		response.TimingMillis.Encode = nil
	}
	nullRevision, ignoreEpisodeMode, echoCostWeight, echoNoCostWeight := fake.nullRevision, fake.ignoreEpisodeMode, fake.echoCostWeight, fake.echoNoCostWeight
	omitTierPick, tierPickAction, tierPickModel := fake.omitTierPick, fake.tierPickAction, fake.tierPickModel
	explorationHeld, omitExploration, explorationChosen := fake.explorationHeld, fake.omitExploration, fake.explorationChosen
	response.PackageVerification = fake.verification
	if fake.echo != nil {
		response.RoutingPricesUsed = fake.echo(request)
	}
	fake.mu.Unlock()
	response.Shadow = []raylinearc.PolicyShadowResult{}
	available := make(map[string]bool, len(request.Selection.AvailableActionIDs))
	for _, actionID := range request.Selection.AvailableActionIDs {
		available[actionID] = true
	}
	response.Actions = response.Actions[:0]
	for index, actionID := range fake.catalog {
		response.Actions = append(response.Actions, raylinearc.PolicyActionScore{
			ActionID: actionID, Available: available[actionID], Supported: true,
			Selected: actionID == selected, Score: float64(index) / 10,
		})
	}
	if fake.contentScores {
		content := fakeContentDigest(fake.t, request.Request)
		for index := range response.Actions {
			response.Actions[index].Score = float64(content[index%len(content)]) / 255
		}
	}
	response.Decision.SelectedActionID = selected
	response.Decision.SelectedArmID = "arm-" + selected[:8]
	// Decisions API v2 (pathfinder#5443): a sent cost weight is echoed with
	// its point_id, and each tier point gets a pick. The fake picks the
	// decision's own action at every tier, and adds one unrequested pick.
	if weight := request.Selection.OperatingPoint.CostWeight; weight != nil {
		echoed := *weight
		if echoCostWeight != nil {
			echoed = *echoCostWeight
		}
		response.Decision.OperatingPoint = raylinearc.PolicyResolvedOperatingPoint{
			Name: request.Selection.OperatingPoint.Name, PointID: fmt.Sprintf("point-%g", echoed), CostWeight: &echoed,
		}
		if echoNoCostWeight {
			response.Decision.OperatingPoint.CostWeight = nil
		}
	}
	if len(request.Selection.TierPoints) > 0 {
		picked := selected
		if tierPickAction != "" {
			picked = tierPickAction
		}
		model := "model-" + picked[:8]
		if tierPickModel != nil {
			model = tierPickModel(picked)
		}
		pick := raylinearc.PolicyTierPick{SelectedActionID: picked, SelectedArmID: "arm-" + picked[:8], Model: model}
		response.Decision.TierPicks = map[string]raylinearc.PolicyTierPick{"unrequested": pick}
		if fake.predictionsRow != nil && slices.Contains(request.ResponseFields, "predictions") {
			pass, cost, tokens := 0.9, 0.0025, 1200.0
			response.Decision.Predictions = &raylinearc.PolicyDecisionPredictions{
				ActionHorizon: "policy_continuation",
				StageOne: &raylinearc.PolicyStageOnePredictions{
					Boundary: "task_start", SelectorID: strings.Repeat("c", 64), Horizon: "rest_of_episode",
					Candidates: []raylinearc.PolicyStageOneCandidatePrediction{
						{Model: "not-offered", Offered: false},
						{
							Model: model, Offered: true, PPass: &pass, PredictedCostUSD: &cost,
							PredictedUsage: &raylinearc.PolicyPredictedUsage{UncachedInput: &tokens},
						},
					},
				},
			}
			row := *fake.predictionsRow
			pick.StageOneCandidate = &row
		}
		for _, tier := range request.Selection.TierPoints {
			if tier.Name != omitTierPick {
				response.Decision.TierPicks[tier.Name] = pick
			}
		}
	}
	if explore := request.Selection.Exploration; explore != nil && explore.Temperature != nil && !omitExploration {
		response.Decision.StageOne = fake.temperatureExplorationRecord(*explore)
		if fake.tierPickExploration {
			for name, pick := range response.Decision.TierPicks {
				pick.Exploration = fake.temperatureExplorationRecord(*explore).Exploration
				result := name
				pick.Exploration.Result = &result
				response.Decision.TierPicks[name] = pick
			}
		}
	} else if explore != nil && explore.Tau > 0 && !omitExploration {
		relaxed := request.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed && !ignoreEpisodeMode
		response.Decision.StageOne = fake.explorationRecord(*explore, explorationHeld, relaxed, explorationChosen)
	}
	if fake.receipt != nil && (request.Policy != nil || request.Event != nil || len(request.Capacity) > 0 ||
		request.TurnKey != "" || request.Selection.LastServedActionID != nil) {
		receipt := *fake.receipt
		response.Decision.Receipt = &receipt
	}
	if fake.cacheTrace != nil && slices.Contains(request.ResponseFields, "cache_trace") {
		trace := *fake.cacheTrace
		response.Decision.CacheTrace = &trace
	}
	if len(request.Signals) > 0 && !fake.omitSignals {
		response.Signals = fakeSignalBlock(request.Signals)
	}
	// A relaxed call advances no session (pathfinder#3068).
	if (request.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed && !ignoreEpisodeMode) || nullRevision {
		response.Encoding.SessionRevision = nil
		response.Encoding.SessionAction = "rebuilt"
	}
	return response
}

// explorationRecord is the exploration fixture's record for this request
// member; a held one has no decision key of its own, only its origin's.
func (fake *fakePolicyService) explorationRecord(
	explore raylinearc.PolicyExplorationRequest,
	held bool,
	relaxed bool,
	chosen raylinearc.PolicyStageOneCandidate,
) *raylinearc.PolicyDecisionStageOne {
	var fixture raylinearc.PolicyDecisionResponse
	fake.readFixture("decision_response_exploration.v1.json", &fixture)
	record := fixture.Decision.StageOne.Exploration
	tau := explore.Tau
	record.Tau, record.RequestedTau, record.Basis, record.Salt = &tau, explore.Tau, explore.Basis, explore.Salt
	if held {
		origin := "origin-" + *record.DecisionKey
		greedy, one, explored := 0.0, 1.0, false
		record.Applied, record.Reason, record.Tau, record.U = false, nil, &greedy, nil
		record.Propensity, record.Explored, record.Held = &one, &explored, true
		record.DecisionKey, record.OriginDecisionKey = nil, &origin
		if relaxed {
			record.OriginDecisionKey = nil
		}
	}
	if chosen != nil {
		record.Chosen = chosen
	}
	return &raylinearc.PolicyDecisionStageOne{Exploration: record}
}

// temperatureExplorationRecord is a temperature-shape record (pathfinder#5637)
// for this request member: the legacy fixture's draw under the requested T,
// with the shape's own members and no tau basis. A hosted record carries a
// floor that put mass in the distribution.
func (fake *fakePolicyService) temperatureExplorationRecord(
	explore raylinearc.PolicyExplorationRequest,
) *raylinearc.PolicyDecisionStageOne {
	var fixture raylinearc.PolicyDecisionResponse
	fake.readFixture("decision_response_exploration.v1.json", &fixture)
	record := fixture.Decision.StageOne.Exploration
	member := explore.Temperature
	requested, applied := member.Temperature, member.Temperature
	surface, source, training := member.Surface, member.Source, member.TrainingData
	record.Tau, record.RequestedTau, record.Basis, record.Salt = nil, 0, raylinearc.PolicyExplorationBasis{}, explore.Salt
	record.Surface, record.Source, record.TrainingData = &surface, &source, &training
	record.RequestedTemperature, record.Temperature = &requested, &applied
	record.Normalization = json.RawMessage(`{"kind":"sd","spread":1.5}`)
	record.PiSoftmax = json.RawMessage(`[0.5,0.5]`)
	clamped := false
	record.Clamped = &clamped
	if surface == raylinearc.PolicyExplorationSurfaceHosted {
		record.Floor = json.RawMessage(`{"models":["m"],"epsilon":0.01,"applied_mass":0.01,"config_digest":"d"}`)
	}
	return &raylinearc.PolicyDecisionStageOne{Exploration: record}
}

func (fake *fakePolicyService) readFixture(name string, target any) {
	fake.t.Helper()
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/" + name)
	if err != nil {
		fake.t.Fatal(err)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		fake.t.Fatal(err)
	}
}

func (fake *fakePolicyService) writeJSON(writer http.ResponseWriter, status int, value any) {
	encoded, err := json.Marshal(value)
	if err != nil {
		fake.t.Errorf("fake policy service: %v", err)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_, _ = writer.Write(encoded)
}

// fakeContentDigest digests a request's shell and conversation as the
// service's projections read them (fakeCanonicalRequest), so it is the same
// for the same content however it was sent.
func fakeContentDigest(t *testing.T, request raylinearc.PolicyClientRequest) []byte {
	sum := sha256.Sum256([]byte(fakeCanonicalRequest(t, request)))
	return sum[:]
}
