package execution

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

type admittedAction struct {
	action     Action
	deployment Deployment
	control    thinkingcontrol.Control
	cell       *thinkingcontrol.Cell
	format     string
}

type Server struct {
	config             Config
	provider           DecisionProvider
	engine             *protocolcodec.Engine
	client             *http.Client
	actions            map[string]admittedAction
	counters           map[string]cachecontext.PromptCounter
	states             stateStore
	token              string
	instance           string
	capabilities       Capabilities
	generationAttempts atomic.Int64
	now                func() time.Time
}

func NewServer(config Config, registry *thinkingcontrol.Registry, provider DecisionProvider, token string) (*Server, error) {
	if config.MaxGenerationAttempts < 0 {
		return nil, fmt.Errorf("max_generation_attempts cannot be negative")
	}
	if len(token) < 32 {
		return nil, fmt.Errorf("execution auth token must contain at least 32 characters")
	}
	if provider == nil || registry == nil {
		return nil, fmt.Errorf("decision provider and thinking registry are required")
	}
	instance := make([]byte, 16)
	if _, err := rand.Read(instance); err != nil {
		return nil, err
	}
	server := &Server{config: config, provider: provider, engine: protocolcodec.NewBuiltinEngine(), actions: make(map[string]admittedAction), states: stateStore{branches: make(map[string]*branchState)}, token: token, instance: hex.EncodeToString(instance), now: time.Now,
		client: &http.Client{Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, ResponseHeaderTimeout: 2 * time.Minute, IdleConnTimeout: 90 * time.Second}, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	deployments := make(map[string]Deployment)
	for _, deployment := range config.Deployments {
		if deployment.ID == "" || deployment.WireModel == "" || deployment.RuntimeIdentity == "" || deployment.MaxOutputTokens <= 0 {
			return nil, fmt.Errorf("deployment needs id, wire_model, runtime_identity and positive max_output_tokens")
		}
		if _, exists := deployments[deployment.ID]; exists {
			return nil, fmt.Errorf("duplicate deployment id %q", deployment.ID)
		}
		if err := validateEndpoint(deployment.URL); err != nil {
			return nil, fmt.Errorf("deployment %s: %w", deployment.ID, err)
		}
		if deployment.Kind != "local" && deployment.Kind != "cloud" {
			return nil, fmt.Errorf("deployment %s kind must be local or cloud", deployment.ID)
		}
		if err := validateCloudTransport(deployment); err != nil {
			return nil, fmt.Errorf("deployment %s: %w", deployment.ID, err)
		}
		if err := validateProviderPin(deployment); err != nil {
			return nil, fmt.Errorf("deployment %s: %w", deployment.ID, err)
		}
		if err := deployment.CapacityEstimate.validate(); err != nil {
			return nil, fmt.Errorf("deployment %s: %w", deployment.ID, err)
		}
		if deployment.Kind == "local" && deployment.AuthEnv != "" &&
			(deployment.AuthEnv == config.Decisions.AuthEnv || (os.Getenv(deployment.AuthEnv) != "" && os.Getenv(deployment.AuthEnv) == os.Getenv(config.Decisions.AuthEnv))) {
			return nil, fmt.Errorf("local inference cannot receive the hosted Decisions credential")
		}
		if deployment.AuthEnv != "" && os.Getenv(deployment.AuthEnv) == "" {
			return nil, fmt.Errorf("deployment %s credential environment is unset", deployment.ID)
		}
		deployments[deployment.ID] = deployment
	}
	for _, action := range config.Actions {
		if !validSHA256(action.ID) || action.Worker == "" || action.Model == "" {
			return nil, fmt.Errorf("action identity is incomplete")
		}
		if _, exists := server.actions[action.ID]; exists {
			return nil, fmt.Errorf("action %s binds more than one deployment", action.ID)
		}
		control, err := thinkingcontrol.ParseControl(action.Control)
		if err != nil {
			return nil, fmt.Errorf("action control: %w", err)
		}
		if control.ID() != action.ControlID {
			return nil, fmt.Errorf("action control identity mismatch")
		}
		if action.ReportedModel != "" && !sameModelName(action.Model, action.ReportedModel) {
			return nil, fmt.Errorf("reported_model must identify the bound action model")
		}
		if action.DeploymentID == "" {
			alias := config.TestAlias
			_, knownControl := registry.Control(action.ControlID)
			if alias == nil || alias.SelectedModel != action.Model || action.ReportedModel == "" || !knownControl {
				return nil, fmt.Errorf("a selection-only binding requires an explicit alias source, reported_model and registry-known control")
			}
			server.actions[action.ID] = admittedAction{action: action, control: control}
			continue
		}
		deployment, ok := deployments[action.DeploymentID]
		if !ok {
			return nil, fmt.Errorf("action names unknown deployment %q", action.DeploymentID)
		}
		if !sameModelName(action.Model, deployment.RegistryModel) {
			return nil, fmt.Errorf("action model does not match its deployment; substitutions require explicit test_alias")
		}
		if err := validateModelPinControl(deployment, control); err != nil {
			return nil, fmt.Errorf("deployment %s: %w", deployment.ID, err)
		}
		if err := validateInbandSystem(deployment, control); err != nil {
			return nil, fmt.Errorf("deployment %s: %w", deployment.ID, err)
		}
		format, err := registryFormat(deployment.Protocol)
		if err != nil {
			return nil, err
		}
		cell, err := registry.Admit(deployment.RegistryModel, deployment.RegistryProvider, format, control, config.AllowExperimentalControls)
		if err != nil {
			return nil, fmt.Errorf("deployment %s control admission: %w", deployment.ID, err)
		}
		if deployment.ProviderPin != nil {
			if err := cell.AdmitEndpoint(control, deployment.ProviderPin.Only[0]); err != nil {
				return nil, fmt.Errorf("deployment %s endpoint admission: %w", deployment.ID, err)
			}
		}
		server.actions[action.ID] = admittedAction{action, deployment, control, cell, format}
	}
	if len(server.actions) == 0 {
		return nil, fmt.Errorf("at least one admitted action is required")
	}
	if err := server.validateTestAlias(); err != nil {
		return nil, err
	}
	var err error
	server.capabilities, err = server.executionCapabilities()
	if err != nil {
		return nil, err
	}
	return server, nil
}

func sameModelName(model, qualified string) bool {
	_, bare, namespaced := strings.Cut(qualified, "/")
	return model == qualified || (namespaced && model == bare)
}

func (server *Server) InstanceID() string { return server.instance }

func (server *Server) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	credential := strings.TrimPrefix(request.Header.Get("Authorization"), "Bearer ")
	if subtle.ConstantTimeCompare([]byte(credential), []byte(server.token)) != 1 {
		writeJSON(writer, http.StatusUnauthorized, map[string]any{"error": map[string]string{"type": "authentication_error", "message": "invalid execution token"}})
		return
	}
	if request.Method == http.MethodGet && request.URL.Path == "/healthz" {
		writeJSON(writer, http.StatusOK, map[string]any{"ready": true, "contract_version": ContractVersion, "instance_id": server.instance, "package_sha256": server.config.PackageSHA256, "capabilities": server.capabilities})
		return
	}
	format, ok := routeFormat(request.URL.Path)
	if !ok || request.Method != http.MethodPost {
		http.NotFound(writer, request)
		return
	}
	server.execute(writer, request, format)
}

func (server *Server) execute(writer http.ResponseWriter, incoming *http.Request, format llmprotocol.WireFormat) {
	ctx := incoming.Context()
	body, err := io.ReadAll(http.MaxBytesReader(writer, incoming.Body, 64<<20))
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "invalid_body", "request body exceeds limits or could not be read")
		return
	}
	// Current and v1 modes keep their pre-v2 behaviour exactly: the request
	// bytes are untouched and no event channel is read.
	v2 := server.config.Decisions.apiMode() == DecisionsExecutionContextV2
	var explicitEvent *ExecutionEvent
	if v2 {
		body, explicitEvent, err = extractExplicitEvent(incoming.Header, body)
		if err != nil {
			server.writeError(writer, format, http.StatusBadRequest, "invalid_execution_event", err.Error())
			return
		}
	}
	// routing.signals (all modes) is caller metadata for Decisions, never sent
	// upstream; a body without it is untouched.
	body, signals, err := extractRoutingSignals(body)
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "invalid_routing_signals", err.Error())
		return
	}
	wireRequest := parseWireRequest(format, body)
	markerSidecall := v2 && explicitEvent == nil && wireRequest.isSidecall()
	ephemeral := false
	session, branch, err := requestIdentity(incoming.Header, body)
	if err != nil {
		// A harness's own auxiliary call (Pi's compaction summary call sends no
		// session header) runs statelessly: it can advance nothing anyway.
		if !(markerSidecall || explicitEvent != nil && explicitEvent.is(EventSidecall)) {
			server.writeError(writer, format, http.StatusBadRequest, "session_identity_required", err.Error())
			return
		}
		session, branch, ephemeral = "sidecall:"+fingerprint(body), "main", true
	}
	request, envelope, _, err := server.engine.DecodeRequestForMutation(format, body)
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	roles, err := wireMessageRoles(format, body)
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if err := validateCompactionIndexes(explicitEvent, roles); err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "invalid_execution_event", err.Error())
		return
	}
	prefix, err := requestPrefix(request)
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	var state *branchState
	if ephemeral {
		state = newBranchState()
		<-state.lease
	} else if state, err = server.states.acquire(ctx, session, branch); err != nil {
		server.writeError(writer, format, http.StatusServiceUnavailable, "execution_state_unavailable", err.Error())
		return
	}
	defer state.release()
	fresh := state.pending == nil && state.completed == 0
	if fresh && hasExecutionHistory(request) && !markerSidecall && !(explicitEvent != nil && explicitEvent.is(EventSidecall)) {
		server.writeError(writer, format, http.StatusConflict, "execution_state_not_found", "this session has assistant or tool history but no execution ledger; start a fresh session")
		return
	}
	wire := wireMessageFingerprints(format, body)
	rewrite := len(state.prefix) > 0 && !extendsPrefix(state.prefix, prefix)
	// v2: a transition reset by an earlier attempt that never committed (its
	// consult or dispatch failed) is still open; detect it again on this request.
	uncommitted := v2 && state.carried != nil && len(state.prefix) == 0
	var event ExecutionEvent
	switch {
	case !v2:
		event = inferEvent(request, fresh, branch)
	case explicitEvent != nil:
		event = *explicitEvent
	case markerSidecall:
		sidecall := EventSidecall
		event = ExecutionEvent{Kind: &sidecall, Basis: BasisMarkerInferred}
	default:
		// The primary v2 source: what the unmodified harness already sends,
		// compared with the last request this branch saw.
		previous := state.seen
		if state.carried != nil {
			previous = state.carried.wire
		}
		event = inferV2Event(wireRequest, previous, wire, rewrite || uncommitted, fresh && branch != "main" && server.states.hasOtherBranch(session, branch))
	}
	// A side call is decided with the held model and never advances this
	// branch: no epoch change, retained decision or committed evidence.
	sidecall := event.is(EventSidecall)
	if !sidecall && (rewrite || uncommitted) {
		// The epoch is bumped once per transition; an uncommitted retry
		// recomputes the retained attribution for the transcript it carries.
		if rewrite {
			state.resetEpoch(v2)
		}
		var retained []attributedTurn
		if event.is(EventCompaction) && state.carried != nil {
			retained = remapRetainedAttribution(state.carried.ledger, state.carried.wire, wire, roles, format, event.Compaction)
		}
		state.ledger = retained
	}
	if !sidecall {
		state.seen = wire
	}
	requestFingerprint := fingerprint(body)
	if signals != nil {
		// Signal names leave the body but still identify the request: a retry
		// naming other signals consults Decisions again.
		requestFingerprint = fingerprint(append(append(append([]byte(nil), body...), 0), signals...))
	}
	requestID := incoming.Header.Get("Idempotency-Key")
	if requestID == "" {
		requestID = requestFingerprint
	}
	pending := state.pending
	if sidecall {
		pending = nil
	}
	if pending != nil && pending.requestID == requestID && pending.fingerprint != requestFingerprint {
		server.writeError(writer, format, http.StatusConflict, "retry_body_changed", "an idempotency key cannot name different requests")
		return
	}
	if pending != nil && pending.fingerprint == requestFingerprint && (pending.delivered || pending.completed) {
		server.writeError(writer, format, http.StatusConflict, "execution_already_observed", "this request already produced output; reconcile the transcript before retrying")
		return
	}
	reuse := make(map[string]*cachecontext.MeasuredPrompt)
	var measuredEvidence ExecutionContext
	if pending == nil || pending.fingerprint != requestFingerprint {
		evidence, measured, err := server.measureExecutionContext(ctx, body, format, roles, state)
		if err != nil {
			server.writeError(writer, format, http.StatusBadGateway, "prefix_measurement_failed", err.Error())
			return
		}
		for deployment, prompt := range measured {
			reuse[deployment] = prompt
		}
		measuredEvidence = evidence
		input := DecisionInput{Body: body, Format: format, SessionID: session, BranchID: branch, RequestID: requestID, State: evidence, Event: event, Signals: signals}
		if v2 {
			facts, prompts, err := server.measureCapacity(ctx, body, format, state)
			if err != nil {
				server.writeError(writer, format, http.StatusBadGateway, "capacity_measurement_failed", err.Error())
				return
			}
			input.Capacity = facts
			for actionID, prompt := range prompts {
				reuse["action:"+actionID] = prompt
			}
			// A prompt no deployment can hold is refused before Decisions or
			// any runtime is contacted; it can only be compacted.
			if server.writeContextExceeded(writer, format, facts) {
				return
			}
		}
		decision, err := server.provider.Decide(ctx, input)
		if err != nil {
			server.writeDecisionError(writer, format, err, input.Capacity)
			return
		}
		pending = &pendingTurn{fingerprint: requestFingerprint, requestID: requestID, decision: decision, input: input}
		if !sidecall {
			state.pending = pending
		}
	}
	var (
		selection   Decision
		selected    admittedAction
		decision    Decision
		aliasReason string
		prepared    *preparedDispatch
		prompt      *cachecontext.MeasuredPrompt
		upstream    *http.Response
	)
	for {
		selection = pending.decision
		selected, decision, aliasReason, err = server.executionFor(selection)
		if err != nil {
			server.writeError(writer, format, http.StatusBadGateway, "decision_identity_mismatch", err.Error())
			return
		}
		prepared, err = server.prepare(request, envelope, selected, state)
		if err != nil {
			server.writeError(writer, format, http.StatusBadRequest, "execution_plan_failed", err.Error())
			return
		}
		if previous := reuse["action:"+selected.action.ID]; previous != nil {
			reuse[selected.deployment.ID] = previous
		}
		prompt, err = server.measurePreparedPrompt(ctx, selected, prepared.body, state, reuse)
		if err != nil {
			server.writeError(writer, format, http.StatusBadGateway, "prefix_measurement_failed", err.Error())
			return
		}
		// The pre-dispatch capacity check. The core may veto the final
		// prepared request with a reason; the service reselects. A refusal
		// already recorded for this turn is never dispatched again.
		if known := pending.input.Capacity[selected.action.ID]; v2 && known != nil && !known.Fits {
			if !server.reconsult(ctx, writer, format, pending, selected.action.ID, decision.RouteID, "veto", known) {
				return
			}
			continue
		}
		if v2 {
			if fact := capacityFact(selected.deployment, prompt, prepared.outputLimit); fact != nil && !fact.Fits {
				if !server.reconsult(ctx, writer, format, pending, selected.action.ID, decision.RouteID, "veto", fact) {
					return
				}
				continue
			}
		}
		if !server.reserveGenerationAttempt() {
			server.writeError(writer, format, http.StatusTooManyRequests, "generation_limit_reached", "execution process reached its configured generation-attempt limit")
			return
		}
		upstream, err = server.upstream(ctx, selected, decision, session, branch, prepared.body)
		if err != nil {
			server.writeError(writer, format, http.StatusBadGateway, "upstream_unavailable", "selected endpoint request failed")
			return
		}
		if upstream.StatusCode >= 200 && upstream.StatusCode < 300 {
			break
		}
		errorBody, readErr := readBounded(upstream.Body, 1<<20)
		upstream.Body.Close()
		// A runtime's own context refusal arrives before any output. It is a
		// fact for the service, never a local reselection or truncation.
		if v2 && readErr == nil && isRuntimeContextRejection(selected.deployment, upstream.StatusCode, errorBody) {
			fact := runtimeContextRejection(selected.deployment, capacityFact(selected.deployment, prompt, prepared.outputLimit))
			if !server.reconsult(ctx, writer, format, pending, selected.action.ID, decision.RouteID, "capacity", fact) {
				return
			}
			continue
		}
		if readErr != nil {
			server.writeError(writer, format, http.StatusBadGateway, "upstream_error", "selected endpoint returned an unreadable error")
			return
		}
		upstream.Body = io.NopCloser(bytes.NewReader(errorBody))
		server.forwardError(writer, format, selected.deployment.Protocol, upstream, selected.deployment.Kind == "cloud" && isProviderContextRefusal(upstream.StatusCode, errorBody))
		return
	}
	defer upstream.Body.Close()
	if selected.deployment.Kind == "cloud" && !dispatchEchoMatches(upstream.Header, selected, decision) {
		server.writeError(writer, format, http.StatusBadGateway, "dispatch_identity_mismatch", "cloud dispatch did not confirm the configured transport and model identity")
		return
	}
	writer.Header().Set("X-Rayline-Route-Id", decision.RouteID)
	writer.Header().Set("X-Rayline-Action-Id", selection.ActionID)
	writer.Header().Set("X-Rayline-Worker", selection.Worker)
	writer.Header().Set("X-Rayline-Control-Id", selection.ControlID)
	writer.Header().Set("X-Rayline-Selected-Model", selection.Model)
	writer.Header().Set("X-Rayline-Executed-Model", decision.Model)
	writer.Header().Set("X-Rayline-Executed-Action-Id", decision.ActionID)
	writer.Header().Set("X-Rayline-Executed-Control-Id", prepared.receipt.ControlID)
	writer.Header().Set("X-Rayline-Executed-Worker", decision.Worker)
	if selection.Receipt != nil {
		writer.Header().Set("X-Rayline-Decided-By", selection.Receipt.decidedBy())
	}
	if v2 && event.Kind != nil {
		writer.Header().Set("X-Rayline-Event", *event.Kind+"/"+event.Basis)
	}
	if aliasReason != "" {
		writer.Header().Set("X-Rayline-Test-Alias", aliasReason)
	}
	writer.Header().Set("X-Rayline-Instance-Id", server.instance)
	// Every generation turn consults Decisions, local or cloud, so its signals
	// block is the one cloud VSR would carry; routing never goes upstream.
	if value := signalsHeaderValue(pending.decision.Signals); pending.input.Signals != nil && value != "" {
		writer.Header().Set(signalsHeader, value)
	}
	var response llmprotocol.Response
	if request.Stream {
		response, err = server.forwardStream(writer, incoming, upstream, format, selected, prepared.response, pending)
	} else {
		response, err = server.forwardBuffered(writer, upstream, format, selected, prepared.response, pending)
	}
	if err != nil {
		slog.Warn("execution aborted", "route_id", decision.RouteID, "worker", decision.Worker, "output_observed", pending.delivered)
		return
	}
	if llmprotocol.ResponseRefused(&response) {
		// Like cloud VSR, deliver a refusal without advancing private history,
		// attribution, hold, completion clocks or cache evidence. An explicit
		// client retry makes a fresh decision instead of retaining the refusal.
		if !sidecall {
			state.pending = nil
		}
		slog.Info("execution refused", "route_id", decision.RouteID, "worker", decision.Worker)
		return
	}
	if sidecall {
		slog.Info("execution completed", "route_id", decision.RouteID, "selected_action_id", selection.ActionID, "executed_action_id", decision.ActionID, "worker", decision.Worker, "deployment", selected.deployment.ID, "event", EventSidecall, "state_advanced", false)
		return
	}
	// Both wire decoding and downstream delivery have completed. No failed
	// generation or decision consult can enter this commit boundary.
	now := server.now()
	ttl, source := dispatchedCacheTTL(selected.deployment, prepared.body)
	call := cachecontext.CallCache{ActionID: decision.ActionID, ContextEpoch: state.evidence(server.instance).ContextEpoch, TTLSeconds: ttl, TTLSource: source}
	call.RecordCompletion(now, cacheUsage(response.Usage))
	committed := cachecontext.CommitCall(nil, call, true, now)
	currentPrefix := prompt.Prefix(state.workers[selected.deployment.ID].prompt)
	state.placements[prepared.key] = prepared.placer.State()
	state.issuers = prepared.nextIssuers
	state.provenance = reasoning.WithReasoningProvenance(state.provenance, prepared.opaqueIssuer, protocolcodec.ResponseOpaqueReasoning(&response))
	state.workers[selected.deployment.ID] = WorkerEvidence{DeploymentID: selected.deployment.ID, SelectedActionID: selection.ActionID, SelectedModel: selection.Model, ExecutedModel: decision.Model, Call: *committed, prompt: prompt}
	state.held = decision.ActionID
	state.heldWorker = decision.Worker
	state.completed++
	state.recordAttribution(format, roles, decision)
	state.prefix = prefix
	state.wire = wire
	state.carried = nil
	pending.completed = true
	var decidedBy any
	if selection.Receipt != nil {
		decidedBy = selection.Receipt.decidedBy()
	}
	slog.Info("execution completed", "route_id", decision.RouteID, "selected_model", selection.Model, "selected_action_id", selection.ActionID, "selected_control_id", selection.ControlID, "executed_model", decision.Model, "executed_action_id", decision.ActionID, "executed_control_id", prepared.receipt.ControlID, "worker", decision.Worker, "deployment", selected.deployment.ID, "deployment_kind", selected.deployment.Kind, "test_alias_reason", aliasReason, "context_epoch", call.ContextEpoch, "event_kind", event.kindName(), "event_basis", event.Basis, "decided_by", decidedBy, "consults", pending.consults+1, "output_tokens", tokenCountOrNil(response.Usage.OutputTotal), "input_tokens", tokenCountOrNil(response.Usage.InputTotal), "cache", completionCacheEvidence(state, selected.deployment.ID, measuredEvidence, currentPrefix))
}

// maxReconsults bounds re-consults for one request: one post-decision veto and
// one runtime context refusal.
const maxReconsults = 2

// reconsult asks the service again for the same turn with one narrowing fact.
// The core reports the fact and executes the service's new choice; it never
// selects another action itself. It writes the visible error on failure.
func (server *Server) reconsult(ctx context.Context, writer http.ResponseWriter, format llmprotocol.WireFormat, pending *pendingTurn, actionID, retryOf, kind string, fact *CapacityFact) bool {
	if pending.consults >= maxReconsults {
		if server.writeContextExceeded(writer, format, withFact(pending.input.Capacity, actionID, fact)) {
			return false
		}
		server.writeError(writer, format, http.StatusConflict, "execution_no_feasible_target", "the selected target does not fit this request and the re-consult limit was reached")
		return false
	}
	input := pending.input
	capacity := withFact(input.Capacity, actionID, fact)
	input.Capacity = capacity
	// Keep the fact even if this consult fails, so a client retry of the same
	// request cannot dispatch the refused target again.
	pending.input.Capacity = capacity
	input.Reconsult = &Reconsult{Kind: kind, RetryOf: retryOf, Attempt: pending.consults + 2}
	slog.Info("execution reconsult", "reconsult", kind, "retry_of", retryOf, "vetoed_action_id", actionID, "required", int64OrNil(fact.Required), "available", int64OrNil(fact.Available), "attempt", input.Reconsult.Attempt)
	decision, err := server.provider.Decide(ctx, input)
	if err != nil {
		server.writeDecisionError(writer, format, err, input.Capacity)
		return false
	}
	if decision.ActionID == actionID {
		if server.writeContextExceeded(writer, format, capacity) {
			return false
		}
		server.writeError(writer, format, http.StatusConflict, "execution_no_feasible_target", "Decisions reselected a target the core reported does not fit")
		return false
	}
	pending.input, pending.decision = input, decision
	pending.consults++
	return true
}

// withFact copies a turn's capacity facts with one narrowing fact added.
func withFact(facts map[string]*CapacityFact, actionID string, fact *CapacityFact) map[string]*CapacityFact {
	capacity := make(map[string]*CapacityFact, len(facts)+1)
	for id, existing := range facts {
		capacity[id] = existing
	}
	capacity[actionID] = fact
	return capacity
}

// Failed and partial attempts consume the same process-wide admission as a
// successful completion. Session changes or transcript rewrites do not reset it.
func (server *Server) reserveGenerationAttempt() bool {
	limit := server.config.MaxGenerationAttempts
	if limit == 0 {
		return true
	}
	for {
		used := server.generationAttempts.Load()
		if used >= limit {
			return false
		}
		if server.generationAttempts.CompareAndSwap(used, used+1) {
			return true
		}
	}
}

// writeDecisionError reports a failed consult. A service that finds no
// feasible target while every target has a known context shortfall gets the
// same compaction message as the core's own refusal, not a generic failure.
func (server *Server) writeDecisionError(writer http.ResponseWriter, format llmprotocol.WireFormat, err error, capacity map[string]*CapacityFact) {
	var refusal *DecisionsRefusal
	noFeasible := errors.Is(err, errSelectedInfeasible) || errors.As(err, &refusal) && (refusal.Code == "execution_no_feasible_target" || refusal.Code == "no_feasible_target")
	if noFeasible && server.writeContextExceeded(writer, format, capacity) {
		return
	}
	server.writeError(writer, format, http.StatusBadGateway, "decision_failed", err.Error())
}

// writeContextExceeded writes the compaction refusal when every target has a
// context shortfall and none is unknown.
func (server *Server) writeContextExceeded(writer http.ResponseWriter, format llmprotocol.WireFormat, capacity map[string]*CapacityFact) bool {
	if !server.exceedsAllDeployments(capacity) {
		return false
	}
	server.writeError(writer, format, http.StatusBadRequest, CodeContextExceedsAllDeployments, server.contextExceededMessage(capacity))
	return true
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}

func (server *Server) writeError(writer http.ResponseWriter, format llmprotocol.WireFormat, status int, code, message string) {
	body, err := server.engine.EncodeError(format, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, code, message, nil))
	if err != nil {
		http.Error(writer, "execution error", http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_, _ = writer.Write(body)
}

func cacheUsage(usage llmprotocol.Usage) cachecontext.Usage {
	count := func(value llmprotocol.TokenCount) *int {
		if value.Value == nil || *value.Value < 0 || (value.Provenance != llmprotocol.UsageAuthoritative && value.Provenance != llmprotocol.UsageDerived) {
			return nil
		}
		result := int(*value.Value)
		if int64(result) != *value.Value {
			return nil
		}
		return &result
	}
	return cachecontext.Usage{PromptTokens: count(usage.InputTotal), CacheReadTokens: count(usage.InputCacheRead), CacheWriteTokens: count(usage.InputCacheWrite)}
}

// cache_control documents Anthropic's prompt cache, not another provider's
// Messages-compatible wire or a local runtime's KV residency.
func dispatchedCacheTTL(deployment Deployment, body []byte) (int, string) {
	if !isAnthropicDeployment(deployment) {
		return 0, ""
	}
	return cachecontext.WireCacheTTL(body)
}

// tokenCountOrNil logs a provider token count only when it is known.
func tokenCountOrNil(count llmprotocol.TokenCount) any {
	if count.Value == nil {
		return nil
	}
	return *count.Value
}
