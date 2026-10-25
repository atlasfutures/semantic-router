package execution

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
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
	session, branch, err := requestIdentity(incoming.Header, body)
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "session_identity_required", err.Error())
		return
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
	prefix, err := requestPrefix(request)
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	state, err := server.states.acquire(ctx, session, branch)
	if err != nil {
		server.writeError(writer, format, http.StatusServiceUnavailable, "execution_state_unavailable", err.Error())
		return
	}
	defer state.release()
	if state.pending == nil && state.completed == 0 && hasExecutionHistory(request) {
		server.writeError(writer, format, http.StatusConflict, "execution_state_not_found", "this session has assistant or tool history but no execution ledger; start a fresh session")
		return
	}
	if len(state.prefix) > 0 && !extendsPrefix(state.prefix, prefix) {
		state.resetEpoch()
	}
	requestFingerprint := fingerprint(body)
	requestID := incoming.Header.Get("Idempotency-Key")
	if requestID == "" {
		requestID = requestFingerprint
	}
	if state.pending != nil && state.pending.requestID == requestID && state.pending.fingerprint != requestFingerprint {
		server.writeError(writer, format, http.StatusConflict, "retry_body_changed", "an idempotency key cannot name different requests")
		return
	}
	if state.pending != nil && state.pending.fingerprint == requestFingerprint && (state.pending.delivered || state.pending.completed) {
		server.writeError(writer, format, http.StatusConflict, "execution_already_observed", "this request already produced output; reconcile the transcript before retrying")
		return
	}
	var measurements map[string]*cachecontext.MeasuredPrompt
	var measuredEvidence ExecutionContext
	if state.pending == nil || state.pending.fingerprint != requestFingerprint {
		evidence, measured, err := server.measureExecutionContext(ctx, body, format, roles, state)
		if err != nil {
			server.writeError(writer, format, http.StatusBadGateway, "prefix_measurement_failed", err.Error())
			return
		}
		measurements = measured
		measuredEvidence = evidence
		decision, err := server.provider.Decide(ctx, DecisionInput{Body: body, Format: format, SessionID: session, BranchID: branch, RequestID: requestID, State: evidence})
		if err != nil {
			server.writeError(writer, format, http.StatusBadGateway, "decision_failed", err.Error())
			return
		}
		state.pending = &pendingTurn{fingerprint: requestFingerprint, requestID: requestID, decision: decision}
	}
	selection := state.pending.decision
	selected, decision, aliasReason, err := server.executionFor(selection)
	if err != nil {
		server.writeError(writer, format, http.StatusBadGateway, "decision_identity_mismatch", err.Error())
		return
	}
	prepared, err := server.prepare(request, envelope, selected, state)
	if err != nil {
		server.writeError(writer, format, http.StatusBadRequest, "execution_plan_failed", err.Error())
		return
	}
	prompt, err := server.measurePreparedPrompt(ctx, selected, prepared.body, state, measurements)
	if err != nil {
		server.writeError(writer, format, http.StatusBadGateway, "prefix_measurement_failed", err.Error())
		return
	}
	if !server.reserveGenerationAttempt() {
		server.writeError(writer, format, http.StatusTooManyRequests, "generation_limit_reached", "execution process reached its configured generation-attempt limit")
		return
	}
	upstream, err := server.upstream(ctx, selected, decision, session, branch, prepared.body)
	if err != nil {
		server.writeError(writer, format, http.StatusBadGateway, "upstream_unavailable", "selected endpoint request failed")
		return
	}
	defer upstream.Body.Close()
	if upstream.StatusCode < 200 || upstream.StatusCode >= 300 {
		server.forwardError(writer, format, selected.deployment.Protocol, upstream)
		return
	}
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
	if aliasReason != "" {
		writer.Header().Set("X-Rayline-Test-Alias", aliasReason)
	}
	writer.Header().Set("X-Rayline-Instance-Id", server.instance)
	var response llmprotocol.Response
	if request.Stream {
		response, err = server.forwardStream(writer, incoming, upstream, format, selected, prepared.response, state.pending)
	} else {
		response, err = server.forwardBuffered(writer, upstream, format, selected, prepared.response, state.pending)
	}
	if err != nil {
		slog.Warn("execution aborted", "route_id", decision.RouteID, "worker", decision.Worker, "output_observed", state.pending.delivered)
		return
	}
	if llmprotocol.ResponseRefused(&response) {
		// Like cloud VSR, deliver a refusal without advancing private history,
		// attribution, hold, completion clocks or cache evidence. An explicit
		// client retry makes a fresh decision instead of retaining the refusal.
		state.pending = nil
		slog.Info("execution refused", "route_id", decision.RouteID, "worker", decision.Worker)
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
	state.completed++
	state.recordAttribution(format, roles, decision)
	state.prefix = prefix
	state.pending.completed = true
	slog.Info("execution completed", "route_id", decision.RouteID, "selected_model", selection.Model, "selected_action_id", selection.ActionID, "selected_control_id", selection.ControlID, "executed_model", decision.Model, "executed_action_id", decision.ActionID, "executed_control_id", prepared.receipt.ControlID, "worker", decision.Worker, "deployment", selected.deployment.ID, "deployment_kind", selected.deployment.Kind, "test_alias_reason", aliasReason, "context_epoch", call.ContextEpoch, "cache", completionCacheEvidence(state, selected.deployment.ID, measuredEvidence, currentPrefix))
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
