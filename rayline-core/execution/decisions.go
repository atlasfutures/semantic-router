package execution

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

// HTTPDecisions supports the current public API and an explicitly selected
// execution-context contract. An unsupported request never changes API modes.
type HTTPDecisions struct {
	config         Config
	client         *http.Client
	actions        map[string]Action
	deployments    map[string]Deployment
	now            func() time.Time
	exclusive      bool
	valueUSD       *float64
	profileID      string
	profileVersion int64
}

var decisionsCellTag = regexp.MustCompile(`^[a-z][a-z0-9-]{0,45}$`)

func NewHTTPDecisions(config Config) (*HTTPDecisions, error) {
	if config.Decisions.CellTag != "" && !decisionsCellTag.MatchString(config.Decisions.CellTag) {
		return nil, fmt.Errorf("Decisions cell_tag must be a Cloud Run revision tag")
	}
	if err := validateEndpoint(config.Decisions.URL); err != nil {
		return nil, fmt.Errorf("Decisions URL: %w", err)
	}
	if config.Decisions.AuthEnv == "" || os.Getenv(config.Decisions.AuthEnv) == "" {
		return nil, fmt.Errorf("Decisions credential environment is unset")
	}
	mode := config.Decisions.apiMode()
	if mode != DecisionsCurrent && mode != DecisionsExecutionContextV1 {
		return nil, fmt.Errorf("Decisions api_mode must be current or execution_context_v1")
	}
	if config.Decisions.Model == "" || (mode == DecisionsExecutionContextV1 && !validSHA256(config.Decisions.OfferSHA256)) || (config.Decisions.OfferSHA256 != "" && !validSHA256(config.Decisions.OfferSHA256)) {
		return nil, fmt.Errorf("Decisions requires a model and valid configured offer SHA256; execution_context_v1 requires an offer pin")
	}
	switch config.Decisions.Result {
	case "top", "swift", "sage", "custom":
	default:
		return nil, fmt.Errorf("Decisions result must be top, swift, sage or custom")
	}
	var routing map[string]json.RawMessage
	if err := json.Unmarshal(config.Decisions.Routing, &routing); err != nil || routing == nil {
		return nil, fmt.Errorf("Decisions routing must be an explicit JSON object")
	}
	if _, exists := routing["execution"]; exists {
		return nil, fmt.Errorf("execution context is owned by the sidecar, not static routing config")
	}
	if mode == DecisionsCurrent {
		for key := range routing {
			if key != "value_usd" && key != "prices" && key != "exclusive" && key != "price_profile_id" && key != "price_profile_version" {
				return nil, fmt.Errorf("current Decisions routing key %q has no supported applied acknowledgement", key)
			}
		}
	}
	if _, exists := routing["prices"]; exists && !validSHA256(config.Decisions.PriceSHA256) {
		return nil, fmt.Errorf("caller prices require a pinned price_sha256")
	}
	provider := &HTTPDecisions{config: config, client: &http.Client{Timeout: 90 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}, actions: make(map[string]Action), deployments: make(map[string]Deployment), now: time.Now}
	if raw, exists := routing["price_profile_id"]; exists {
		if err := json.Unmarshal(raw, &provider.profileID); err != nil || provider.profileID == "" || strings.TrimSpace(provider.profileID) != provider.profileID || len(provider.profileID) > 128 {
			return nil, fmt.Errorf("routing price_profile_id must be a nonempty profile ID")
		}
		if _, inline := routing["prices"]; inline {
			return nil, fmt.Errorf("routing price_profile_id and prices are mutually exclusive")
		}
	}
	if raw, exists := routing["price_profile_version"]; exists {
		if provider.profileID == "" || json.Unmarshal(raw, &provider.profileVersion) != nil || provider.profileVersion <= 0 || provider.profileVersion > 9007199254740991 {
			return nil, fmt.Errorf("routing price_profile_version requires a profile ID and a positive safe integer")
		}
	}
	if config.Decisions.PriceSHA256 != "" {
		if !validSHA256(config.Decisions.PriceSHA256) {
			return nil, fmt.Errorf("price_sha256 must be a SHA256 digest")
		}
		if _, inline := routing["prices"]; !inline && provider.profileID == "" {
			return nil, fmt.Errorf("price_sha256 requires caller prices or a price profile")
		}
	}
	if raw, exists := routing["value_usd"]; exists {
		var value float64
		if err := json.Unmarshal(raw, &value); err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || string(raw) == "null" {
			return nil, fmt.Errorf("routing value_usd must be a nonnegative finite number")
		}
		if config.Decisions.Result != "custom" {
			return nil, fmt.Errorf("routing value_usd requires the custom Decisions result")
		}
		provider.valueUSD = &value
	}
	if config.Decisions.Result == "custom" && provider.valueUSD == nil {
		return nil, fmt.Errorf("custom Decisions result requires explicit routing value_usd")
	}
	if raw, exists := routing["exclusive"]; exists {
		if err := json.Unmarshal(raw, &provider.exclusive); err != nil {
			return nil, fmt.Errorf("routing exclusive must be boolean")
		}
	}
	for _, action := range config.Actions {
		provider.actions[action.ID] = action
	}
	for _, deployment := range config.Deployments {
		provider.deployments[deployment.ID] = deployment
	}
	return provider, nil
}

type executionTurn struct {
	ID      string `json:"id"`
	Attempt int    `json:"attempt"`
}

type externalContext struct {
	Version      string            `json:"version"`
	ContextEpoch string            `json:"context_epoch"`
	Turn         executionTurn     `json:"turn"`
	Expect       map[string]string `json:"expect"`
	Attribution  []Attribution     `json:"attribution"`
	Hold         any               `json:"hold"`
	Workers      []externalWorker  `json:"workers"`
}

type externalWorker struct {
	ActionID      string          `json:"action_id"`
	Worker        string          `json:"worker"`
	Placement     string          `json:"placement"`
	ContextEpoch  *string         `json:"context_epoch"`
	Gap           *float64        `json:"gap_s"`
	TTL           *externalTTL    `json:"ttl"`
	Residency     *Residency      `json:"residency"`
	Usage         *externalUsage  `json:"usage"`
	CurrentPrefix *externalPrefix `json:"current_prefix"`
}

type externalTTL struct {
	Seconds float64 `json:"seconds"`
	Source  string  `json:"source"`
	Basis   string  `json:"basis"`
}

type externalUsage struct {
	PromptTokens     *int   `json:"prompt_tokens"`
	CacheReadTokens  *int   `json:"cache_read_tokens"`
	CacheWriteTokens *int   `json:"cache_write_tokens"`
	Units            string `json:"units"`
	Provenance       string `json:"provenance"`
}

type externalPrefix struct {
	Tokens        int    `json:"tokens"`
	OverlapTokens *int   `json:"overlap_tokens"`
	Units         string `json:"units"`
	Provenance    string `json:"provenance"`
}

func (provider *HTTPDecisions) project(input DecisionInput) externalContext {
	attribution := append([]Attribution{}, input.State.Attribution...)
	external := externalContext{Version: "arc-execution-context.v1", ContextEpoch: input.State.ContextEpoch, Turn: executionTurn{ID: fingerprint([]byte(input.SessionID + "\x00" + input.BranchID + "\x00" + input.RequestID)), Attempt: 1}, Expect: map[string]string{"package_sha256": provider.config.PackageSHA256, "thinking_controls_sha256": provider.config.RegistrySHA256, "offer_sha256": provider.config.Decisions.OfferSHA256}, Attribution: attribution, Workers: []externalWorker{}}
	if input.State.HeldActionID != "" {
		external.Hold = map[string]string{"scope": "model", "action_id": input.State.HeldActionID}
	}
	// Reuse VSR's completion clock, validity checks, retention and ordering.
	workers := append([]WorkerEvidence(nil), input.State.Workers...)
	sort.Slice(workers, func(i, j int) bool { return workers[i].DeploymentID < workers[j].DeploymentID })
	warmth := make([]*cachecontext.WorkerWarmth, len(workers))
	arms := make(map[string]int)
	byAction := make(map[string]WorkerEvidence)
	for index, evidence := range workers {
		if _, bound := provider.actions[evidence.Call.ActionID]; !bound {
			continue
		}
		call := evidence.Call
		warmth[index] = &cachecontext.WorkerWarmth{Call: &call}
		arms[call.ActionID] = index
		byAction[call.ActionID] = evidence
	}
	cache := cachecontext.BuildPolicyCacheContext(warmth, provider.now(), func(action string) (int, bool) { index, ok := arms[action]; return index, ok })
	for _, entry := range cache.Actions {
		evidence := byAction[entry.ActionID]
		action := provider.actions[entry.ActionID]
		deployment := provider.deployments[evidence.DeploymentID]
		worker := externalWorker{ActionID: entry.ActionID, Worker: action.Worker, Placement: deployment.Kind, ContextEpoch: entry.ContextEpoch, Gap: entry.GapS, Residency: evidence.Residency}
		if entry.TTLS != nil && entry.TTLSource != nil {
			worker.TTL = &externalTTL{Seconds: *entry.TTLS, Source: *entry.TTLSource, Basis: "wire_declared"}
		}
		if deployment.TokenUnits != "" && (entry.PromptTokens != nil || entry.CacheReadTokens != nil || entry.CacheWriteTokens != nil) {
			worker.Usage = &externalUsage{PromptTokens: entry.PromptTokens, CacheReadTokens: entry.CacheReadTokens, CacheWriteTokens: entry.CacheWriteTokens, Units: deployment.TokenUnits, Provenance: "provider_reported"}
		}
		if evidence.PrefixTokens != nil && *evidence.PrefixTokens > 0 && evidence.TokenUnits != "" {
			worker.CurrentPrefix = &externalPrefix{Tokens: *evidence.PrefixTokens, OverlapTokens: evidence.OverlapTokens, Units: evidence.TokenUnits, Provenance: evidence.PrefixProvenance}
		}
		external.Workers = append(external.Workers, worker)
	}
	return external
}

type decisionIdentity struct {
	PackageSHA256  string  `json:"package_sha256"`
	ActionID       string  `json:"action_id"`
	PolicyActionID string  `json:"policy_action_id"`
	ControlID      string  `json:"control_id"`
	ArmID          *string `json:"arm_id"`
	RegistrySHA256 string  `json:"thinking_controls_sha256"`
	OfferSHA256    string  `json:"offer_sha256"`
	Prices         struct {
		Basis     string  `json:"basis"`
		SHA256    *string `json:"price_sha256"`
		Exclusive *bool   `json:"exclusive"`
	} `json:"prices"`
}

type publicDecision struct {
	Model    string            `json:"model"`
	Provider string            `json:"provider"`
	Worker   string            `json:"worker"`
	ActionID string            `json:"policy_action_id"`
	Identity *decisionIdentity `json:"identity"`
}

type notAdmittedEcho struct {
	Field  string `json:"field"`
	Reason string `json:"reason"`
}

type workerEcho struct {
	ActionID     string            `json:"action_id"`
	Worker       string            `json:"worker"`
	Held         *bool             `json:"held"`
	Pricing      string            `json:"pricing"`
	Read         *bool             `json:"read"`
	Reason       *string           `json:"reason"`
	Basis        *string           `json:"basis"`
	PWarm        *float64          `json:"p_warm"`
	PrefixTokens *int              `json:"prefix_tokens"`
	PrefixBasis  *string           `json:"prefix_basis"`
	NotAdmitted  []notAdmittedEcho `json:"not_admitted"`
}

type decisionsResponse struct {
	publicDecision
	RouteID string                    `json:"route_id"`
	Presets map[string]publicDecision `json:"presets"`
	Custom  *publicDecision           `json:"custom"`
	Routing struct {
		ValueUSD *struct {
			Status    string   `json:"status"`
			Requested *float64 `json:"requested"`
		} `json:"value_usd"`
		Execution *struct {
			Status              string            `json:"status"`
			Turn                executionTurn     `json:"turn"`
			Pinned              map[string]string `json:"pinned"`
			UserTurnCache       string            `json:"user_turn_cache"`
			UserTurnCacheSource string            `json:"user_turn_cache_source"`
			EpisodeIdentity     string            `json:"episode_identity"`
			Multiwarm           *bool             `json:"multiwarm"`
			Boundary            string            `json:"boundary"`
			Workers             []workerEcho      `json:"workers"`
		} `json:"execution"`
		Unapplied    []string `json:"unapplied_keys"`
		PriceProfile *struct {
			Status string `json:"status"`
			AppliedPriceProfile
		} `json:"price_profile"`
		Prices *struct {
			Status    string `json:"status"`
			SHA256    string `json:"price_sha256"`
			Exclusive *bool  `json:"exclusive"`
		} `json:"prices"`
	} `json:"routing"`
}

func (provider *HTTPDecisions) Decide(ctx context.Context, input DecisionInput) (Decision, error) {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(input.Body, &body); err != nil {
		return Decision{}, fmt.Errorf("invalid Decisions request body")
	}
	if _, exists := body["routing"]; exists {
		return Decision{}, fmt.Errorf("hybrid routing preferences belong in the execution config")
	}
	var routing map[string]json.RawMessage
	if err := json.Unmarshal(provider.config.Decisions.Routing, &routing); err != nil {
		return Decision{}, err
	}
	var external externalContext
	if provider.config.Decisions.apiMode() == DecisionsExecutionContextV1 {
		external = provider.project(input)
		executionBytes, err := json.Marshal(external)
		if err != nil {
			return Decision{}, err
		}
		routing["execution"] = executionBytes
	}
	routingBytes, err := json.Marshal(routing)
	if err != nil {
		return Decision{}, err
	}
	body["routing"] = routingBytes
	body["model"], err = json.Marshal(provider.config.Decisions.Model)
	if err != nil {
		return Decision{}, err
	}
	requestBytes, err := json.Marshal(body)
	if err != nil {
		return Decision{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, provider.config.Decisions.URL, bytes.NewReader(requestBytes))
	if err != nil {
		return Decision{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+os.Getenv(provider.config.Decisions.AuthEnv))
	request.Header.Set("X-Conversation-Id", fingerprint([]byte(input.SessionID)))
	request.Header.Set("X-Rayline-Branch-Id", fingerprint([]byte(input.BranchID)))
	request.Header.Set("X-Rayline-Wire-Format", string(input.Format))
	if provider.config.Decisions.CellTag != "" {
		request.Header.Set("X-Rayline-Cell", provider.config.Decisions.CellTag)
	}
	response, err := provider.client.Do(request)
	if err != nil {
		return Decision{}, fmt.Errorf("Decisions request failed")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return Decision{}, fmt.Errorf("Decisions request refused (HTTP %d)", response.StatusCode)
	}
	if tag := provider.config.Decisions.CellTag; tag != "" && response.Header.Get("X-Rayline-Cell") != tag {
		return Decision{}, fmt.Errorf("Decisions did not use the configured cell_tag")
	}
	responseBytes, err := readBounded(response.Body, 8<<20)
	if err != nil {
		return Decision{}, err
	}
	var result decisionsResponse
	if err := json.Unmarshal(responseBytes, &result); err != nil {
		return Decision{}, fmt.Errorf("invalid Decisions response")
	}
	selected := result.publicDecision
	switch provider.config.Decisions.Result {
	case "swift", "sage":
		var exists bool
		selected, exists = result.Presets[provider.config.Decisions.Result]
		if !exists {
			return Decision{}, fmt.Errorf("Decisions returned no requested preset")
		}
	case "custom":
		if result.Custom == nil {
			return Decision{}, fmt.Errorf("Decisions returned no custom result")
		}
		selected = *result.Custom
	}
	if len(result.Routing.Unapplied) > 0 {
		return Decision{}, fmt.Errorf("Decisions did not apply requested routing keys: %s", strings.Join(result.Routing.Unapplied, ", "))
	}
	if provider.valueUSD != nil {
		value := result.Routing.ValueUSD
		if value == nil || value.Status != "applied" || value.Requested == nil || *value.Requested != *provider.valueUSD {
			return Decision{}, fmt.Errorf("Decisions did not apply the requested value_usd")
		}
	}
	if selected.Identity == nil && provider.config.Decisions.apiMode() == DecisionsCurrent {
		return provider.currentTopDecision(result, selected, routing)
	}
	identity := selected.Identity
	if identity == nil || identity.PackageSHA256 != provider.config.PackageSHA256 || identity.RegistrySHA256 != provider.config.RegistrySHA256 || (provider.config.Decisions.OfferSHA256 != "" && identity.OfferSHA256 != provider.config.Decisions.OfferSHA256) || identity.ActionID != identity.PolicyActionID || (selected.ActionID != "" && selected.ActionID != identity.ActionID) {
		return Decision{}, fmt.Errorf("Decisions identity does not match the pinned execution package, registry and offer")
	}
	action, ok := provider.actions[identity.ActionID]
	if !ok || identity.ControlID != action.ControlID || selected.Worker != action.Worker || !provider.matchesActionModel(selected.Model, action) || result.RouteID == "" {
		return Decision{}, fmt.Errorf("Decisions selected an unbound action or control")
	}
	if provider.config.Decisions.apiMode() == DecisionsExecutionContextV1 {
		echo := result.Routing.Execution
		if echo == nil || echo.Status != "applied" || echo.Turn != external.Turn {
			return Decision{}, fmt.Errorf("Decisions did not apply this execution context")
		}
		for _, name := range []string{"package_sha256", "thinking_controls_sha256", "offer_sha256"} {
			if echo.Pinned[name] != "matched" {
				return Decision{}, fmt.Errorf("Decisions did not confirm pinned %s", name)
			}
		}
		if err := validateCachePolicyEcho(echo.UserTurnCacheSource, echo.UserTurnCache, echo.EpisodeIdentity, echo.Multiwarm); err != nil {
			return Decision{}, err
		}
		if err := validateWorkerEcho(external.Workers, echo.Workers, echo.Boundary, echo.UserTurnCache); err != nil {
			return Decision{}, err
		}
		slog.Info("execution context applied", "route_id", result.RouteID, "boundary", echo.Boundary, "cache_policy", echo.UserTurnCache, "cache_policy_source", echo.UserTurnCacheSource, "episode_identity", echo.EpisodeIdentity, "worker_evidence", len(echo.Workers))
	}

	var appliedProfile *AppliedPriceProfile
	if provider.profileID != "" {
		profile := result.Routing.PriceProfile
		if profile == nil || profile.Status != "applied" || profile.ID != provider.profileID || profile.Version <= 0 || profile.Version > 9007199254740991 || (provider.profileVersion != 0 && profile.Version != provider.profileVersion) || !strings.HasPrefix(profile.ContentHash, "sha256:") || !validSHA256(strings.TrimPrefix(profile.ContentHash, "sha256:")) {
			return Decision{}, fmt.Errorf("Decisions did not apply the requested price profile")
		}
		appliedProfile = &profile.AppliedPriceProfile
	}
	if provider.config.Decisions.PriceSHA256 != "" || appliedProfile != nil {
		prices := result.Routing.Prices
		if prices == nil || prices.Status != "applied" || !validSHA256(prices.SHA256) || (provider.config.Decisions.PriceSHA256 != "" && prices.SHA256 != provider.config.Decisions.PriceSHA256) || identity.Prices.Basis != "caller" || identity.Prices.SHA256 == nil || *identity.Prices.SHA256 != prices.SHA256 || identity.Prices.Exclusive == nil || prices.Exclusive == nil || *identity.Prices.Exclusive != *prices.Exclusive || *prices.Exclusive != provider.exclusive {
			return Decision{}, fmt.Errorf("Decisions did not apply the requested caller price world")
		}
	} else if identity.Prices.Basis != "catalog" || identity.Prices.SHA256 != nil || identity.Prices.Exclusive == nil || *identity.Prices.Exclusive != provider.exclusive {
		return Decision{}, fmt.Errorf("Decisions did not confirm the requested catalog price world")
	}
	decision := Decision{RouteID: result.RouteID, ActionID: identity.ActionID, Worker: selected.Worker, Model: action.Model, ReportedModel: selected.Model, PackageSHA256: identity.PackageSHA256, ControlID: identity.ControlID, ArmID: identity.ArmID, Result: provider.config.Decisions.Result, IdentitySource: "response", AppliedValueUSD: provider.valueUSD}
	decision.PriceProfile = appliedProfile
	provider.logDecision(decision, identity.Prices.Basis, identity.Prices.SHA256)
	return decision, nil
}

// The deployed top-only API names an action but does not attest package,
// registry, offer or prices. It can be used explicitly with empty routing;
// the local manifest supplies control identity, clearly labeled in evidence.
func (provider *HTTPDecisions) currentTopDecision(result decisionsResponse, selected publicDecision, routing map[string]json.RawMessage) (Decision, error) {
	if provider.config.Decisions.Result != "top" || len(routing) != 0 || provider.config.Decisions.OfferSHA256 != "" {
		return Decision{}, fmt.Errorf("this Decisions response only supports top with empty routing and no returned identity pins")
	}
	action, ok := provider.actions[selected.ActionID]
	reason := ""
	switch {
	case !ok:
		reason = "unbound_action"
	case result.RouteID == "":
		reason = "missing_route_id"
	case selected.Worker != action.Worker:
		reason = "worker_mismatch"
	case !provider.matchesActionModel(selected.Model, action):
		reason = "model_mismatch"
	}
	if reason != "" {
		provider.logSelectionRefusal(result.RouteID, selected, action, reason)
		return Decision{}, fmt.Errorf("Decisions selected an action, worker or model absent from the configured manifest")
	}
	decision := Decision{RouteID: result.RouteID, ActionID: action.ID, Worker: action.Worker, Model: action.Model, ReportedModel: selected.Model, PackageSHA256: provider.config.PackageSHA256, ControlID: action.ControlID, Result: "top", IdentitySource: "configured_manifest"}
	provider.logDecision(decision, "unattested", nil)
	return decision, nil
}

// Only selected identity metadata is logged. Provider request/response bodies,
// prompts, credentials and transcript/session material never enter this event.
func (provider *HTTPDecisions) logSelectionRefusal(routeID string, selected publicDecision, bound Action, reason string) {
	var controlID any
	if selected.Identity != nil {
		controlID = selected.Identity.ControlID
	}
	slog.Warn("decision refused", "route_id", routeID, "selected_model", selected.Model, "selected_worker", selected.Worker, "selected_action_id", selected.ActionID, "selected_control_id", controlID, "reason", reason, "bound_model", bound.Model, "bound_worker", bound.Worker, "bound_control_id", bound.ControlID, "api_mode", provider.config.Decisions.apiMode())
}

func (provider *HTTPDecisions) matchesActionModel(model string, action Action) bool {
	return model != "" && (model == action.Model || model == action.ReportedModel || model == provider.deployments[action.DeploymentID].RegistryModel)
}

func (provider *HTTPDecisions) logDecision(decision Decision, priceBasis string, priceSHA *string) {
	slog.Info("decision selected", "route_id", decision.RouteID, "selected_model", decision.Model, "reported_model", decision.ReportedModel, "action_id", decision.ActionID, "control_id", decision.ControlID, "package_sha256", decision.PackageSHA256, "identity_source", decision.IdentitySource, "result", decision.Result, "applied_value_usd", decision.AppliedValueUSD, "price_basis", priceBasis, "price_sha256", priceSHA, "price_profile", decision.PriceProfile, "api_mode", provider.config.Decisions.apiMode())
}

func validateWorkerEcho(workers []externalWorker, echoed []workerEcho, boundary, cachePolicy string) error {
	if boundary == "" || len(workers) != len(echoed) {
		return fmt.Errorf("Decisions omitted a boundary or supplied worker evidence")
	}
	for index, worker := range echoed {
		evidence := workers[index]
		reason, basis, prefixBasis := "", "", ""
		if worker.Reason != nil {
			reason = *worker.Reason
		}
		if worker.Basis != nil {
			basis = *worker.Basis
		}
		if worker.PrefixBasis != nil {
			prefixBasis = *worker.PrefixBasis
		}

		if evidence.ActionID != worker.ActionID || evidence.Worker != worker.Worker || worker.Read == nil {
			return fmt.Errorf("Decisions returned an invalid worker evidence acknowledgement")
		}
		if worker.NotAdmitted == nil {
			return fmt.Errorf("Decisions omitted worker evidence admission results")
		}
		admission, prefixAdmitted, residencyAdmitted := expectedAdmission(evidence)
		for _, rejected := range worker.NotAdmitted {
			if reason, exists := admission[rejected.Field]; !exists || reason != rejected.Reason {
				return fmt.Errorf("Decisions returned an unexpected worker evidence rejection")
			}
			delete(admission, rejected.Field)
		}
		if len(admission) != 0 {
			return fmt.Errorf("Decisions omitted rejected worker evidence")
		}
		if boundary == "user_message" {
			if (prefixAdmitted || residencyAdmitted) && cachePolicy != "repartition_multiwarm" {
				return fmt.Errorf("Decisions did not apply admitted executor cache evidence")
			}
			if reason == "boundary" {
				return fmt.Errorf("Decisions treated a user-message cache decision as unpriced")
			}
			if prefixAdmitted && *worker.Read && (prefixBasis != "worker_count" || worker.PrefixTokens == nil || *worker.PrefixTokens != evidence.CurrentPrefix.Tokens) {
				return fmt.Errorf("Decisions did not price the admitted worker prefix count")
			}
			if prefixAdmitted && reason == "package_mode" {
				return fmt.Errorf("Decisions silently dropped the admitted worker prefix count")
			}
		} else if worker.Pricing != "not_priced" || reason != "boundary" {
			return fmt.Errorf("Decisions applied cache pricing outside a user-message boundary")
		}

		if *worker.Read {
			if worker.Reason != nil {
				return fmt.Errorf("Decisions gave an unread reason for read worker evidence")
			}
		} else if reason != "boundary" && reason != "not_in_offer" && reason != "package_mode" && reason != "superseded" {
			return fmt.Errorf("Decisions did not explain unread worker evidence")
		}
		if worker.Pricing == "not_priced" {
			if *worker.Read || worker.Held != nil || worker.Basis != nil || worker.PWarm != nil || worker.PrefixTokens != nil || worker.PrefixBasis != nil ||
				(reason != "boundary" && reason != "not_in_offer") {
				return fmt.Errorf("Decisions returned cache prices for an unpriced worker")
			}
		} else {
			if worker.Held == nil || basis == "" || worker.PWarm == nil || math.IsNaN(*worker.PWarm) || math.IsInf(*worker.PWarm, 0) || *worker.PWarm < 0 || *worker.PWarm > 1 ||
				worker.PrefixTokens == nil || *worker.PrefixTokens < 0 || (prefixBasis != "worker_count" && prefixBasis != "encoder_fallback") {
				return fmt.Errorf("Decisions returned incomplete worker cache pricing")
			}
			switch worker.Pricing {
			case "observed":
				if !*worker.Read || !strings.HasPrefix(basis, "observed_") {
					return fmt.Errorf("Decisions claimed observed pricing without read worker evidence")
				}
			case "default_held_warm":
				if !*worker.Held {
					return fmt.Errorf("Decisions applied a held-worker default to a rival")
				}
			case "default_rival_cold":
				if *worker.Held {
					return fmt.Errorf("Decisions applied a rival default to the held worker")
				}
			default:
				return fmt.Errorf("Decisions returned an unknown worker pricing mode")
			}
		}

	}
	return nil
}

// Admission is an API acknowledgement check, never a local cache-price decision.
func expectedAdmission(worker externalWorker) (map[string]string, bool, bool) {
	rejected := make(map[string]string)
	if worker.TTL != nil && worker.TTL.Basis == "caller_estimated" {
		rejected["ttl"] = "estimated_not_admitted"
	}
	residency := worker.Residency != nil
	if residency && worker.Residency.Provenance != "local_measured" {
		rejected["residency"] = "residency_requires_local_measured"
		residency = false
	} else if residency && worker.Placement != "local" {
		rejected["residency"] = "residency_requires_local_placement"
		residency = false
	}
	prefix := worker.CurrentPrefix != nil
	if prefix && worker.CurrentPrefix.Provenance == "caller_estimated" {
		rejected["current_prefix"] = "estimated_not_admitted"
		prefix = false
	}
	if worker.Usage != nil {
		if worker.Usage.Provenance == "caller_estimated" {
			rejected["usage"] = "estimated_not_admitted"
		} else if prefix && worker.Usage.Units != worker.CurrentPrefix.Units {
			rejected["usage"] = "units_mismatch"
		}
	}
	return rejected, prefix, residency
}

func validateCachePolicyEcho(source, mode, episodeIdentity string, multiwarm *bool) error {
	switch episodeIdentity {
	case "account_scoped", "per_lookup_random":
	default:
		return fmt.Errorf("Decisions did not identify its trusted episode isolation mode")
	}
	switch source {
	case "package", "request", "service_config":
	default:
		return fmt.Errorf("Decisions did not identify the applied cache policy source")
	}
	if mode == "" || multiwarm == nil || *multiwarm != (mode == "repartition_multiwarm") {
		return fmt.Errorf("Decisions returned an inconsistent multiwarm acknowledgement")
	}
	return nil
}
