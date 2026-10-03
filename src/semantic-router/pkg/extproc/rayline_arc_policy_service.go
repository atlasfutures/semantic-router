/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

// The Rayline ARC policy-service mode: an external service runs the encoder,
// head and selection rule and returns the decision; this router binds the
// chosen action to a worker (website/docs/proposals/rayline-arc-policy-service.md).

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	policyFormatAnthropic = "anthropic_messages"
	policyFormatOpenAI    = "openai_chat"
	policyFormatResponses = "openai_responses"

	arcFailurePolicyRequestFormat = "policy_request_format"
)

// policySessionActions is the closed set the encoder session contract names;
// the service's value becomes a metric label, so any other is dropped.
var policySessionActions = map[string]bool{"created": true, "rebuilt": true, "appended": true, "reused": true}

type policyBinding struct {
	arm   int
	level string
	// model is the action's trained model, when the binding declares it.
	model string
}

// policyServiceScorer carries the worker catalog and action bindings. It does
// not score embeddings: the policy service decides, and Select branches to it
// before any encode.
type policyServiceScorer struct {
	schedule string
	// busyWait bounds how long a decision retries the service's
	// session_busy (the episode's acquire timeout).
	busyWait    time.Duration
	alias       string
	sha256      string
	workerIDs   []string
	workers     []raylinearc.WorkerManifest
	bindings    map[string]policyBinding
	actionOrder []string
	// fallback is ADR 0120's fallback: excluded models leave the offer, and
	// an excluded held model forces a decision.
	fallback bool
	// routes names each worker's route in the cell exclusion table.
	routes []string
	// episodeMode is the decide request's episode_mode: empty (strict) or
	// relaxed.
	episodeMode string
	// sideStrictUntil (unix nanoseconds) is how long a strict cell sends its
	// side calls strict after the service last showed it cannot serve them
	// relaxed; zero or past means relaxed.
	sideStrictUntil atomic.Int64
}

// policySideCallStrictFor is how long a strict cell keeps its side calls
// strict once the service has shown it cannot serve them relaxed. The service
// may be redeployed with relaxed support, so the router asks again after it.
const policySideCallStrictFor = 10 * time.Minute

// sideCallEpisodeMode is the episode_mode a side call is sent with. A side
// call commits nothing and holds no lease, so it is decided relaxed: the
// service takes no session lock and leaves the main conversation's session
// untouched. Only a strict cell whose service recently could not serve
// relaxed sends it strict, as before.
func (scorer *policyServiceScorer) sideCallEpisodeMode(now time.Time) string {
	if scorer.episodeMode == raylinearc.PolicyEpisodeModeRelaxed ||
		now.UnixNano() >= scorer.sideStrictUntil.Load() {
		return raylinearc.PolicyEpisodeModeRelaxed
	}
	return scorer.episodeMode
}

// sideCallsRelaxedUnsupported records that the service served a relaxed side
// call strict (it predates episode_mode) or refused it (its package cannot
// serve relaxed), so side calls go strict for a while.
func (scorer *policyServiceScorer) sideCallsRelaxedUnsupported(now time.Time, how string) {
	scorer.sideStrictUntil.Store(now.Add(policySideCallStrictFor).UnixNano())
	logging.ComponentWarnEvent("extproc", "rayline_arc_policy_side_call_strict", map[string]interface{}{
		"package_alias": scorer.alias,
		"reason":        how,
		"retry_seconds": policySideCallStrictFor.Seconds(),
	})
}

func (scorer *policyServiceScorer) WorkerIDs() []string { return scorer.workerIDs }
func (scorer *policyServiceScorer) ArtifactID() string  { return scorer.alias }
func (scorer *policyServiceScorer) EncoderRevision() string {
	return "policy-service"
}

func (scorer *policyServiceScorer) Select(
	[]float32, []bool, *raylinearc.EpisodeState, int, time.Time,
) (raylinearc.Decision, error) {
	return raylinearc.Decision{}, errors.New("the policy service decides; there is no embedding to score")
}

func (scorer *policyServiceScorer) Worker(index int) (raylinearc.WorkerManifest, bool) {
	if index < 0 || index >= len(scorer.workers) {
		return raylinearc.WorkerManifest{}, false
	}
	return scorer.workers[index], true
}

// raylineARCScorerGeneration numbers the scorers this process builds. A
// route's exclusion is scoped to the scorer that named it, so a config reload,
// which builds new scorers, starts with no route excluded, whatever it
// changed about the routes (credentials, a base path) that the route's name
// cannot see.
var raylineARCScorerGeneration atomic.Uint64

func newPolicyServiceScorer(
	cfg *config.RouterConfig,
	decision *config.Decision,
) *policyServiceScorer {
	policy := decision.Algorithm.RaylineARC.PolicyService
	generation := strconv.FormatUint(raylineARCScorerGeneration.Add(1), 10)
	scorer := &policyServiceScorer{
		fallback: policy.FallbackEnabled(),
		schedule: policy.ModelSchedule,
		busyWait: time.Duration(decision.Algorithm.RaylineARC.Episode.AcquireTimeoutSeconds) * time.Second,
		alias:    policy.PackageAlias,
		sha256:   policy.PackageSHA256,
		bindings: make(map[string]policyBinding, len(policy.Bindings)),
	}
	if decision.Algorithm.RaylineARC.Episode.RelaxedConsistency() {
		scorer.episodeMode = raylinearc.PolicyEpisodeModeRelaxed
	}
	index := make(map[string]int, len(decision.ModelRefs))
	for arm, modelRef := range decision.ModelRefs {
		index[modelRef.Model] = arm
		scorer.workerIDs = append(scorer.workerIDs, modelRef.Model)
		scorer.workers = append(scorer.workers, policyWorkerManifest(cfg, modelRef))
		route := policyWorkerRoute(cfg, modelRef)
		if route != "" {
			route = generation + "\x00" + route
		}
		scorer.routes = append(scorer.routes, route)
	}
	for _, binding := range policy.Bindings {
		level, model := binding.Level, binding.Model
		// A v5 binding is only its action and worker; the package names the
		// model and the control's level.
		if action, ok := policy.PackageV5Action(binding.ActionID); ok {
			level, model = "", action.Model
			if action.Control.Instruction != nil {
				level = action.Control.Instruction.Level
			}
		}
		scorer.bindings[binding.ActionID] = policyBinding{arm: index[binding.Worker], level: level, model: model}
		scorer.actionOrder = append(scorer.actionOrder, binding.ActionID)
	}
	return scorer
}

// policyWorkerManifest describes a worker from its model card and modelRef,
// the facts the artifact manifest would otherwise carry: rates from the
// card's pricing (zero when absent, as the v3 manifest would for an unpriced
// arm), the provider pin from its provider_preferences, the thinking mode
// from use_reasoning, and which dispatch backend its provider is, so
// OpenRouter-only accounting stays OpenRouter-only.
// policyWorkerRoute names the route a worker dispatches to: the worker, the
// backend endpoint it resolves to, the provider model there and the
// OpenRouter provider preferences sent with it. A config reload that changes
// any of them is another route. A worker served by several endpoints has no
// route here: Envoy balances its calls across them, so a failure cannot be
// pinned on one, and excluding the whole worker would also take its healthy
// endpoints out of the offer.
func policyWorkerRoute(cfg *config.RouterConfig, modelRef config.ModelRef) string {
	if len(cfg.GetEndpointsForModel(modelRef.Model)) > 1 {
		return ""
	}
	address, endpoint, _, _ := cfg.ResolvePrimaryBackendForModel(modelRef.Model)
	preferences, _ := json.Marshal(cfg.ProviderPreferencesForModel(modelRef.Model))
	return strings.Join([]string{
		modelRef.Model, endpoint, address, cfg.ResolveExternalModelID(modelRef.Model, endpoint), string(preferences),
	}, "\x00")
}

func policyWorkerManifest(cfg *config.RouterConfig, modelRef config.ModelRef) raylinearc.WorkerManifest {
	model := modelRef.Model
	// The route dispatch takes: the primary backend, not the first listed.
	_, endpoint, _, _ := cfg.ResolvePrimaryBackendForModel(model)
	worker := raylinearc.WorkerManifest{
		ID: model, Model: cfg.ResolveExternalModelID(model, endpoint),
		ThinkingMode: "off", DispatchBackend: raylinearc.DispatchOpenAICompat,
	}
	if modelRef.UseReasoning != nil && *modelRef.UseReasoning {
		worker.ThinkingMode = "on"
	}
	if profile, err := cfg.GetProviderProfileForEndpoint(endpoint); err == nil && providerIsOpenRouter(profile) {
		worker.DispatchBackend = raylinearc.DispatchOpenRouter
	}
	if pin := cfg.ProviderPreferencesForModel(model); pin != nil {
		worker.OpenRouterProviderOrder = append([]string(nil), pin.Order...)
		// A provider is the worker's only when OpenRouter may use no other:
		// with fallbacks allowed, the first preference need not serve.
		if len(pin.Order) > 0 && worker.DispatchBackend == raylinearc.DispatchOpenRouter &&
			pin.AllowFallbacks != nil && !*pin.AllowFallbacks {
			worker.OpenRouterProviderSlug = pin.Order[0]
		}
		worker.OpenRouterAllowFallbacks = pin.AllowFallbacks == nil || *pin.AllowFallbacks
	}
	if params, ok := cfg.ModelConfig[model]; ok {
		pricing := params.Pricing
		worker.EstimatedInputCostPerToken = pricing.PromptPer1M / tokensPerMillion
		worker.EstimatedOutputCostPerToken = pricing.CompletionPer1M / tokensPerMillion
		worker.EstimatedCacheReadCostPerToken = pricing.CachedInputPer1M / tokensPerMillion
		if pricing.CacheWritePer1M != nil {
			worker.EstimatedCacheWriteCostPerToken = *pricing.CacheWritePer1M / tokensPerMillion
		}
	}
	return worker
}

func policyRequestFormat(format llmprotocol.WireFormat) string {
	switch format {
	case llmprotocol.AnthropicMessagesV1:
		return policyFormatAnthropic
	case llmprotocol.OpenAIChatV1:
		return policyFormatOpenAI
	case llmprotocol.OpenAIResponsesV1:
		return policyFormatResponses
	default:
		return ""
	}
}

// createRaylineARCPolicySelector builds the selector for the policy-service
// mode and arms it in the background once the service reports the configured
// package loaded.
func createRaylineARCPolicySelector(
	cfg *config.RouterConfig,
	decision *config.Decision,
	recipe config.RecipeName,
) (
	selection.Selector,
	raylinearc.EpisodeStore,
	func() error,
	raylineARCSessionCloseFunc,
	string,
) {
	arcConfig := decision.Algorithm.RaylineARC
	policy := arcConfig.PolicyService
	unavailable := func(class string) (
		selection.Selector, raylinearc.EpisodeStore, func() error, raylineARCSessionCloseFunc, string,
	) {
		return newUnrecoverableRaylineARCSelector(policy.PackageSHA256), nil, nil, nil, class
	}
	if !raylineARCPolicyDispatchReady(cfg, decision) || !raylineARCPolicyActionsCarriable(cfg, decision) {
		return unavailable("dispatch_contract")
	}
	modalKey, keyErr := raylineARCOptionalSecret(policy.ModalKeyEnv)
	modalSecret, secretErr := raylineARCOptionalSecret(policy.ModalSecretEnv)
	if keyErr != nil || secretErr != nil {
		return unavailable("policy_service_auth")
	}
	client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
		BaseURL:        policy.BaseURL,
		ModalKey:       modalKey,
		ModalSecret:    modalSecret,
		TotalTimeout:   time.Duration(policy.TotalTimeoutSeconds) * time.Second,
		ConnectTimeout: time.Duration(policy.ConnectTimeoutSeconds) * time.Second,
	})
	episodeStore, closeStore, err := createRaylineARCEpisodeStore(arcConfig.Episode)
	if err != nil {
		return unavailable("episode_store")
	}
	probeContext, cancelProbe := context.WithCancel(context.Background())
	closeResources := func() error {
		cancelProbe()
		if closeStore != nil {
			return closeStore()
		}
		return nil
	}
	selector := newRaylineARCSelector(nil, nil, nil, policy.PackageSHA256)
	selector.recipe = recipe
	armed := &raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(cfg, decision),
		admission: raylinearc.NewAdmissionGate(policy.MaxInflightCalls),
		policy:    client,
	}
	scorer := armed.scorer.(*policyServiceScorer)
	probe := func(ctx context.Context) error {
		if err := client.RequirePackage(ctx, policy.PackageAlias, policy.PackageSHA256); err != nil {
			return err
		}
		if scorer.episodeMode == raylinearc.PolicyEpisodeModeRelaxed {
			return probeRelaxedPolicyDecide(ctx, client, scorer)
		}
		return nil
	}
	raylineARCArmInBackground(
		probeContext,
		selector,
		armed,
		probe,
		raylineARCProbeBackoffFromConfig(config.RaylineARCEncoderConfig{}),
		raylineARCWait,
	)
	return selector, episodeStore, closeResources, nil, raylineARCReadinessPendingClass
}

// policySessionRevision is the trace's session revision: a relaxed call has
// none, and reports 0 as a first call would.
func policySessionRevision(revision *int) int {
	if revision == nil {
		return 0
	}
	return *revision
}

// relaxedProbeEpisodeIDHash names the readiness probe's episode. A relaxed
// call touches no session state, so it collides with nothing.
var relaxedProbeEpisodeIDHash = strings.Repeat("0", 64)

// probeRelaxedPolicyDecide asks for one relaxed decision before a relaxed
// cell arms, because only a decide call shows whether the service can serve
// relaxed: a package on the vLLM encoder, or pinned to a runtime without
// unretained prediction, answers unsupported_request. Until it can, the cell
// stays not ready rather than failing every turn. A relaxed call holds no
// session, so the probe leaves the service's state as it found it.
func probeRelaxedPolicyDecide(
	ctx context.Context,
	client *raylinearc.PolicyServiceClient,
	scorer *policyServiceScorer,
) error {
	pkg := raylinearc.PolicyPackageRef{Alias: scorer.alias, PackageSHA256: scorer.sha256}
	response, err := client.Decide(ctx, raylinearc.PolicyDecisionRequest{
		SchemaVersion: raylinearc.PolicyDecisionRequestSchema,
		Package:       pkg,
		EpisodeIDHash: relaxedProbeEpisodeIDHash,
		ContextEpoch:  "0",
		RequestFormat: policyFormatAnthropic,
		Request: raylinearc.PolicyClientRequest{
			Messages: json.RawMessage(`[{"role":"user","content":"readiness probe"}]`),
		},
		Attribution: []raylinearc.PolicyAttribution{},
		Selection: raylinearc.PolicySelection{
			AvailableActionIDs:  append([]string(nil), scorer.actionOrder...),
			OperatingPoint:      raylinearc.PolicyOperatingPoint{Name: "default"},
			PreferenceDimension: "overall",
		},
		Shadow:      []raylinearc.PolicyPackageRef{},
		EpisodeMode: raylinearc.PolicyEpisodeModeRelaxed,
	})
	var failure *raylinearc.PolicyServiceError
	if errors.As(err, &failure) && failure.Class == raylinearc.PolicyRelaxedUnsupportedClass {
		logRelaxedPolicyUnsupported(scorer, failure.Class)
	}
	if err != nil {
		return err
	}
	// A service that predates episode_mode may accept the field and serve the
	// call strict. Only an answer for this package with no session revision
	// shows the call ran unretained.
	if response.Package != pkg || response.Encoding.SessionRevision != nil {
		logRelaxedPolicyUnsupported(scorer, errRelaxedPolicyIgnored.Class)
		return errRelaxedPolicyIgnored
	}
	return nil
}

// errRelaxedPolicyIgnored is a relaxed probe the service answered as strict.
var errRelaxedPolicyIgnored = &raylinearc.PolicyServiceError{Class: "relaxed_ignored"}

func logRelaxedPolicyUnsupported(scorer *policyServiceScorer, class string) {
	logging.ComponentErrorEvent("extproc", "rayline_arc_policy_relaxed_unsupported", map[string]interface{}{
		"package_alias": scorer.alias,
		"class":         class,
	})
}

type policyClientRequest struct {
	System   json.RawMessage `json:"system"`
	Tools    json.RawMessage `json:"tools"`
	Messages json.RawMessage `json:"messages"`
}

// selectViaPolicyService asks the policy service for the decision and maps
// the chosen action onto the worker pool.
func (selector *raylineARCSelector) selectViaPolicyService(
	ctx context.Context,
	armed *raylineARCArmedComponents,
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	workerIDs []string,
	state *raylinearc.EpisodeState,
	excluded []bool,
	hard []bool,
) (*selection.SelectionResult, error) {
	scorer, ok := armed.scorer.(*policyServiceScorer)
	if !ok {
		return nil, arcSelectionFailure("policy_scorer")
	}
	if arcContext.RequestFormat == "" {
		return nil, arcSelectionFailure(arcFailurePolicyRequestFormat)
	}
	clientRequest, messages, roles, err := policyClientRequestOf(arcContext)
	if err != nil {
		return nil, arcSelectionFailure("policy_request_body")
	}
	signals := raylineARCPolicyTurnSignals(
		arcContext.PolicySignalHeaders, arcContext.RequestFormat, clientRequest, messages, arcContext.EpisodeIDHash,
	)
	// A request classified as a side call ahead of the episode read holds no
	// lease, so it is served as one. An unknown pre-classification holds the
	// lease and stays open to a positive reading of the materialized request.
	if arcContext.PolicyCallKind != "" && arcContext.PolicyCallKind != raylinearc.PolicyCallUnknown {
		signals.CallKind, signals.CallKindSource = arcContext.PolicyCallKind, arcContext.PolicyCallKindSource
	}
	turn, attribution, transition := raylinearc.PolicyTurn(state.Policy, messages, roles, state.TurnIndex, signals)
	sideCall := signals.CallKind == raylinearc.PolicyCallSide
	scheduled := scorer.schedule != "" && !sideCall
	atBoundary := scheduled &&
		raylinearc.ModelChangeAllowed(state.TurnIndex, turn.EpochStartTurn, turn.CompactionCount)
	retainedArm, retained := -1, false
	if atBoundary {
		retainedArm, retained = state.PolicyBoundary.RetainedArm(messages, state.TurnIndex, turn)
	}
	held := -1
	// A side call keeps the held arm whatever the schedule says; a turn holds
	// it between the schedule's boundaries; a retry at a boundary keeps the
	// arm that boundary already decided.
	switch {
	case retained:
		held = retainedArm
	case sideCall && state.PolicyBoundary != nil && state.PolicyBoundary.TurnIndex == state.TurnIndex:
		// A main turn decided this boundary and has not committed yet: its
		// arm is the one in use.
		held = state.PolicyBoundary.Arm
	case state.PreviousArm != nil && (sideCall || scheduled && !atBoundary):
		held = *state.PreviousArm
	}
	// ADR 0120: a model this context excluded, or a route the cell excluded,
	// cannot be held. Its turn decides again among the rest, as at a schedule
	// boundary, and that decision is held from here as a boundary decision is.
	var cellOut map[int]string
	if scorer.fallback && !sideCall && turn != nil {
		cellOut = scorer.cellExclusions(selector.now())
	}
	fallback := scorer.fallback && !sideCall && turn != nil && (len(turn.Exclusions) > 0 || len(cellOut) > 0)
	offer := func(cellOut map[int]string) ([]string, int, bool) {
		held, forced := held, false
		out := func(arm int, model string) bool {
			_, routeOut := cellOut[arm]
			return fallback && (turn.Excludes(model) || routeOut)
		}
		if held >= 0 && out(held, scorer.armModel(held)) {
			held, forced = -1, true
		}
		available := make([]string, 0, len(scorer.actionOrder))
		for _, actionID := range scorer.actionOrder {
			arm := scorer.bindings[actionID].arm
			if len(excluded) == len(workerIDs) && excluded[arm] {
				continue
			}
			if held >= 0 && arm != held {
				continue
			}
			if out(arm, scorer.actionModel(actionID)) {
				continue
			}
			available = append(available, actionID)
		}
		return available, held, forced
	}
	// loggedRoutes keeps the route exclusions in force for the log, even when
	// the offer sets them aside.
	loggedRoutes := cellOut
	available, offeredHeld, forced := offer(cellOut)
	if len(available) == 0 && !slices.Equal(excluded, hard) {
		// The tool-loop family hold yields to the fallback's exclusions,
		// a model's and a route's alike: when they leave its family nothing
		// to serve, the turn leaves the family, before any failed route is
		// offered again.
		logToolLoopHold(arcContext, "lifted_by_exclusion", 0)
		excluded = hard
		available, offeredHeld, forced = offer(cellOut)
	}
	if len(available) == 0 && len(cellOut) > 0 {
		// Route exclusions are advice: when they would leave nothing, the
		// turn is offered as though no route had failed, rather than failing
		// every request until they expire.
		available, offeredHeld, forced = offer(nil)
	}
	if forced {
		held, retained, atBoundary = -1, false, true
	} else {
		held = offeredHeld
	}
	logRaylineARCPolicyTurn(arcContext.EpisodeIDHash, signals, transition, state.TurnIndex, turn, held >= 0, retained)
	// Every model excluded leaves nothing to serve, and the turn fails. The
	// package's fallback action is one of these same actions, so it is never
	// left when its peers are not.
	if len(available) == 0 {
		return nil, arcSelectionFailure("policy_no_available_action")
	}
	request := raylinearc.PolicyDecisionRequest{
		SchemaVersion: raylinearc.PolicyDecisionRequestSchema,
		Package:       raylinearc.PolicyPackageRef{Alias: scorer.alias, PackageSHA256: scorer.sha256},
		EpisodeIDHash: arcContext.EpisodeIDHash,
		ContextEpoch:  turn.ContextEpoch(),
		RequestFormat: arcContext.RequestFormat,
		Request:       clientRequest,
		Attribution:   attribution,
		Selection: raylinearc.PolicySelection{
			AvailableActionIDs:  available,
			OperatingPoint:      raylinearc.PolicyOperatingPoint{Name: "default"},
			PreferenceDimension: "overall",
		},
		Shadow:      []raylinearc.PolicyPackageRef{},
		EpisodeMode: scorer.episodeMode,
	}
	if sideCall {
		request.EpisodeMode = scorer.sideCallEpisodeMode(selector.now())
	}
	// Admission is checked after the episode lease and before each service
	// call, as the artifact mode checks it before encoding: a shed request
	// answers 429 and never occupies the service. A slot is held for one
	// call only, never through a session_busy backoff.
	admit := func() (func(), error) {
		release, acquireErr := armed.admission.Acquire()
		recordARCAdmission(armed.admission, acquireErr == nil)
		if acquireErr != nil {
			return nil, acquireErr
		}
		metrics.SetRaylineARCEncoderInflight(armed.admission.Inflight())
		return func() {
			release()
			metrics.SetRaylineARCEncoderInflight(armed.admission.Inflight())
		}, nil
	}
	started := selector.now()
	response, err := decidePolicyThroughBusy(ctx, armed.policy, request, scorer.busyWait, admit)
	// A strict cell's relaxed side call that the service refused as
	// unsupported is decided strict instead, as it was before side calls
	// went relaxed.
	var refused *raylinearc.PolicyServiceError
	if request.EpisodeMode != scorer.episodeMode && errors.As(err, &refused) &&
		refused.Class == raylinearc.PolicyRelaxedUnsupportedClass {
		scorer.sideCallsRelaxedUnsupported(selector.now(), "refused")
		request.EpisodeMode = scorer.episodeMode
		response, err = decidePolicyThroughBusy(ctx, armed.policy, request, scorer.busyWait, admit)
	}
	var shed *policyAdmissionError
	if errors.As(err, &shed) {
		return nil, boundedARCEncoderFailure(shed.err)
	}
	latency := selector.now().Sub(started)
	if err != nil {
		class := "transport"
		var failure *raylinearc.PolicyServiceError
		if errors.As(err, &failure) {
			class = failure.Class
		}
		logging.ComponentErrorEvent("extproc", "rayline_arc_policy_service_failed", map[string]interface{}{
			"class": class, "episode_id_hash": arcContext.EpisodeIDHash,
		})
		return nil, arcSelectionFailure("policy_service_" + class)
	}
	// A decision is only this package's if the service says it is: a service
	// that swapped the package under the alias is not the one readiness armed.
	if response.Package != request.Package {
		logging.ComponentErrorEvent("extproc", "rayline_arc_policy_service_failed", map[string]interface{}{
			"class": "package_mismatch", "episode_id_hash": arcContext.EpisodeIDHash,
		})
		return nil, arcSelectionFailure("policy_package_mismatch")
	}
	// Every action the package scores must be one this router can dispatch.
	// A package whose catalog outgrew the bindings is a different package in
	// all but its alias, so no turn is served from it.
	for _, action := range response.Actions {
		if _, bound := scorer.bindings[action.ActionID]; !bound {
			logging.ComponentErrorEvent("extproc", "rayline_arc_policy_service_failed", map[string]interface{}{
				"class": "catalog_unbound", "episode_id_hash": arcContext.EpisodeIDHash,
			})
			return nil, arcSelectionFailure("policy_catalog_unbound")
		}
	}
	binding, ok := scorer.bindings[response.Decision.SelectedActionID]
	if !ok {
		return nil, arcSelectionFailure("policy_unbound_action")
	}
	// Only an offered action may be dispatched: the offer already removed
	// excluded arms (vision, capability, operator-disabled) and, while the
	// schedule holds the model, every other model's actions.
	if !slices.Contains(available, response.Decision.SelectedActionID) ||
		!policySelectedActionAvailable(response) {
		return nil, arcSelectionFailure("policy_action_not_offered")
	}
	// A strict call advances the service's session and a relaxed one advances
	// none, so the revision says which the service actually served. A
	// service that ignored episode_mode would otherwise hold a relaxed cell's
	// episodes exclusively without anyone seeing it.
	// A strict cell's side call asked relaxed of a service that predates
	// episode_mode is answered strict, with a revision: that is how side
	// calls were served before, so it stands, and later ones go strict.
	servedRelaxed := request.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed
	if servedRelaxed && response.Encoding.SessionRevision != nil && request.EpisodeMode != scorer.episodeMode {
		scorer.sideCallsRelaxedUnsupported(selector.now(), "served_strict")
		servedRelaxed = false
	}
	if (response.Encoding.SessionRevision == nil) != servedRelaxed {
		return nil, arcSelectionFailure("policy_session_revision")
	}
	decision := policyDecision(scorer, response, binding, workerIDs, excluded)
	if !validARCDecision(decision, workerIDs) {
		return nil, arcSelectionFailure("artifact_result")
	}
	encoded := &raylinearc.EncoderResult{
		SerializedTokens:  response.Encoding.TokenCount,
		FullHistoryTokens: response.Encoding.TokenCount,
		SessionAction:     boundedPolicySessionAction(response.Encoding.SessionAction),
		SessionRevision:   policySessionRevision(response.Encoding.SessionRevision),
		EngineBuildID:     response.Encoding.EngineBuildID,
	}
	// encoder_latency keeps its artifact-mode meaning, the encode alone, as
	// the service timed it; the round trip is policy_latency.
	var encodeLatency time.Duration
	if encode := response.TimingMillis.Encode; encode != nil && *encode > 0 {
		encodeLatency = time.Duration(*encode * float64(time.Millisecond))
	}
	result := selector.selectionResult(armed, selCtx, arcContext, state, encoded, decision, encodeLatency)
	result.RaylineARC.PolicyLatency = latency
	result.RaylineARC.EncoderLatencyUnknown = response.TimingMillis.Encode == nil
	result.Reasoning = "policy-service ARC decision (" + response.Decision.Reason + ")"
	result.RaylineARC.PolicyActionID = response.Decision.SelectedActionID
	result.RaylineARC.PolicyArmID = response.Decision.SelectedArmID
	result.RaylineARC.ThinkingLevel = binding.level
	result.RaylineARC.PolicyActionModel = binding.model
	result.RaylineARC.WorkerProviderModel = scorer.workers[binding.arm].Model
	result.RaylineARC.WorkerRoute = scorer.routes[binding.arm]
	result.RaylineARC.PolicySideCall = sideCall
	if !sideCall {
		result.RaylineARC.PolicyTurnState = turn.Clone()
		result.RaylineARC.PolicyNextState = turn.Next(
			messages, response.Decision.SelectedActionID, response.Decision.SelectedArmID,
		)
	}
	if atBoundary && !retained {
		result.RaylineARC.PolicyBoundary = raylinearc.NewPolicyBoundaryDecision(binding.arm, state.TurnIndex, turn, messages)
	}
	if fallback {
		logRaylineARCFallbackDecision(arcContext.EpisodeIDHash, arcContext.RequestID, state.TurnIndex, turn, scorer.cellOutWorkers(loggedRoutes), forced, available, response)
	}
	return result, nil
}

// actionModel is the trained model an action serves: its binding's declared
// model, or its worker when the binding declares none.
func (scorer *policyServiceScorer) actionModel(actionID string) string {
	binding, ok := scorer.bindings[actionID]
	if !ok {
		return ""
	}
	if binding.model != "" {
		return binding.model
	}
	return scorer.armModel(binding.arm)
}

// cellExclusions are the arms whose routes the cell has excluded at now,
// with the failure class that excluded each.
func (scorer *policyServiceScorer) cellExclusions(now time.Time) map[int]string {
	var out map[int]string
	for arm, route := range scorer.routes {
		if route == "" {
			continue
		}
		if class, excluded := raylineARCWorkerExclusions.active(route, now); excluded {
			if out == nil {
				out = map[int]string{}
			}
			out[arm] = class
		}
	}
	return out
}

// cellOutWorkers lists route exclusions for the fallback log.
func (scorer *policyServiceScorer) cellOutWorkers(cellOut map[int]string) []map[string]string {
	routes := make([]map[string]string, 0, len(cellOut))
	for arm, worker := range scorer.workerIDs {
		if class, ok := cellOut[arm]; ok {
			routes = append(routes, map[string]string{"worker": worker, "class": class})
		}
	}
	return routes
}

// armModel is the trained model a worker serves, read from its bindings.
func (scorer *policyServiceScorer) armModel(arm int) string {
	for _, actionID := range scorer.actionOrder {
		if binding := scorer.bindings[actionID]; binding.arm == arm && binding.model != "" {
			return binding.model
		}
	}
	if arm >= 0 && arm < len(scorer.workerIDs) {
		return scorer.workerIDs[arm]
	}
	return ""
}

// logRaylineARCFallbackDecision records a decision taken with exclusions in
// force: what was excluded and why, what was offered, what was chosen and
// how the package scored it, so Phase 2 can learn refusal risk from it
// (ADR 0120). The request id joins it to the turn's llm_usage line.
func logRaylineARCFallbackDecision(
	episodeIDHash string,
	requestID string,
	turnIndex uint64,
	turn *raylinearc.PolicyEpisodeState,
	routes []map[string]string,
	forced bool,
	offered []string,
	response *raylinearc.PolicyDecisionResponse,
) {
	excluded := make([]map[string]string, 0, len(turn.Exclusions))
	for _, exclusion := range turn.Exclusions {
		excluded = append(excluded, map[string]string{"model": exclusion.Model, "class": exclusion.Class})
	}
	scores := make([]map[string]interface{}, 0, len(response.Actions))
	for _, action := range response.Actions {
		scores = append(scores, map[string]interface{}{
			"action_id": action.ActionID, "score": action.Score,
			"available": action.Available, "supported": action.Supported,
		})
	}
	reason := "excluded_offer"
	if forced {
		reason = "fallback_redecide"
	}
	logging.ComponentEvent("extproc", "rayline_arc_fallback_decision", map[string]interface{}{
		"episode_id_hash":    episodeIDHash,
		"request_id":         requestID,
		"turn_index":         turnIndex,
		"reason":             reason,
		"excluded":           excluded,
		"excluded_routes":    routes,
		"offered_action_ids": offered,
		"selected_action_id": response.Decision.SelectedActionID,
		"decision_reason":    response.Decision.Reason,
		"scores":             scores,
	})
}

const (
	policyBusyRetryFirst = 25 * time.Millisecond
	policyBusyRetryMax   = 400 * time.Millisecond
)

// policyAdmissionError is a decision refused admission before its call.
type policyAdmissionError struct{ err error }

func (err *policyAdmissionError) Error() string { return err.err.Error() }

// decidePolicyThroughBusy asks the service for the decision and retries a
// session_busy answer for up to wait. The service holds a strict episode for
// the length of one decision. Two main turns never reach it at once, because
// the episode lease serializes them, but a side call takes no lease: its
// decision can overlap a main turn's (Claude Code sends its title call
// alongside the first turn). Either one then waits out the other's decision,
// which is short, instead of being refused. Each call takes its own admission
// slot and returns it before any backoff.
func decidePolicyThroughBusy(
	ctx context.Context,
	client *raylinearc.PolicyServiceClient,
	request raylinearc.PolicyDecisionRequest,
	wait time.Duration,
	admit func() (func(), error),
) (*raylinearc.PolicyDecisionResponse, error) {
	deadline := time.Now().Add(wait)
	backoff := policyBusyRetryFirst
	for {
		release, admitErr := admit()
		if admitErr != nil {
			return nil, &policyAdmissionError{err: admitErr}
		}
		response, err := client.Decide(ctx, request)
		release()
		var failure *raylinearc.PolicyServiceError
		if err == nil || !errors.As(err, &failure) || failure.Class != "session_busy" ||
			time.Now().Add(backoff).After(deadline) {
			return response, err
		}
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			return response, err
		case <-timer.C:
		}
		if backoff *= 2; backoff > policyBusyRetryMax {
			backoff = policyBusyRetryMax
		}
	}
}

// policyClientRequestOf is the request the service projects, with the
// messages (Responses: items) attribution indexes and the role of each. A
// Responses request was materialized when the selection context was built;
// the others are the client's system, tools and messages exactly as received.
func policyClientRequestOf(
	arcContext *selection.RaylineARCSelectionContext,
) (raylinearc.PolicyClientRequest, []json.RawMessage, []string, error) {
	if arcContext.RequestFormat == policyFormatResponses {
		if len(arcContext.PolicyInput) == 0 {
			return raylinearc.PolicyClientRequest{}, nil, nil, errPolicyResponsesInputEmpty
		}
		roles, err := policyResponsesRoles(arcContext.PolicyInput)
		if err != nil {
			return raylinearc.PolicyClientRequest{}, nil, nil, err
		}
		return raylinearc.PolicyClientRequest{
			Input: arcContext.PolicyInput, Instructions: arcContext.PolicyInstructions,
		}, arcContext.PolicyInput, roles, nil
	}
	var body policyClientRequest
	if err := json.Unmarshal(arcContext.RawRequest, &body); err != nil || len(body.Messages) == 0 {
		return raylinearc.PolicyClientRequest{}, nil, nil, errors.New("messages must be a nonempty array")
	}
	messages, roles, err := policyMessages(body.Messages)
	if err != nil {
		return raylinearc.PolicyClientRequest{}, nil, nil, err
	}
	return raylinearc.PolicyClientRequest{System: body.System, Tools: body.Tools, Messages: body.Messages}, messages, roles, nil
}

// policyMessages splits the raw messages array without re-encoding any
// message, and reads each role for attribution.
func policyMessages(raw json.RawMessage) ([]json.RawMessage, []string, error) {
	var messages []json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil || len(messages) == 0 {
		return nil, nil, errors.New("messages must be a nonempty array")
	}
	roles := make([]string, len(messages))
	for index, message := range messages {
		var envelope struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(message, &envelope); err != nil {
			return nil, nil, err
		}
		roles[index] = envelope.Role
	}
	return messages, roles, nil
}

// policyDecision reports each arm's best action score, so the trace and the
// alternatives rank workers the way the service ranked their actions.
func policyDecision(
	scorer *policyServiceScorer,
	response *raylinearc.PolicyDecisionResponse,
	binding policyBinding,
	workerIDs []string,
	excluded []bool,
) raylinearc.Decision {
	count := len(workerIDs)
	scores := make([]float32, count)
	scored := make([]bool, count)
	for _, action := range response.Actions {
		bound, ok := scorer.bindings[action.ActionID]
		if !ok || !action.Available {
			continue
		}
		if !scored[bound.arm] || float32(action.Score) > scores[bound.arm] {
			scores[bound.arm] = float32(action.Score)
			scored[bound.arm] = true
		}
	}
	excludedArms := make([]bool, count)
	for arm := range excludedArms {
		excludedArms[arm] = !scored[arm] || (len(excluded) == count && excluded[arm])
	}
	return raylinearc.Decision{
		SelectedArm:                 binding.arm,
		SelectedWorker:              workerIDs[binding.arm],
		RawScores:                   scores,
		AdjustedScores:              append([]float32(nil), scores...),
		SwitchCostUSD:               make([]float64, count),
		CacheMissTokens:             make([]int, count),
		ColdSwitchUpgradeExemptions: make([]bool, count),
		ExcludedArms:                excludedArms,
	}
}

// policySelectedActionAvailable reports whether the response scores its own
// selection as available; a response that selects an action it marks
// unavailable contradicts itself and is not served.
func policySelectedActionAvailable(response *raylinearc.PolicyDecisionResponse) bool {
	for _, action := range response.Actions {
		if action.ActionID == response.Decision.SelectedActionID {
			return action.Available
		}
	}
	return false
}

func boundedPolicySessionAction(action string) string {
	if policySessionActions[action] {
		return action
	}
	return ""
}
