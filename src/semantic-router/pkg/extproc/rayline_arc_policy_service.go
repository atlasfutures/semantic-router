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
	schedule    string
	alias       string
	sha256      string
	workerIDs   []string
	workers     []raylinearc.WorkerManifest
	bindings    map[string]policyBinding
	actionOrder []string
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

func newPolicyServiceScorer(
	cfg *config.RouterConfig,
	decision *config.Decision,
) *policyServiceScorer {
	policy := decision.Algorithm.RaylineARC.PolicyService
	scorer := &policyServiceScorer{
		schedule: policy.ModelSchedule,
		alias:    policy.PackageAlias,
		sha256:   policy.PackageSHA256,
		bindings: make(map[string]policyBinding, len(policy.Bindings)),
	}
	index := make(map[string]int, len(decision.ModelRefs))
	for arm, modelRef := range decision.ModelRefs {
		index[modelRef.Model] = arm
		scorer.workerIDs = append(scorer.workerIDs, modelRef.Model)
		scorer.workers = append(scorer.workers, policyWorkerManifest(cfg, modelRef))
	}
	for _, binding := range policy.Bindings {
		scorer.bindings[binding.ActionID] = policyBinding{arm: index[binding.Worker], level: binding.Level, model: binding.Model}
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
		return newRaylineARCSelector(nil, nil, nil, policy.PackageSHA256), nil, nil, nil, class
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
	probe := func(ctx context.Context) error {
		return client.RequirePackage(ctx, policy.PackageAlias, policy.PackageSHA256)
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
	turn, attribution := raylinearc.PolicyTurn(state.Policy, messages, roles, state.TurnIndex)
	held := -1
	if scorer.schedule != "" && state.PreviousArm != nil &&
		!raylinearc.ModelChangeAllowed(state.TurnIndex, turn.EpochStartTurn) {
		held = *state.PreviousArm
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
		available = append(available, actionID)
	}
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
		Shadow: []raylinearc.PolicyPackageRef{},
	}
	// Admission is checked after the episode lease and before the service
	// call, as the artifact mode checks it before encoding: a shed request
	// answers 429 and never occupies the service.
	release, admitErr := armed.admission.Acquire()
	if admitErr != nil {
		recordARCAdmission(armed.admission, false)
		return nil, boundedARCEncoderFailure(admitErr)
	}
	defer func() {
		release()
		metrics.SetRaylineARCEncoderInflight(armed.admission.Inflight())
	}()
	recordARCAdmission(armed.admission, true)
	started := selector.now()
	response, err := armed.policy.Decide(ctx, request)
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
	decision := policyDecision(scorer, response, binding, workerIDs, excluded)
	if !validARCDecision(decision, workerIDs) {
		return nil, arcSelectionFailure("artifact_result")
	}
	encoded := &raylinearc.EncoderResult{
		SerializedTokens:  response.Encoding.TokenCount,
		FullHistoryTokens: response.Encoding.TokenCount,
		SessionAction:     boundedPolicySessionAction(response.Encoding.SessionAction),
		SessionRevision:   response.Encoding.SessionRevision,
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
	result.RaylineARC.PolicyNextState = turn.Next(
		messages, response.Decision.SelectedActionID, response.Decision.SelectedArmID,
	)
	return result, nil
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
