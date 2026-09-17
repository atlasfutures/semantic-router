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

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineARCDecisionService answers decision-only route consults out of the
// configured Rayline ARC decision.
//
// It runs the same episode-and-selection path the request pipeline runs, and
// stops where dispatch would begin: no candidate binding, no body mutation, no
// upstream call. The caller executes the chosen worker itself.
type raylineARCDecisionService struct {
	router *OpenAIRouter
}

// routeDecisionRuntimeState exposes decision-only routing to the management
// listener without handing it the router.
func (r *OpenAIRouter) routeDecisionRuntimeState() routerruntime.RouteDecisionRuntime {
	if r == nil {
		return nil
	}
	return &raylineARCDecisionService{router: r}
}

func (service *raylineARCDecisionService) RouteDecision(
	ctx context.Context,
	request routerruntime.RouteDecisionRequest,
) (routerruntime.RouteDecision, error) {
	if service == nil || service.router == nil {
		return routerruntime.RouteDecision{}, errors.New("router is unavailable")
	}
	decision, algorithm, err := service.router.decisionOnlyRoutingTarget()
	if err != nil {
		return routerruntime.RouteDecision{}, err
	}

	requestContext, err := service.decisionOnlyRequestContext(ctx, algorithm, request)
	if err != nil {
		return routerruntime.RouteDecision{}, err
	}
	selectionContext := &selection.SelectionContext{
		DecisionName:    decision.Name,
		CandidateModels: decision.ModelRefs,
		SessionID:       request.SessionID,
	}
	// Reuses the request pipeline's own preparation: it normalizes the body
	// into turns, hashes the episode identity, and acquires the episode lease.
	// From here on every exit must be terminal for that lease.
	selectionContext.RaylineARC = service.router.buildRaylineARCSelectionContext(
		algorithm,
		requestContext,
		decision.ModelRefs,
	)
	if failure := selectionContext.RaylineARC.PreparationFailure; failure != "" {
		service.router.finalizeRaylineARCAbort(requestContext, failure)
		return routerruntime.RouteDecision{}, prepareFailureError(failure)
	}

	selected, err := service.selectWorker(ctx, algorithm, selectionContext, requestContext)
	if err != nil {
		return routerruntime.RouteDecision{}, err
	}
	worker := selected.worker
	decisionFacts := routerruntime.RouteDecision{
		SelectedWorker: worker.ID,
		WorkerModel:    worker.Model,
		// Only the declared provider slug. The dispatch backend would be a
		// plausible substitute and is exactly the kind of near-miss that reads
		// as measured in an offline join, so an undeclared provider stays empty.
		Provider: worker.OpenRouterProviderSlug,
		Thinking: routerruntime.RouteThinking{
			Mode:         worker.ThinkingMode,
			BudgetTokens: worker.ReasoningBudgetTokens,
		},
		SelectedPricing: workerPricing(worker),
		Warnings:        []string{},
	}
	trace := selected.result.RaylineARC
	decisionFacts.Checkpoint = trace.ArtifactRevision
	decisionFacts.Alternatives = routeAlternatives(trace, selected.catalog)
	decisionFacts.Usage = routerruntime.RouteUsage{
		EncodedInputTokens: trace.SerializedTokens,
		CacheReadTokens:    trace.CachedPrefixTokens,
	}
	decisionFacts.Baseline = routeBaseline(selected.catalog, selected.reference)
	// The episode is reported only when the caller joined one. A consult that
	// did not gets its own single-turn trajectory, and reporting that back as
	// turn zero would read as continuity the caller does not have.
	if request.SessionID != "" {
		decisionFacts.Episode = &routerruntime.RouteEpisode{
			TurnIndex: trace.SessionRevision,
			Stayed:    trace.Stayed,
		}
	}
	return decisionFacts, nil
}

// selectedRoute is everything one decision publishes, kept together so the
// assembly above reads facts rather than re-deriving them.
type selectedRoute struct {
	worker    raylinearc.WorkerManifest
	result    *selection.SelectionResult
	catalog   raylineARCWorkerProvider
	reference string
}

func workerPricing(worker raylinearc.WorkerManifest) routerruntime.RoutePricing {
	return routerruntime.RoutePricing{
		InputPerMTok:  worker.EstimatedInputCostPerToken * tokensPerMillion,
		OutputPerMTok: worker.EstimatedOutputCostPerToken * tokensPerMillion,
	}
}

// tokensPerMillion converts the manifest's per-token rates to the per-million
// unit every provider publishes and every caller reasons in.
const tokensPerMillion = 1_000_000

// routeAlternatives reports the arms that were scored and not chosen, best
// first.
//
// Arms a hard constraint removed before scoring are left out: their adjusted
// score is not a comparison this router made, and publishing it would invite
// a caller to read a filtered-out arm as a near miss.
func routeAlternatives(
	trace *selection.RaylineARCTrace,
	catalog raylineARCWorkerProvider,
) []routerruntime.RouteAlternative {
	alternatives := make([]routerruntime.RouteAlternative, 0, len(trace.AdjustedScores))
	for arm, score := range trace.AdjustedScores {
		if arm == trace.SelectedArm {
			continue
		}
		if arm < len(trace.ExcludedArms) && trace.ExcludedArms[arm] {
			continue
		}
		worker, found := catalog.Worker(arm)
		if !found {
			continue
		}
		alternatives = append(alternatives, routerruntime.RouteAlternative{
			Model: worker.Model,
			Score: float64(score),
		})
	}
	sort.SliceStable(alternatives, func(first, second int) bool {
		return alternatives[first].Score > alternatives[second].Score
	})
	return alternatives
}

// routeBaseline resolves the artifact's declared reference worker to a rate
// card. An artifact that declares none yields an empty baseline rather than a
// guess: a counterfactual nobody chose is worse than no counterfactual.
func routeBaseline(
	catalog raylineARCWorkerProvider,
	reference string,
) routerruntime.RouteBaseline {
	if reference == "" {
		return routerruntime.RouteBaseline{}
	}
	for arm := 0; ; arm++ {
		worker, found := catalog.Worker(arm)
		if !found {
			return routerruntime.RouteBaseline{}
		}
		if worker.ID != reference {
			continue
		}
		return routerruntime.RouteBaseline{
			Model:   worker.Model,
			Pricing: workerPricing(worker),
		}
	}
}

func (service *raylineARCDecisionService) selectWorker(
	ctx context.Context,
	algorithm *config.AlgorithmConfig,
	selectionContext *selection.SelectionContext,
	requestContext *RequestContext,
) (selectedRoute, error) {
	route, failure, err := service.resolveWorker(
		ctx,
		algorithm,
		selectionContext,
		requestContext,
	)
	if err != nil {
		service.router.finalizeRaylineARCAbort(requestContext, failure)
		return selectedRoute{}, err
	}
	if err := commitDecisionOnlyEpisode(ctx, requestContext); err != nil {
		return selectedRoute{}, err
	}
	return route, nil
}

// resolveWorker runs selection and maps the chosen arm back to its worker. It
// deliberately stops short of the dispatch contract: a decision-only consult
// publishes selection facts, so binding a candidate for execution here would
// promise an upstream call that never happens.
func (service *raylineARCDecisionService) resolveWorker(
	ctx context.Context,
	algorithm *config.AlgorithmConfig,
	selectionContext *selection.SelectionContext,
	requestContext *RequestContext,
) (selectedRoute, string, error) {
	if err := selection.ValidateSelectionContext(selectionContext); err != nil {
		return selectedRoute{}, "invalid_context", err
	}
	selector := service.router.selectorForDecisionMethod(
		selection.MethodRaylineARC,
		algorithm,
		requestContext,
	)
	if selector == nil {
		return selectedRoute{}, "missing_selector", errors.New(
			"no rayline ARC selector is registered",
		)
	}
	result, err := selector.Select(ctx, selectionContext)
	if err != nil {
		return selectedRoute{}, "selection_failed", selectionFailureError(err)
	}
	if err := selection.ValidateSelectionResult(selectionContext, result); err != nil {
		return selectedRoute{}, "invalid_result", err
	}
	if result.RaylineARC == nil {
		return selectedRoute{}, "missing_trace", errors.New(
			"rayline ARC selection returned no trace",
		)
	}
	provider, ok := selector.(raylineARCWorkerProvider)
	if !ok {
		return selectedRoute{}, "missing_manifest", errors.New(
			"rayline ARC selector exposes no worker manifest",
		)
	}
	worker, found := provider.Worker(result.RaylineARC.SelectedArm)
	if !found || worker.ID != result.SelectedModel {
		return selectedRoute{}, "worker_mismatch", errors.New(
			"rayline ARC selected arm does not match its worker manifest",
		)
	}
	requestContext.VSRRaylineARC = result.RaylineARC
	requestContext.RaylineARCTransaction.markSelectionWithAffinity(
		result.RaylineARC.SelectedArm,
		result.RaylineARC.SerializedTokens,
		result.RaylineARC.EncoderReplicaID,
		result.RaylineARC.EncoderVisitedReplicaIDs,
	)
	route := selectedRoute{worker: worker, result: result, catalog: provider}
	if baseline, ok := selector.(raylineARCReferenceWorkerProvider); ok {
		route.reference = baseline.ReferenceWorker()
	}
	return route, "", nil
}

// commitDecisionOnlyEpisode advances the episode at decision time. The request
// pipeline waits for upstream headers before committing, but a decision-only
// consult has no upstream: the caller executes the worker out of this router's
// sight. Deferring would leave the lease pending forever and score the next
// turn against a trajectory that never advanced, so the selected arm becomes
// the previous arm optimistically, here. The status argument is a formality
// the ARC transaction ignores.
func commitDecisionOnlyEpisode(ctx context.Context, requestContext *RequestContext) error {
	if requestContext.SelectionTransaction == nil {
		return errors.New("decision-only routing prepared no episode transaction")
	}
	if _, err := requestContext.SelectionTransaction.commitOnHeaders(
		ctx,
		http.StatusOK,
	); err != nil {
		return fmt.Errorf("decision-only routing could not commit the episode: %w", err)
	}
	return nil
}

// defaultConsultWireFormat is the wire contract a route consult body is read
// as when the caller names none. Callers of the legacy consult bridge relay
// the Anthropic Messages request they are about to send, so that body reads
// under the same codec the routed ingress path uses.
const defaultConsultWireFormat = llmprotocol.AnthropicMessagesV1

// decisionOnlyRequestContext is the minimum the ARC preparation path reads. It
// is not a real request: there is no stream, no upstream, and no dispatch.
//
// The body is decoded through the public codec here, because the selector
// reads the decoded request and nothing else. A body the codec refuses fails
// the consult closed, which is what it did before the decode moved here.
//
// The caller's session identity is placed under the algorithm's configured
// episode header rather than under a header name hardcoded here, so the
// episode contract stays owned by config. A consult with no session gets its
// own single-turn episode: sharing one would braid unrelated callers into a
// single trajectory.
//
// request.ExecutedModel is absent on purpose. It is record-only, and this
// struct is exactly the state selection reads.
func (service *raylineARCDecisionService) decisionOnlyRequestContext(
	ctx context.Context,
	algorithm *config.AlgorithmConfig,
	request routerruntime.RouteDecisionRequest,
) (*RequestContext, error) {
	engine, err := service.router.protocolEngine()
	if err != nil {
		return nil, fmt.Errorf(
			"decision-only routing has no protocol runtime: %w",
			err,
		)
	}
	wireFormat := request.WireFormat
	if wireFormat == "" {
		wireFormat = defaultConsultWireFormat
	}
	decoded, envelope, _, err := engine.DecodeRequestForMutation(
		wireFormat,
		request.Body,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"decision-only routing could not decode the consult body: %w",
			err,
		)
	}
	decoded.Trusted.SourceFormat = wireFormat
	decoded.Trusted.CorrelationID = request.DecisionID
	episodeIdentity := request.SessionID
	if episodeIdentity == "" {
		episodeIdentity = "decision-only:" + uuid.NewString()
	}
	return &RequestContext{
		Headers:          map[string]string{algorithm.RaylineARC.Episode.IDHeader: episodeIdentity},
		RequestID:        request.DecisionID,
		SourceFormat:     wireFormat,
		SemanticRequest:  &decoded,
		ProtocolEnvelope: envelope,
		TraceContext:     ctx,
	}, nil
}

// decisionOnlyRoutingTarget finds the decision that serves route consults.
//
// Exactly one Rayline ARC decision may claim it. Zero means the deployment
// never configured decision-only routing; more than one means the consult
// carries nothing that could choose between them, and guessing would route
// live traffic through a policy nobody selected.
func (r *OpenAIRouter) decisionOnlyRoutingTarget() (
	*config.Decision,
	*config.AlgorithmConfig,
	error,
) {
	if r.Config == nil {
		return nil, nil, errors.New("router configuration is unavailable")
	}
	var target *config.Decision
	for index := range r.Config.Decisions {
		decision := &r.Config.Decisions[index]
		// The algorithm block, not just the type name. A decision can name
		// rayline_arc without carrying its configuration; config validation
		// normally catches that, but this path answers over the network and
		// must not turn a bad config into a crashed router.
		if !raylineARCSelection(decision.Algorithm) ||
			decision.Algorithm.RaylineARC == nil {
			continue
		}
		if target != nil {
			return nil, nil, fmt.Errorf(
				"decision-only routing is ambiguous: decisions '%s' and '%s' both use %s",
				target.Name,
				decision.Name,
				config.RaylineARCAlgorithmType,
			)
		}
		target = decision
	}
	if target == nil {
		return nil, nil, fmt.Errorf(
			"decision-only routing requires a decision with algorithm.type=%s",
			config.RaylineARCAlgorithmType,
		)
	}
	return target, target.Algorithm, nil
}

// prepareFailureError classifies a preparation failure for the caller.
//
// Contention failures are wrapped so the adapter can answer 429: the router is
// healthy and the request is well formed, the session's episode was simply
// already in use or the store was full. Every other failure stays a plain
// error and reads as 503, because waiting will not fix it.
// selectionFailureError classifies a selection failure for the caller.
//
// An admission shed is the same back-pressure as a contended lease: the router
// is healthy and the request is well formed, the encoder's in-flight cap is
// simply spent. It escaped prepareFailureError because it surfaces during
// selection, not episode preparation. Every other selection failure stays a
// plain error and reads as 503, because waiting will not fix it.
func selectionFailureError(err error) error {
	var failure *raylineARCSelectionFailure
	if errors.As(err, &failure) && failure.contended() {
		return fmt.Errorf("%w: %w", routerruntime.ErrRouteDecisionContended, err)
	}
	return err
}

func prepareFailureError(failure string) error {
	base := fmt.Errorf(
		"decision-only routing could not prepare the episode: %s",
		failure,
	)
	if selectionFailureIsContended(failure) {
		return fmt.Errorf("%w: %w", routerruntime.ErrRouteDecisionContended, base)
	}
	return base
}
