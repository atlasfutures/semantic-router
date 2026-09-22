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
	"math"
	"net/http"
	"sort"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	decision, algorithm, err := service.router.decisionOnlyRoutingTarget(request.Surface)
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
	episodeMode := raylineARCEpisodeRequired
	if request.Ephemeral {
		episodeMode = raylineARCEpisodeEphemeral
	}
	selectionContext.RaylineARC = service.router.buildRaylineARCSelectionContext(
		algorithm,
		requestContext,
		decision.ModelRefs,
		episodeMode,
	)
	if failure := selectionContext.RaylineARC.PreparationFailure; failure != "" {
		service.router.finalizeRaylineARCAbort(requestContext, failure)
		return routerruntime.RouteDecision{}, prepareFailureError(failure)
	}

	// An ephemeral lookup minted its own encoder session and owns closing it,
	// whatever happens next. Nothing else will: the ephemeral path prepares
	// no transaction, and the transaction is what closes a retained session
	// on the routed path. Left open, every lookup strands one retained
	// session on the encoder until eviction, and sustained lookup traffic
	// then evicts the live conversations it shares the card with.
	if request.Ephemeral {
		// Detached from response delivery on purpose. Run inline, this close
		// blocked the answer and was granted a fresh deadline of its own, so
		// a lookup that spent most of its budget selecting could take nearly
		// twice deadline_ms and still return 200 -- breaking the end-to-end
		// bound this endpoint documents, to do housekeeping the caller is not
		// waiting for.
		defer func() {
			go service.closeEphemeralEncoderSession(
				context.WithoutCancel(ctx),
				algorithm,
				selectionContext,
				requestContext,
			)
		}()
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
	// Both kinds of reuse count. A resumable encoder reports reuse as
	// RetainedPrefixTokens and leaves CachedPrefixTokens at zero, so reading
	// only the latter published cache_read_tokens: 0 for lookups whose input
	// came almost entirely from the encoder's retained cache -- and that
	// number is what the gateway meters on.
	decisionFacts.Usage = routerruntime.RouteUsage{
		EncodedInputTokens: trace.SerializedTokens,
		CacheReadTokens:    trace.CachedPrefixTokens + trace.RetainedPrefixTokens,
	}
	decisionFacts.Baseline = routeBaseline(selected.catalog, selected.reference)
	// The episode is reported only when the caller joined one. An ephemeral
	// consult scored against a fresh in-memory episode, so its turn index and
	// stay flag describe a trajectory that does not exist; reporting them
	// would read as continuity the caller does not have.
	if !request.Ephemeral && request.SessionID != "" {
		// The episode's own turn index, not the encoder's session revision.
		// Those are different counters: the revision tracks the retained
		// encoder session and stays at zero on the non-resumable path, while
		// the trajectory this field describes advances on every committed
		// decision. Reporting the revision made turn_index read as zero for
		// the ordinary encoder and lag behind the stored episode for the
		// resumable one.
		decisionFacts.Episode = &routerruntime.RouteEpisode{
			TurnIndex: committedEpisodeTurnIndex(requestContext),
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

// workerPricing publishes the whole rate card, cache included.
//
// Dropping the cache rates left a caller unable to reproduce the cost this
// endpoint invites them to compute: on a provider with discounted reads or
// charged writes, input and output alone do not add up to what they will be
// billed, and the savings figure derived from them is wrong in the direction
// that flatters us.
func workerPricing(worker raylinearc.WorkerManifest) routerruntime.RoutePricing {
	return routerruntime.RoutePricing{
		InputPerMTok:      worker.EstimatedInputCostPerToken * tokensPerMillion,
		OutputPerMTok:     worker.EstimatedOutputCostPerToken * tokensPerMillion,
		CacheReadPerMTok:  worker.EstimatedCacheReadCostPerToken * tokensPerMillion,
		CacheWritePerMTok: worker.EstimatedCacheWriteCostPerToken * tokensPerMillion,
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

// closeEphemeralEncoderSession releases the retained encoder session a
// minted identity created.
//
// It is a no-op on an encoder that retains nothing, which is the common
// deployment: the close call is cheap and the alternative is a leak that only
// appears under the resumable capability, where it would be found in
// production rather than here.
//
// Failure is logged and swallowed. The caller already has their route, and a
// close that did not land costs one stranded session rather than a wrong
// answer -- surfacing it as a lookup failure would trade a real answer for a
// housekeeping problem.
func (service *raylineARCDecisionService) closeEphemeralEncoderSession(
	ctx context.Context,
	algorithm *config.AlgorithmConfig,
	selectionContext *selection.SelectionContext,
	requestContext *RequestContext,
) {
	if service.router.raylineARCSessionClose == nil ||
		selectionContext == nil || selectionContext.RaylineARC == nil {
		return
	}
	episodeIDHash := selectionContext.RaylineARC.EpisodeIDHash
	if episodeIDHash == "" {
		return
	}
	// The replicas this session actually touched. The finished trace carries
	// them on a successful lookup; the selection context carries them from
	// the moment the encode returned, which is the only source that survives
	// a failure between encoding and a validated result. An empty list then
	// means the encode never ran, and there is genuinely nothing to close.
	var visited []string
	if requestContext != nil && requestContext.VSRRaylineARC != nil {
		visited = requestContext.VSRRaylineARC.EncoderVisitedReplicaIDs
	}
	if len(visited) == 0 {
		visited = selectionContext.RaylineARC.EncoderVisitedReplicaIDs
	}
	closeContext, cancel := context.WithTimeout(
		ctx,
		ephemeralSessionCloseTimeout(algorithm),
	)
	defer cancel()
	if _, err := service.router.raylineARCSessionClose(
		closeContext,
		episodeIDHash,
		visited,
	); err != nil {
		logging.ComponentErrorEvent("extproc", "routing_decision_session_close_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
}

// ephemeralSessionCloseTimeout bounds the close so a wedged encoder cannot
// hold the request goroutine after the answer has been produced.
func ephemeralSessionCloseTimeout(algorithm *config.AlgorithmConfig) time.Duration {
	if algorithm != nil && algorithm.RaylineARC != nil {
		if seconds := algorithm.RaylineARC.RoutesAPI.DeadlineMS; seconds > 0 {
			return time.Duration(seconds) * time.Millisecond
		}
	}
	return config.DefaultRaylineARCRoutesDeadlineMS * time.Millisecond
}

// committedEpisodeTurnIndex reads the trajectory position this decision
// advanced, from the state the commit stored.
//
// Selection does not mutate the episode state: the commit clones it,
// increments the clone, persists that, and swaps it onto the transaction. So
// the pre-commit state is always one behind, and reading it reported
// turn_index 0 for the first committed lookup of an episode already at 1.
func committedEpisodeTurnIndex(requestContext *RequestContext) int {
	if requestContext == nil || requestContext.RaylineARCTransaction == nil ||
		requestContext.RaylineARCTransaction.state == nil {
		return 0
	}
	turnIndex := requestContext.RaylineARCTransaction.state.TurnIndex
	if turnIndex > math.MaxInt32 {
		return math.MaxInt32
	}
	return int(turnIndex)
}

// commitDecisionOnlyEpisode advances the episode at decision time. The request
// pipeline waits for upstream headers before committing, but a decision-only
// consult has no upstream: the caller executes the worker out of this router's
// sight. Deferring would leave the lease pending forever and score the next
// turn against a trajectory that never advanced, so the selected arm becomes
// the previous arm optimistically, here. The status argument is a formality
// the ARC transaction ignores.
func commitDecisionOnlyEpisode(ctx context.Context, requestContext *RequestContext) error {
	// An ephemeral consult prepared nothing, so there is nothing to advance.
	// That is the intended terminal state, not a missing step.
	if requestContext.VSRRaylineARC != nil && requestContext.RaylineARCTransaction == nil {
		return nil
	}
	if requestContext.SelectionTransaction == nil {
		return errors.New("decision-only routing prepared no episode transaction")
	}
	// The commit RUNS on its own bounded context, detached from the lookup's
	// deadline. Sharing it meant a deadline that expired mid-commit cancelled
	// the cleanup too: lease renewal stops, the lease is neither committed nor
	// released, and every later turn on that session is refused as contended
	// until the lease TTL runs out. The answer is late either way; the session
	// does not have to be broken as well.
	commitContext, cancel := context.WithTimeout(
		context.WithoutCancel(ctx),
		episodeFinalizeTimeout,
	)
	committed := make(chan error, 1)
	go func() {
		// Cancelled from in here, not by this function returning: the point of
		// detaching is that the commit outlives a caller who stopped waiting.
		defer cancel()
		_, err := requestContext.SelectionTransaction.commitOnHeaders(
			commitContext,
			http.StatusOK,
		)
		committed <- err
	}()
	// ...but the caller only WAITS for it inside the lookup's own budget.
	// Detaching the run and the wait together let a slow commit add its whole
	// finalization timeout on top of deadline_ms -- nearly 6.5s against a
	// documented 1.5s -- and still answer 200, which breaks the bound this
	// endpoint publishes. It cannot simply answer without the commit either:
	// the episode turn index it reports is the one the commit stores, so an
	// unwaited commit would publish a trajectory position that is one behind.
	select {
	case err := <-committed:
		if err != nil {
			return fmt.Errorf("decision-only routing could not commit the episode: %w", err)
		}
		return nil
	case <-ctx.Done():
		// The commit is still running on its detached context and will resolve
		// the lease on its own. What is gone is this caller's budget, and the
		// deadline is reported as a deadline -- the endpoint's own 504 -- not
		// hidden behind a 200 carrying a stale episode.
		return fmt.Errorf(
			"decision-only routing ran out of budget before the episode committed: %w",
			ctx.Err(),
		)
	}
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
			"%w: decision-only routing could not decode the consult body: %w",
			routerruntime.ErrRouteDecisionInvalidBody,
			err,
		)
	}
	decoded.Trusted.SourceFormat = wireFormat
	decoded.Trusted.CorrelationID = request.DecisionID
	// An ephemeral consult sets no episode header at all: the absence is what
	// the builder reads to skip preparation. Every other consult keeps the
	// existing contract, where a caller that named no conversation still gets
	// its own single-turn episode rather than sharing one.
	headers := map[string]string{}
	if !request.Ephemeral {
		episodeIdentity := request.SessionID
		if episodeIdentity == "" {
			episodeIdentity = "decision-only:" + uuid.NewString()
		}
		headers[algorithm.RaylineARC.Episode.IDHeader] = episodeIdentity
	}
	return &RequestContext{
		Headers:          headers,
		RequestID:        request.DecisionID,
		SourceFormat:     wireFormat,
		SemanticRequest:  &decoded,
		ProtocolEnvelope: envelope,
		TraceContext:     ctx,
	}, nil
}

// decisionOnlyRoutingTarget finds the decision that serves a route consult on
// the surface that asked.
//
// Exactly one Rayline ARC decision may serve it. Zero means the deployment
// never configured decision-only routing; more than one means the consult
// carries nothing that could choose between them, and guessing would route
// live traffic through a policy nobody selected.
//
// The surface is a parameter because the two endpoints disambiguate
// differently and must not borrow each other's rule. Only POST /v1/routes has
// a routes_api block to read, so only it can treat enabling that block as a
// claim; the legacy POST /v1/route keeps its original fail-closed rule, and
// letting the claim decide for it would silently move it onto a policy its
// caller never selected the moment some unrelated decision turned the new
// endpoint on.
func (r *OpenAIRouter) decisionOnlyRoutingTarget(
	surface routerruntime.RouteDecisionSurface,
) (
	*config.Decision,
	*config.AlgorithmConfig,
	error,
) {
	if r.Config == nil {
		return nil, nil, errors.New("router configuration is unavailable")
	}
	var arcDecisions []*config.Decision
	var enabled []*config.Decision
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
		arcDecisions = append(arcDecisions, decision)
		if decision.Algorithm.RaylineARC.RoutesAPI.Enabled {
			enabled = append(enabled, decision)
		}
	}

	// A decision that turns the endpoint on has claimed it, and that is the
	// only thing that can disambiguate a multi-decision deployment. Scanning
	// pairwise instead meant two disabled decisions ahead of the enabled one
	// raised ambiguity before the enabled one was reached.
	//
	// Confined to the surface that owns routes_api. The legacy consult never
	// reads this block, so a decision enabling it has claimed nothing on that
	// surface, and honouring the claim there would answer a /v1/route consult
	// from a decision its caller had no way to choose.
	if surface == routerruntime.RouteDecisionSurfaceRoutes {
		if len(enabled) == 1 {
			return enabled[0], enabled[0].Algorithm, nil
		}
		if len(enabled) > 1 {
			return nil, nil, fmt.Errorf(
				"decision-only routing is ambiguous: decisions '%s' and '%s' both enable routes_api on %s; enable it on exactly one",
				enabled[0].Name,
				enabled[1].Name,
				config.RaylineARCAlgorithmType,
			)
		}
	}

	// Nothing claimed it, or the caller is the legacy consult, which has no
	// routes_api to read. Either way the original rule applies: one ARC
	// decision serves it, and more than one is ambiguous with nothing to
	// choose between them. Failing closed here is deliberate -- guessing
	// would route live traffic through a policy nobody selected.
	if len(arcDecisions) == 1 {
		return arcDecisions[0], arcDecisions[0].Algorithm, nil
	}
	if len(arcDecisions) > 1 {
		return nil, nil, fmt.Errorf(
			"decision-only routing is ambiguous: decisions '%s' and '%s' both use %s",
			arcDecisions[0].Name,
			arcDecisions[1].Name,
			config.RaylineARCAlgorithmType,
		)
	}
	return nil, nil, fmt.Errorf(
		"decision-only routing requires a decision with algorithm.type=%s",
		config.RaylineARCAlgorithmType,
	)
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
