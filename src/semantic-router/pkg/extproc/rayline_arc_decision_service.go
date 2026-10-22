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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
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
	// A Decisions draw is never trajectory state. A lookup that asks for
	// exploration (routing.temperature > 0) joins no episode, so its drawn
	// action never becomes a committed previous arm and no later turn is
	// attributed to it, whatever routes_api.episode_writes says. It is
	// training_data: not_allowed by contract.
	untracked := false
	if request.Routing != nil && request.Routing.Exploration != nil && !request.Ephemeral {
		request.Ephemeral, untracked = true, true
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
	selectionContext.RaylineARC.RouteControls = request.Routing
	if failure := selectionContext.RaylineARC.PreparationFailure; failure != "" {
		// Counted here because this path returns before Select(), whose
		// failure constructor is where the routed path counts the same
		// class; without this a preparation refusal on a lookup is invisible
		// to the failure metric.
		metrics.RecordRaylineARCFailure(failure)
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
	if untracked {
		decisionFacts.Warnings = append(decisionFacts.Warnings, raylineRoutesExplorationUntracked)
	}
	trace := selected.result.RaylineARC
	decisionFacts.ThinkingLevel = trace.ThinkingLevel
	decisionFacts.PolicyActionID = trace.PolicyActionID
	decisionFacts.Checkpoint = trace.ArtifactRevision
	// On a snapshot cell, the package of the snapshot that decided.
	toolsPolicy := routePolicyService(algorithm)
	if toolsPolicy != nil && requestContext.VSRSelectedDecision != nil {
		toolsPolicy = raylineARCServedPolicy(requestContext)
	}
	decisionFacts.ToolsEncoded = routeToolsEncoded(trace.PolicyToolDefinitionsStatus, toolsPolicy)
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
	if request.Routing != nil {
		decisionFacts.Results = routeResults(request.Routing, trace, selected.catalog)
		decisionFacts.Exploration = trace.RouteExploration
		decisionFacts.Predictions = trace.RoutePredictions
		decisionFacts.PredictionsApplied = trace.RoutePredictionsApplied
		decisionFacts.PackageSHA256 = trace.PolicyPackageSHA256
	}
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

// routeResults publishes the checked result picks in the order the points
// were asked, each identified as the selected arm is.
func routeResults(
	controls *selection.RaylineARCRouteControls,
	trace *selection.RaylineARCTrace,
	catalog raylineARCWorkerProvider,
) []routerruntime.RouteResult {
	results := make([]routerruntime.RouteResult, 0, len(controls.Points))
	for _, point := range controls.Points {
		picked, ok := trace.RouteResults[point.Name]
		if !ok {
			continue
		}
		worker, found := catalog.Worker(picked.Arm)
		if !found {
			continue
		}
		results = append(results, routerruntime.RouteResult{
			Name:           point.Name,
			CostWeight:     point.CostWeight,
			SelectedWorker: worker.ID,
			WorkerModel:    worker.Model,
			Provider:       worker.OpenRouterProviderSlug,
			Thinking: routerruntime.RouteThinking{
				Mode:         worker.ThinkingMode,
				BudgetTokens: worker.ReasoningBudgetTokens,
			},
			ThinkingLevel:     picked.ThinkingLevel,
			PolicyActionID:    picked.Pick.SelectedActionID,
			Exploration:       picked.Pick.Exploration,
			StageOneCandidate: picked.Pick.StageOneCandidate,
		})
	}
	return results
}

// raylineRoutesExplorationUntracked says a lookup that would have joined its
// conversation's episode did not, because it drew at a caller temperature.
const raylineRoutesExplorationUntracked = "exploration_untracked: a lookup that draws at routing.temperature joins no " +
	"episode; a Decisions draw never becomes trajectory state"

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
		// A model the package's stage one does not offer is not one ARC
		// can pick: its trained actions are scored, never an alternative.
		if trace.PolicyArmOffered != nil && (arm >= len(trace.PolicyArmOffered) || !trace.PolicyArmOffered[arm]) {
			continue
		}
		worker, found := catalog.Worker(arm)
		if !found {
			continue
		}
		alternative := routerruntime.RouteAlternative{
			Model: worker.Model,
			// Same identity the selected arm publishes. Without it two arms
			// serving one model through different providers print as two
			// identical lines, and the list that claims to explain the choice
			// cannot say what was rejected.
			Worker:   worker.ID,
			Provider: worker.OpenRouterProviderSlug,
			Score:    float64(score),
		}
		if arm < len(trace.PolicyArmActions) {
			alternative.ActionID = trace.PolicyArmActions[arm].ActionID
			alternative.ThinkingLevel = trace.PolicyArmActions[arm].ThinkingLevel
		}
		alternatives = append(alternatives, alternative)
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
	stageRaylineARCPolicySelection(requestContext, result.RaylineARC, result.SelectedModel)
	service.router.publishRaylineARCDecision(requestContext, result)
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
// pipeline waits for the complete upstream response before committing, but a decision-only
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
		_, err := requestContext.SelectionTransaction.commit(
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
		// The caller is getting a 504 and no route, so the episode should not
		// advance: committing anyway records the selected arm as the previous
		// arm of a turn nobody ran, and the retry is then scored against it.
		cancel()
		// Cancelling is not enough on its own to know that. Not every store
		// honours the context -- MemoryEpisodeStore.Commit ignores it
		// outright -- so a commit already inside the store can still land
		// after cancel() returns. Waiting for the outcome is what turns
		// "probably did not commit" into a fact, and it is bounded: the
		// commit was just cancelled, so a store that honours cancellation
		// returns at once and one that does not is finishing a map write.
		select {
		case err := <-committed:
			if err == nil {
				// It landed. There is no uncommit, so the honest thing is to
				// say so rather than issue an abort that will silently do
				// nothing against a terminal transaction. The caller still
				// gets the deadline, because their budget is still gone.
				logging.ComponentErrorEvent("extproc", "routing_decision_episode_committed_after_deadline", map[string]interface{}{
					"detail": "the episode advanced for a lookup that answered 504; the next turn is scored against an arm the caller was never given",
				})
			} else {
				raylineARCReleaseTimedOutEpisode(ctx, requestContext)
			}
		case <-time.After(raylineARCRoutesCommitGrace):
			// The store is wedged: neither committed nor cancelled inside a
			// grace that is generous for a lease write. Release on a detached
			// context and let the transaction's own lock serialise it, which
			// is a no-op if the commit turns out to have landed.
			go raylineARCReleaseTimedOutEpisode(context.WithoutCancel(ctx), requestContext)
		}
		return fmt.Errorf(
			"decision-only routing ran out of budget before the episode committed: %w",
			ctx.Err(),
		)
	}
}

// raylineARCRoutesCommitGrace bounds how long a timed-out lookup waits to
// learn whether its cancelled commit landed. It is not a second deadline: the
// commit has already been cancelled, so this only covers a store that ignores
// cancellation and is finishing a write it had started.
const raylineARCRoutesCommitGrace = 250 * time.Millisecond

// raylineARCReleaseTimedOutEpisode returns the lease without advancing the
// trajectory.
//
// A transaction that already finalized -- because its commit failed, which is
// what a cancelled commit is -- released the lease on that path itself, and
// reports the commit's own error here rather than an abort failure. That is
// an observation about which path did the releasing, not a leak, so it is
// recorded as one.
func raylineARCReleaseTimedOutEpisode(
	ctx context.Context,
	requestContext *RequestContext,
) {
	released, err := requestContext.SelectionTransaction.abort(
		context.WithoutCancel(ctx),
		raylineARCRoutesDeadlineAbortClass,
	)
	if err == nil {
		return
	}
	event := "routing_decision_episode_released_by_commit_path"
	if released {
		event = "routing_decision_episode_abort_failed"
	}
	logging.ComponentEvent("extproc", event, map[string]interface{}{
		"detail": err.Error(),
	})
}

// raylineARCRoutesDeadlineAbortClass names why a tracked lookup released its
// episode instead of committing it: the endpoint ran out of the budget it
// publishes, so there is no executed turn to record.
const raylineARCRoutesDeadlineAbortClass = "routes_deadline"

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
	// The privacy headers alone: the builder reads them only on a cell that
	// trusts them, and a consult must be masked as the routed turn would be.
	// So is the snapshot pin, which a consult is decided under as the routed
	// turn would be.
	for _, name := range []string{raylineARCPrivacyModeHeader, raylineARCPrivacyAllowedArmsHeader, config.RaylineARCSnapshotHeader} {
		if value, stated := request.Headers[name]; stated {
			headers[name] = value
		}
	}
	return &RequestContext{
		Headers:              headers,
		RaylineARCRawBody:    request.Body,
		RequestID:            request.DecisionID,
		SourceFormat:         wireFormat,
		SemanticRequest:      &decoded,
		ProtocolEnvelope:     envelope,
		TraceContext:         ctx,
		VSRContextTokenCount: service.consultContextTokenCount(&decoded),
		// The controls live outside the body the key digests.
		RaylineRoutesControlsKey: raylineRoutesRoutingControlsKey(request.Routing),
	}, nil
}

// consultContextTokenCount is the token estimate a routed turn of this body
// would be selected under, so a lookup masks the arms a routed turn would
// mask. Without it a lookup could name an arm the prompt does not fit.
//
// The routed path counts with the recipe's context signal when the recipe
// declares one -- a counter that may be calibrated from observed provider
// usage -- and with the character heuristic otherwise. The consult serves the
// flat profile, which is the default recipe, so that recipe's classifier is
// asked first and the heuristic is the fallback, on the same text and floor.
func (service *raylineARCDecisionService) consultContextTokenCount(request *llmprotocol.Request) int {
	snapshot := extractSemanticRequestSignals(request)
	signalInput := service.router.prepareSignalEvaluationInput(signalConversationHistoryFromSnapshot(snapshot))
	if count, ok := service.router.recipeClassifier(config.DefaultRecipeName).ContextTokenCount(
		signalInput.allMessagesText, snapshot.ContextTokenFloor,
	); ok {
		return count
	}
	return contextTokenEstimate(0, contextTokenText(signalInput), snapshot.ContextTokenFloor)
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
// selection, not episode preparation.
//
// A decision service that cannot represent what the request carries is a
// third outcome: not contention and not breakage. It is wrapped as an invalid
// body so this entrypoint answers 400 like the ExtProc path does, rather than
// reporting a healthy router as unavailable (memex-desktop#7328). It also
// carries errRaylineRoutesRequestShape, so the surface that answers it can say
// which of the two invalid-body causes this was.
//
// Every other selection failure stays a plain error and reads as 503, because
// waiting will not fix it.
func selectionFailureError(err error) error {
	var failure *raylineARCSelectionFailure
	if errors.As(err, &failure) {
		if failure.contended() {
			return fmt.Errorf("%w: %w", routerruntime.ErrRouteDecisionContended, err)
		}
		if selectionShapeRefusalClasses[failure.class] {
			return fmt.Errorf("%w: %w: %w",
				routerruntime.ErrRouteDecisionInvalidBody, errRaylineRoutesRequestShape, err)
		}
		// A private turn with no allowed arm is refused the same way: no
		// model is served in its place, and retrying cannot change that.
		if failure.class == arcFailureNoPrivateArm {
			return fmt.Errorf("%w: %w: %w",
				routerruntime.ErrRouteDecisionUndecidable, routerruntime.ErrRouteDecisionNoPrivateArm, err)
		}
		if failure.class == arcFailurePolicyUndecidable {
			return fmt.Errorf("%w: %w", routerruntime.ErrRouteDecisionUndecidable, err)
		}
		if failure.class == arcFailureRoutesUnknownPriceModel {
			return fmt.Errorf("%w: %w: %w",
				routerruntime.ErrRouteDecisionInvalidBody, errRaylineRoutesUnknownPriceModel, err)
		}
		if routesInconsistentClasses[failure.class] {
			return fmt.Errorf("%w: %w", routerruntime.ErrRouteDecisionInconsistent, err)
		}
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
	if selectionShapeRefusalClasses[failure] {
		return fmt.Errorf("%w: %w: %w",
			routerruntime.ErrRouteDecisionInvalidBody, errRaylineRoutesRequestShape, base)
	}
	// Both entrypoints must answer this the same way. The adapter reads
	// sentinels rather than classes, so without one here a turn naming
	// previous_response_id would be a 400 on the ExtProc path and a 503 on
	// the decision API -- the inconsistency memex-desktop#7484 is about.
	if failure == arcFailureResponsesPreviousID {
		return fmt.Errorf("%w: %w: %w",
			routerruntime.ErrRouteDecisionInvalidBody, errRaylineRoutesStatelessPath, base)
	}
	if failure == arcFailureUnknownSnapshot {
		return fmt.Errorf("%w: %w: %w",
			routerruntime.ErrRouteDecisionInvalidBody, errRaylineRoutesUnknownSnapshot, base)
	}
	return base
}

// routeToolsEncoded is whether the decision's encoding included the
// request's tool definitions. The service's own statement wins
// (encoding.tool_definitions_status: present or absent; unknown says
// nothing); without it, a package (the policy block that decided) whose
// encoding profile encodes request tools (PackageEncodesRequestTools)
// encoded them; otherwise nothing says, and nil keeps the endpoint's
// configured account.
func routeToolsEncoded(status *string, policy *config.RaylineARCPolicyServiceConfig) *bool {
	encoded := func(value bool) *bool { return &value }
	if status != nil {
		switch *status {
		case "present":
			return encoded(true)
		case "absent":
			return encoded(false)
		}
	}
	if policy == nil {
		return nil
	}
	if policy.PackageEncodesRequestTools() {
		return encoded(true)
	}
	return nil
}

// routePolicyService is the decision's policy_service block; nil outside
// the policy-service mode.
func routePolicyService(algorithm *config.AlgorithmConfig) *config.RaylineARCPolicyServiceConfig {
	if algorithm == nil || algorithm.RaylineARC == nil {
		return nil
	}
	return algorithm.RaylineARC.PolicyService
}
