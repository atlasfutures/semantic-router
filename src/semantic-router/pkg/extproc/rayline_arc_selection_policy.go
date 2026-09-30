package extproc

import (
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	routermetrics "github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// modelSelectionFailure is the bounded error an authoritative, fail-closed
// algorithm surfaces instead of silently downgrading to the default candidate.
type modelSelectionFailure struct {
	algorithm string
	class     string
}

func (failure *modelSelectionFailure) Error() string {
	return "model selection failed (algorithm=" + failure.algorithm +
		",class=" + failure.class + ")"
}

// selectRaylineARCModel is the whole selection for a rayline_arc decision.
// ARC is authoritative and fail-closed: every failure surfaces a bounded
// error instead of downgrading to the default candidate, and Router Learning,
// the capability pre-filter and the agentic session record never re-select or
// re-index its arms (ARC owns capability exclusion and its own commit seam).
// It owns bounded ordinal telemetry: the generic selection metric uses model
// IDs as Prometheus labels, which would export artifact arm identity and
// create artifact-controlled cardinality.
func (r *OpenAIRouter) selectRaylineARCModel(
	selCtx *selection.SelectionContext,
	algorithm *config.AlgorithmConfig,
	method selection.SelectionMethod,
	ctx *RequestContext,
) (*config.ModelRef, string, error) {
	failedMethod := string(method)
	if failedMethod == "" {
		failedMethod = string(selectionMethodForAuthoritativeAlgorithm(algorithm))
	}
	if firstValidCandidateModelRef(selCtx) == nil {
		return nil, "", selectionFailureForAlgorithm(algorithm, "no_candidate")
	}
	if err := selection.ValidateSelectionContext(selCtx); err != nil {
		logging.Warnf("[ModelSelection] Invalid selection context: %v", err)
		return nil, failedMethod, selectionFailureForAlgorithm(algorithm, "invalid_context")
	}
	selector := r.selectorForDecisionMethod(method, algorithm, ctx)
	if selector == nil {
		logging.Warnf("[ModelSelection] No selector available for method %s", method)
		return nil, failedMethod, selectionFailureForAlgorithm(algorithm, "missing_selector")
	}
	requestCtx := selectionRequestContext(ctx)
	selectionStart := time.Now()
	result, err := selector.Select(requestCtx, selCtx)
	selection.RecordSelectionDuration(method, selector.Tier(), time.Since(selectionStart))
	if requestCtx.Err() != nil {
		return nil, string(method), requestCtx.Err()
	}
	if err != nil {
		logging.Warnf("[ModelSelection] Selection failed: %v", err)
		return nil, failedMethod, selectionFailureForAlgorithm(
			algorithm,
			authoritativeSelectionFailureClass(algorithm, err),
		)
	}
	if err := selection.ValidateSelectionResult(selCtx, result); err != nil {
		logging.Warnf("[ModelSelection] Invalid selection result: %v", err)
		return nil, failedMethod, selectionFailureForAlgorithm(algorithm, "invalid_result")
	}
	selectedModelRef := selectedModelRefFromResult(selCtx, result)
	if selectedModelRef == nil {
		logging.Warnf("[ModelSelection] Selected model %s not found in candidates", result.SelectedModel)
		return nil, failedMethod, selectionFailureForAlgorithm(algorithm, "unknown_model")
	}
	if err := bindRaylineARCDispatchContract(
		selector,
		algorithm,
		result,
		selectedModelRef,
		ctx,
	); err != nil {
		return nil, "", err
	}
	if ctx != nil {
		ctx.VSRRaylineARC = result.RaylineARC
		if ctx.RaylineARCTransaction != nil &&
			result.RaylineARC != nil {
			ctx.RaylineARCTransaction.markSelectionWithAffinity(
				result.RaylineARC.SelectedArm,
				result.RaylineARC.SerializedTokens,
				result.RaylineARC.EncoderReplicaID,
				result.RaylineARC.EncoderVisitedReplicaIDs,
			)
			ctx.RaylineARCTransaction.markPolicyState(result.RaylineARC.PolicyNextState)
		}
	}
	observeRaylineARCSelection(ctx, result.RaylineARC)
	return selectedModelRef, string(method), nil
}

func failClosedSelection(algorithm *config.AlgorithmConfig) bool {
	return authoritativeSelectionKind(algorithm) != "" &&
		algorithm.OnError == "fail_closed"
}

func raylineARCSelection(algorithm *config.AlgorithmConfig) bool {
	return algorithm != nil &&
		algorithm.Type == config.RaylineARCAlgorithmType &&
		algorithm.OnError == "fail_closed"
}

func authoritativeSelectionKind(
	algorithm *config.AlgorithmConfig,
) string {
	if algorithm == nil {
		return ""
	}
	if algorithm.Type == config.RaylineARCAlgorithmType {
		return configRaylineARC
	}
	return ""
}

func selectionFailureForAlgorithm(
	algorithm *config.AlgorithmConfig,
	class string,
) error {
	if !failClosedSelection(algorithm) {
		return nil
	}
	if class == "" {
		class = "selection"
	}
	return &modelSelectionFailure{
		algorithm: authoritativeSelectionKind(algorithm),
		class:     class,
	}
}

func authoritativeSelectionFailureClass(
	algorithm *config.AlgorithmConfig,
	err error,
) string {
	if raylineARCSelection(algorithm) {
		var failure *raylineARCSelectionFailure
		if errors.As(err, &failure) {
			return failure.class
		}
	}
	return "selector"
}

func selectionMethodForAuthoritativeAlgorithm(
	_ *config.AlgorithmConfig,
) selection.SelectionMethod {
	return selection.MethodRaylineARC
}

func observeRaylineARCSelection(
	ctx *RequestContext,
	trace *selection.RaylineARCTrace,
) {
	if ctx == nil || trace == nil {
		return
	}
	previousArm := -1
	if trace.PreviousArm != nil {
		previousArm = *trace.PreviousArm
	}
	switchCost := 0.0
	cacheMissTokens := 0
	if trace.SelectedArm >= 0 &&
		trace.SelectedArm < len(trace.SwitchCostUSD) &&
		trace.SelectedArm < len(trace.CacheMissTokens) {
		switchCost = trace.SwitchCostUSD[trace.SelectedArm]
		cacheMissTokens = trace.CacheMissTokens[trace.SelectedArm]
	}
	var encoderLatency *time.Duration
	if !trace.EncoderLatencyUnknown {
		encoderLatency = &trace.EncoderLatency
	}
	routermetrics.RecordRaylineARCSelection(
		encoderLatency,
		trace.SerializedTokens,
		trace.FullHistoryTokens,
		trace.TruncatedTokens,
		trace.CachedPrefixTokens,
		trace.RetainedPrefixTokens,
		trace.AppendedTokens,
		trace.SessionAction,
		switchCost,
		cacheMissTokens,
	)
	routermetrics.RecordRaylineARCEncoderReplicaRoute(
		trace.EncoderAttempts,
		trace.EncoderFailover,
	)
	fields := map[string]interface{}{
		"request_id":             ctx.RequestID,
		"artifact_id_hash":       trace.ArtifactID,
		"artifact_revision_hash": trace.ArtifactRevision,
		"encoder_revision":       trace.EncoderRevision,
		"episode_id_hash":        trace.EpisodeIDHash,
		"selected_arm":           trace.SelectedArm,
		"previous_arm":           previousArm,
		"raw_scores":             trace.RawScores,
		"adjusted_scores":        trace.AdjustedScores,
		"switch_cost_usd":        trace.SwitchCostUSD,
		"cache_miss_tokens":      trace.CacheMissTokens,
		"stayed":                 trace.Stayed,
		"upgrade_exemptions":     trace.UpgradeExemptions,
		"stay_upgrade_exempted":  trace.StayUpgradeExempted,
		"excluded_arms":          trace.ExcludedArms,
		"serialized_tokens":      trace.SerializedTokens,
		"full_history_tokens":    trace.FullHistoryTokens,
		"truncated_tokens":       trace.TruncatedTokens,
		"cached_prefix_tokens":   trace.CachedPrefixTokens,
		"retained_prefix_tokens": trace.RetainedPrefixTokens,
		"appended_tokens":        trace.AppendedTokens,
		"session_action":         trace.SessionAction,
		"session_revision":       trace.SessionRevision,
		"encoder_replica_index":  trace.EncoderReplicaIndex,
		"encoder_attempts":       trace.EncoderAttempts,
		"encoder_failover":       trace.EncoderFailover,
		"recipe":                 raylineARCRecipeLabel(ctx.Routing.RecipeName()),
	}
	// The policy-service facts exist only when a policy service decided; an
	// artifact-mode selection carries none rather than zeros.
	if trace.PolicyActionID != "" {
		fields["policy_action_id"] = trace.PolicyActionID
		fields["thinking_level"] = trace.ThinkingLevel
		fields["policy_latency_millis"] = trace.PolicyLatency.Milliseconds()
		fields["policy_action_model"] = trace.PolicyActionModel
		fields["worker_provider_model"] = trace.WorkerProviderModel
		addPolicyDispatchEffortFields(fields, ctx)
	}
	if !trace.EncoderLatencyUnknown {
		fields["encoder_latency_millis"] = trace.EncoderLatency.Milliseconds()
	}
	logging.ComponentEvent("extproc", "rayline_arc_selection", fields)
}
