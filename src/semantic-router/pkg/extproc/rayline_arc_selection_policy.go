package extproc

import (
	"errors"

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

// completeModelSelection finishes a successful selection. ARC owns bounded
// ordinal telemetry: the generic selection metric uses model IDs as
// Prometheus labels, which would export artifact arm identity and create
// artifact-controlled cardinality, and Router Learning must not post-select.
func (r *OpenAIRouter) completeModelSelection(
	selCtx *selection.SelectionContext,
	algorithm *config.AlgorithmConfig,
	ctx *RequestContext,
	method selection.SelectionMethod,
	result *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
) (*config.ModelRef, string, error) {
	if raylineARCSelection(algorithm) {
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
			}
		}
		observeRaylineARCSelection(ctx, result.RaylineARC)
		return selectedModelRef, string(method), nil
	}
	recordSelCtx, result, selectedModelRef, learningApplied := r.applyRouterLearning(selCtx, result, selectedModelRef, ctx)
	if ctx != nil {
		ctx.VSRSelectionReasoning = selectionReasoningForDiagnostics(method, result.Reasoning)
	}
	recordPromptHelperTelemetry(ctx, result)
	logSelectionResult(method, result, selectedModelRef, learningApplied)
	selection.RecordSelection(
		string(method),
		selectionDecisionStateKey(selCtx),
		selectedModelRef.Model,
		result.Tier,
		result.Score,
	)
	recordAgenticSessionDecision(recordSelCtx, result, selectedModelRef, ctx)
	return selectedModelRef, string(method), nil
}

// handleSelectionFallback either surfaces a bounded failure for fail-closed
// algorithms or records the generic default-candidate fallback.
func (r *OpenAIRouter) handleSelectionFallback(
	algorithm *config.AlgorithmConfig,
	class string,
	method selection.SelectionMethod,
	reason string,
	selCtx *selection.SelectionContext,
	result *selection.SelectionResult,
	defaultCandidate *config.ModelRef,
	selector selection.Selector,
	ctx *RequestContext,
	warning string,
	arguments ...interface{},
) (*config.ModelRef, string, error) {
	if failure := selectionFailureForAlgorithm(
		algorithm,
		class,
	); failure != nil {
		failedMethod := method
		if failedMethod == "" {
			failedMethod = selectionMethodForAuthoritativeAlgorithm(algorithm)
		}
		return nil, string(failedMethod), failure
	}
	logging.Warnf(warning, arguments...)
	selected := r.recordSelectionFallback(
		method,
		reason,
		selCtx,
		result,
		defaultCandidate,
		selector,
		ctx,
	)
	return selected, string(method), nil
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
	routermetrics.RecordRaylineARCSelection(
		trace.EncoderLatency,
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
	logging.ComponentEvent("extproc", "rayline_arc_selection", map[string]interface{}{
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
		"serialized_tokens":      trace.SerializedTokens,
		"full_history_tokens":    trace.FullHistoryTokens,
		"truncated_tokens":       trace.TruncatedTokens,
		"cached_prefix_tokens":   trace.CachedPrefixTokens,
		"retained_prefix_tokens": trace.RetainedPrefixTokens,
		"appended_tokens":        trace.AppendedTokens,
		"session_action":         trace.SessionAction,
		"session_revision":       trace.SessionRevision,
		"encoder_latency_millis": trace.EncoderLatency.Milliseconds(),
		"encoder_replica_index":  trace.EncoderReplicaIndex,
		"encoder_attempts":       trace.EncoderAttempts,
		"encoder_failover":       trace.EncoderFailover,
	})
}
