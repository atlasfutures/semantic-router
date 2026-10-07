package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type contextCompressionStats struct {
	appliedMessages int
	appliedBlocks   int
	beforeTokens    int
	afterTokens     int
	omittedChunks   int
	skippedRAG      int
	jsonBlocks      int
}

// applySemanticContextCompression runs compression directly against neutral
// content blocks. The service mutates only the semantic request; codecs remain
// the sole wire-format boundary.
//
//nolint:cyclop,funlen // Compression keeps all configured fail-open and fail-closed outcomes explicit.
func (r *OpenAIRouter) applySemanticContextCompression(
	ctx *RequestContext,
	request *llmprotocol.Request,
) error {
	if ctx == nil || request == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	pluginConfig := ctx.VSRSelectedDecision.GetContextCompressionConfig()
	if pluginConfig == nil || !pluginConfig.Enabled {
		return nil
	}
	applyContextCompressionRequestControls(ctx, pluginConfig)
	if ctx.ContextCompressionSkipReason == "request_bypass" {
		recordContextCompressionStatus(ctx, "bypassed", 0)
		return nil
	}
	start := time.Now()
	service := r.contextCompressionService()
	if service == nil {
		return semanticCompressionFailure(pluginConfig, fmt.Errorf("context compression service is unavailable"))
	}
	policy := contextCompressionPolicy(pluginConfig, ctx.ContextCompressionTargetTokens)
	policy = compressionPolicyForRequest(policy, ctx)
	if revision, err := cache.FingerprintValue(pluginConfig); err == nil {
		ctx.ContextCompressionRevision = revision[:12]
	}
	provenance := contextcompression.Provenance{
		RAGToolCallIDs:       ctx.RAGToolCallIDs,
		MemoryMessageIndexes: ctx.MemoryMessageIndexes,
	}
	requestIR := contextcompression.ParseSemanticRequest(request, provenance)
	model := strings.TrimSpace(ctx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	callContext := ctx.TraceContext
	if callContext == nil {
		callContext = context.Background()
	}
	compressionRequest := contextcompression.Request{
		Model:        model,
		Scope:        r.contextCompressionScope(ctx),
		Request:      requestIR,
		Policy:       policy,
		Capabilities: r.semanticContextCompressionCapabilities(ctx, request),
		TokenCounter: r.contextCompressionTokenCounter(ctx),
		Scorer:       r.contextCompressionScorer(callContext, pluginConfig),
		Recovery:     r.contextCompressionRecoveryStore(pluginConfig),
		Provenance:   provenance,
	}
	result := service.Apply(callContext, compressionRequest)
	if result.Failure != nil {
		recordContextCompressionStatus(ctx, "compression_failed", time.Since(start).Seconds())
		return semanticCompressionFailure(pluginConfig, result.Failure)
	}
	r.recordCompressionPlan(ctx, result)
	if !result.Applied {
		status := string(result.Plan.SkipReason)
		if status == "" {
			status = "not_applied"
		}
		recordContextCompressionStatus(ctx, status, time.Since(start).Seconds())
		return nil
	}
	if len(result.RecoveryKeys) > 0 {
		if err := injectSemanticContextRecoveryTool(request, result.RecoveryKeys); err != nil {
			return semanticCompressionFailure(pluginConfig, err)
		}
	}
	ctx.ContextCompressionRecoveryKeys = append(ctx.ContextCompressionRecoveryKeys[:0], result.RecoveryKeys...)
	recordContextCompressionApplied(ctx, contextCompressionStats{
		appliedMessages: result.MessagesCompressed,
		appliedBlocks:   result.BlocksCompressed,
		beforeTokens:    result.TokensBefore,
		afterTokens:     result.TokensAfter,
		omittedChunks:   result.OmittedChunks,
		jsonBlocks:      result.JSONBlocks,
	}, start)
	request.Generation++
	return nil
}

func semanticCompressionFailure(plugin *config.ContextCompressionPluginConfig, err error) error {
	if plugin.EffectiveFailureMode() == config.ContextCompressionFailureClosed {
		return err
	}
	return nil
}

func (r *OpenAIRouter) semanticContextCompressionCapabilities(
	ctx *RequestContext,
	request *llmprotocol.Request,
) contextcompression.ModelContextCapabilities {
	model := strings.TrimSpace(ctx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	contextWindow := 0
	if r != nil && r.Config != nil {
		if params, ok := r.Config.ModelConfig[model]; ok {
			contextWindow = params.ContextWindowSize
		}
	}
	capabilities := contextcompression.ModelContextCapabilities{ContextWindow: contextWindow}
	if request != nil {
		capabilities.RequestedOutput = int(r.compressionOutputReserve(model, ctx, request))
	}
	return capabilities
}

// compressionOutputReserve is the output limit the request will be dispatched
// with, which compression keeps free in the context window. A request with no
// limit of its own gets the one dispatch will set (planDispatchOutputBound),
// so a prompt near the window is compressed before the card value or the
// fallback is added, not refused upstream after.
//
// Compression runs before dispatch, but the target format does not depend on
// anything dispatch decides: dispatchTargetFormat is the same pure function of
// the selected model and the client's format, and the decision's cap is
// already known. The thinking budget dispatch will plan with is not on the
// request yet when a policy action or a v5 control sets it, so
// pendingMessagesThinkingBudget derives it from the selection already on the
// context.
func (r *OpenAIRouter) compressionOutputReserve(
	model string,
	ctx *RequestContext,
	request *llmprotocol.Request,
) int64 {
	if request.Sampling.MaxOutputTokens != nil {
		return *request.Sampling.MaxOutputTokens
	}
	if r == nil || r.Config == nil || model == "" {
		return 0
	}
	format, err := r.dispatchTargetFormat(model, ctx.SourceFormat)
	if err != nil {
		return 0
	}
	return r.planDispatchOutputBound(
		model, format, decisionMaxTokensLimit(ctx), pendingMessagesThinkingBudget(request, format, ctx),
		int64(r.Config.GetModelContextWindowSize(model)), int64(ctx.VSRContextTokenCount),
	).maxTokens
}

func injectSemanticContextRecoveryTool(request *llmprotocol.Request, keys []string) error {
	for _, tool := range request.Tools {
		if tool.Name == contextcompression.RetrieveToolName {
			return fmt.Errorf("request defines reserved tool %q", contextcompression.RetrieveToolName)
		}
	}
	schema, err := json.Marshal(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"key": map[string]interface{}{"type": "string", "enum": keys},
		},
		"required": []string{"key"},
	})
	if err != nil {
		return err
	}
	request.Tools = append(request.Tools, llmprotocol.Tool{
		Name:        contextcompression.RetrieveToolName,
		Description: "Retrieve original context omitted by vLLM Semantic Router compression.",
		InputSchema: schema,
	})
	return nil
}

func (stats contextCompressionStats) format() string {
	switch stats.jsonBlocks {
	case 0:
		return "text"
	case stats.appliedBlocks:
		return "json"
	default:
		return "mixed"
	}
}

func (r *OpenAIRouter) recordCompressionPlan(
	ctx *RequestContext,
	result contextcompression.ServiceResult,
) {
	ctx.ContextCompressionStrategy = "extractive"
	if len(result.RecoveryKeys) > 0 {
		ctx.ContextCompressionStrategy = "recoverable"
	}
	ctx.ContextCompressionBudgetMode = "request_and_item"
	ctx.ContextCompressionTokenSource = result.Plan.TokenCounterSource
	ctx.ContextCompressionTrigger = result.Plan.TriggerReason
	ctx.ContextCompressionQuality = result.Plan.Quality
	ctx.ContextCompressionFallback = result.Plan.FallbackReason
	status := "applied"
	if !result.Applied {
		status = string(result.Plan.SkipReason)
	}
	metrics.RecordContextCompressionPlan(
		requestDecisionStateKey(ctx),
		ctx.ContextCompressionStrategy,
		"mixed",
		status,
		ctx.ContextCompressionTokenSource,
		result.TokensBefore,
		result.TokensAfter,
		len(result.RecoveryKeys),
		result.Plan.Quality,
		result.Plan.FallbackReason,
	)
	savedTokens := max(0, result.TokensBefore-result.TokensAfter)
	r.recordContextCompressionCostSavings(ctx, savedTokens)
}

func (r *OpenAIRouter) recordContextCompressionCostSavings(
	ctx *RequestContext,
	savedTokens int,
) {
	if r == nil || r.Config == nil || savedTokens <= 0 {
		return
	}
	model := strings.TrimSpace(ctx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok || params.Pricing.PromptPer1M <= 0 {
		return
	}
	ctx.ContextCompressionCostSaved = float64(savedTokens) *
		params.Pricing.PromptPer1M / 1_000_000
	metrics.RecordContextCompressionCostSavings(
		model,
		params.Pricing.Currency,
		ctx.ContextCompressionCostSaved,
	)
	if service := r.contextCompressionService(); service != nil {
		service.RecordEstimatedCostSavings(ctx.ContextCompressionCostSaved)
	}
}

func recordContextCompressionStatus(
	ctx *RequestContext,
	status string,
	latencySeconds float64,
) {
	ctx.ContextCompressionSkipReason = status
	metrics.RecordPluginExecution(
		"context_compression",
		requestDecisionStateKey(ctx),
		status,
		latencySeconds,
	)
}

func recordContextCompressionApplied(
	ctx *RequestContext,
	stats contextCompressionStats,
	start time.Time,
) {
	ctx.ContextCompressionApplied = true
	ctx.ContextCompressionBefore = stats.beforeTokens
	ctx.ContextCompressionAfter = stats.afterTokens
	ctx.ContextCompressionMessages = stats.appliedMessages
	ctx.ContextCompressionFormat = stats.format()
	ctx.ContextCompressionOmitted = stats.omittedChunks
	ctx.ContextCompressionSkipReason = ""
	metrics.RecordContextCompression(
		stats.beforeTokens,
		stats.afterTokens,
		ctx.ContextCompressionFormat,
	)
	metrics.RecordPluginExecution(
		"context_compression",
		requestDecisionStateKey(ctx),
		"applied",
		time.Since(start).Seconds(),
	)
	logging.ComponentDebugEvent("extproc", "context_compression_applied", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"messages":       stats.appliedMessages,
		"blocks":         stats.appliedBlocks,
		"tokens_before":  stats.beforeTokens,
		"tokens_after":   stats.afterTokens,
		"tokens_saved":   stats.beforeTokens - stats.afterTokens,
		"omitted_chunks": stats.omittedChunks,
		"json_blocks":    stats.jsonBlocks,
		"rag_skipped":    stats.skippedRAG,
		"latency_millis": time.Since(start).Milliseconds(),
	})
}
