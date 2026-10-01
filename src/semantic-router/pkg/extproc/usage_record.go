package extproc

import (
	"encoding/json"
	"reflect"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/usagerecords"
)

// llmUsageRecordSchema versions the shape below. A consumer reading the
// durable stream can refuse a record it does not understand.
const llmUsageRecordSchema = "vsr.llm_usage.v1"

// Values of llmUsageRecord.Pricing.
const (
	usagePricingPriced               = "priced"
	usagePricingNotConfigured        = "not_configured"
	usagePricingBreakdownUnavailable = "usage_breakdown_unavailable"
	usagePricingCacheHit             = "cache_hit"
)

// llmUsageRecord is one upstream call as the Router accounted for it: the
// llm_usage log line and the durable usage record are this one value.
//
// Every member is always written. A fact nobody stated is null, never zero
// and never absent: a zero cost asserts a free call, a zero count asserts
// tokens nobody counted, and an absent key cannot be told apart from a field
// an older Router did not write.
//
// request_id is the caller's x-request-id. A gateway that sends one id per
// call joins its own usage row to this one on it.
type llmUsageRecord struct {
	Schema     string  `json:"record_schema"`
	RecordedAt string  `json:"recorded_at"`
	RequestID  *string `json:"request_id"`
	Model      *string `json:"model"`

	PromptTokens        *int   `json:"prompt_tokens"`
	CachedPromptTokens  *int   `json:"cached_prompt_tokens"`
	CacheWriteTokens    *int   `json:"cache_write_tokens"`
	CompletionTokens    *int   `json:"completion_tokens"`
	TotalTokens         *int   `json:"total_tokens"`
	ReasoningTokens     *int64 `json:"reasoning_tokens"`
	CompletionLatencyMS *int64 `json:"completion_latency_ms"`
	// UsageSource says whether the counts are the upstream's or the Router's
	// own estimate of a turn it cut.
	UsageSource string `json:"usage_source"`
	// Streaming is what the upstream did, read from the response headers.
	Streaming bool `json:"streaming"`
	// Truncated marks a turn that did not finish: its counts are real and
	// the turn is not a whole one.
	Truncated bool `json:"truncated"`
	// FromCache marks a turn served from the Router's response cache; no
	// upstream was called. CacheHit is the older spelling of the same fact.
	FromCache    bool    `json:"from_cache"`
	CacheHit     bool    `json:"cache_hit"`
	FailureClass *string `json:"failure_class"`

	StopReason       *string `json:"stop_reason"`
	NativeStopReason *string `json:"native_stop_reason"`
	// UpstreamProvider is the provider that served the call, as the upstream
	// named it; ProviderOrder is the route the dispatched body pinned.
	UpstreamProvider *string  `json:"upstream_provider"`
	ProviderOrder    []string `json:"provider_order"`

	// The rate-card price: Cost is cache-aware, from the rates beside it,
	// which come from the table PricingSnapshot names. Cost is null whenever
	// the call could not be priced; Pricing says why.
	Pricing                  string   `json:"pricing"`
	PricingSnapshot          string   `json:"pricing_snapshot"`
	Cost                     *float64 `json:"cost"`
	Currency                 *string  `json:"currency"`
	PricingPromptPer1M       *float64 `json:"pricing_prompt_per_1m"`
	PricingCachedInputPer1M  *float64 `json:"pricing_cached_input_per_1m"`
	PricingCacheWritePer1M   *float64 `json:"pricing_cache_write_per_1m"`
	PricingCompletionPer1M   *float64 `json:"pricing_completion_per_1m"`
	ProviderReportedCost     *float64 `json:"provider_reported_cost"`
	ProviderReportedUpstream *float64 `json:"provider_reported_upstream_inference_cost"`
	ProviderReportedBYOK     *bool    `json:"provider_reported_byok"`

	// The ARC attribution of the call, all null outside ARC.
	EpisodeIDHash         *string `json:"episode_id_hash"`
	TurnIndex             *uint64 `json:"turn_index"`
	SelectedArm           *int    `json:"selected_arm"`
	ArtifactIDHash        *string `json:"artifact_id_hash"`
	ArtifactRevisionHash  *string `json:"artifact_revision_hash"`
	PolicyPackageSHA256   *string `json:"policy_package_sha256"`
	PolicyActionID        *string `json:"policy_action_id"`
	PolicyArmID           *string `json:"policy_arm_id"`
	ThinkingLevelInForce  *string `json:"thinking_level_in_force"`
	ThinkingControlSHA256 *string `json:"thinking_control_sha256"`
}

// newLLMUsageRecord fills what the request context and the counts say. The
// rate-card members are left for the caller, which knows how it priced.
func (r *OpenAIRouter) newLLMUsageRecord(ctx *RequestContext, usage responseUsageMetrics) llmUsageRecord {
	record := llmUsageRecord{
		Schema:          llmUsageRecordSchema,
		RecordedAt:      time.Now().UTC().Format(time.RFC3339Nano),
		UsageSource:     responseUsageSource(usage),
		Pricing:         usagePricingNotConfigured,
		PricingSnapshot: r.pricingSnapshot(),
	}
	record.PromptTokens = reportedCount(usage.promptTokens, usage.promptTokensReported)
	record.CachedPromptTokens = reportedCount(usage.cachedPromptTokens, usage.cachedPromptTokensReported)
	record.CacheWriteTokens = reportedCount(usage.cacheWriteTokens, usage.cacheWriteTokensReported)
	record.CompletionTokens = reportedCount(usage.completionTokens, usage.completionTokensReported)
	if (usage.promptTokensReported && usage.completionTokensReported) || usage.totalTokensReported {
		total := responseUsageTotal(usage)
		record.TotalTokens = &total
	}
	record.ProviderReportedCost = usage.providerCost.Charged
	record.ProviderReportedUpstream = usage.providerCost.UpstreamInference
	record.ProviderReportedBYOK = usage.providerCost.BYOK
	if ctx == nil {
		return record
	}
	record.RequestID = nonEmpty(ctx.RequestID)
	record.Model = nonEmpty(ctx.RequestModel)
	record.Streaming = ctx.IsStreamingResponse
	record.Truncated = ctx.StreamingAborted
	record.FailureClass = nonEmpty(ctx.ResponseFailureClass)
	if len(ctx.DispatchedProviderOrder) > 0 {
		record.ProviderOrder = append([]string(nil), ctx.DispatchedProviderOrder...)
	}
	if response := attributedResponse(ctx); response != nil {
		record.StopReason = nonEmpty(string(response.StopReason))
		record.NativeStopReason = nonEmpty(response.SourceStopReason)
		record.UpstreamProvider = nonEmpty(response.UpstreamProvider)
		record.ReasoningTokens = response.Usage.OutputReasoning.Value
	} else if state := ctx.SemanticStreamState; state != nil {
		// A cut stream has no reconstructed response; what it did say is
		// on the stream state.
		record.UpstreamProvider = nonEmpty(state.provider)
	}
	addUsageRecordARCAttribution(&record, ctx)
	return record
}

func addUsageRecordARCAttribution(record *llmUsageRecord, ctx *RequestContext) {
	if trace := ctx.VSRRaylineARC; trace != nil {
		record.EpisodeIDHash = nonEmpty(trace.EpisodeIDHash)
		turn := trace.TurnIndex
		record.TurnIndex = &turn
		if trace.SelectedArm >= 0 {
			arm := trace.SelectedArm
			record.SelectedArm = &arm
		}
		record.ArtifactIDHash = nonEmpty(trace.ArtifactID)
		record.ArtifactRevisionHash = nonEmpty(trace.ArtifactRevision)
		record.PolicyActionID = nonEmpty(trace.PolicyActionID)
		record.PolicyArmID = nonEmpty(trace.PolicyArmID)
		if decision := ctx.VSRSelectedDecision; trace.PolicyActionID != "" && decision != nil &&
			decision.Algorithm != nil && decision.Algorithm.RaylineARC != nil &&
			decision.Algorithm.RaylineARC.PolicyService != nil {
			record.PolicyPackageSHA256 = nonEmpty(decision.Algorithm.RaylineARC.PolicyService.PackageSHA256)
		}
	}
	if thinking := ctx.RaylineARCThinking; thinking != nil {
		record.ThinkingLevelInForce = nonEmpty(thinking.LevelInForce)
		record.ThinkingControlSHA256 = nonEmpty(thinking.ControlInForce)
	}
}

// pricingSnapshot names the rate table this Router prices with.
func (r *OpenAIRouter) pricingSnapshot() string {
	if r == nil {
		return (*config.RouterConfig)(nil).PricingSnapshotID()
	}
	return r.Config.PricingSnapshotID()
}

// priceUsageRecord prices the call from the rate card, when it can be priced,
// and records the cost metric.
func (r *OpenAIRouter) priceUsageRecord(record *llmUsageRecord, model string, usage responseUsageMetrics) {
	if r == nil || r.Config == nil {
		return
	}
	pricing, ok := r.Config.GetFullModelPricing(model)
	if !ok {
		return
	}
	if !responseUsageHasPricableBreakdown(usage) {
		record.Pricing = usagePricingBreakdownUnavailable
		return
	}
	cost := costForResponseUsage(usage, pricing)
	metrics.RecordModelCost(model, pricing.Currency, cost)
	cacheWriteRate := effectiveCacheWriteRate(pricing)
	record.Pricing = usagePricingPriced
	record.Cost = &cost
	record.Currency = nonEmpty(pricing.Currency)
	record.PricingPromptPer1M = &pricing.PromptPer1M
	record.PricingCachedInputPer1M = &pricing.CachedInputPer1M
	record.PricingCacheWritePer1M = &cacheWriteRate
	record.PricingCompletionPer1M = &pricing.CompletionPer1M
}

// emitLLMUsageRecord writes the record as the llm_usage log line and, when a
// durable sink is configured, as one entry there.
func emitLLMUsageRecord(record llmUsageRecord) {
	logging.LogEvent("llm_usage", record.fields())
	if !usagerecords.Active() {
		return
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "usage_record_encode_failed", map[string]interface{}{
			"request_id": record.RequestID,
			"error":      err.Error(),
		})
		return
	}
	usagerecords.Publish(encoded)
}

// fields is the record keyed by its JSON names, with a null member as nil,
// so the log line and the record are one shape.
func (record llmUsageRecord) fields() map[string]interface{} {
	value := reflect.ValueOf(record)
	recordType := value.Type()
	fields := make(map[string]interface{}, recordType.NumField())
	for index := 0; index < recordType.NumField(); index++ {
		name := strings.Split(recordType.Field(index).Tag.Get("json"), ",")[0]
		member := value.Field(index)
		switch {
		case member.Kind() == reflect.Pointer && member.IsNil(),
			member.Kind() == reflect.Slice && member.IsNil():
			fields[name] = nil
		case member.Kind() == reflect.Pointer:
			fields[name] = member.Elem().Interface()
		default:
			fields[name] = member.Interface()
		}
	}
	return fields
}

func reportedCount(value int, reported bool) *int {
	if !reported {
		return nil
	}
	return &value
}

func nonEmpty(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}
