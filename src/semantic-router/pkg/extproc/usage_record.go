package extproc

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/usagerecords"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// llmUsageRecordSchema versions the shape below. A consumer reading the
// durable stream can refuse a record it does not understand.
//
// v2 added content_sent_before_failure and failure_detail, and bounded
// failure_class to the turn failure vocabulary (a v1 failure_class could
// carry a protocol code, which v2 moves to failure_detail). A v2 record is
// also written for a provider error, which v1 left without a line. Readers
// accept both.
const llmUsageRecordSchema = "vsr.llm_usage.v2"

// openRouterChargeCurrency is the unit of OpenRouter's usage.cost: credits,
// each worth one US dollar.
const openRouterChargeCurrency = "USD"

// Values of llmUsageRecord.Pricing.
const (
	usagePricingPriced               = "priced"
	usagePricingNotConfigured        = "not_configured"
	usagePricingBreakdownUnavailable = "usage_breakdown_unavailable"
	usagePricingCacheHit             = "cache_hit"
	// usagePricingNoUsage is a call the provider failed without stating usage.
	usagePricingNoUsage = "no_usage"
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
	// TTFTMS is request-body start to the first upstream response byte
	// (streamed) or headers (buffered); RoutingLatencyMS is the part of it
	// spent before dispatch, the policy decide included. Their difference is
	// the upstream's own time to first byte. Null when not observed.
	TTFTMS           *int64 `json:"ttft_ms"`
	RoutingLatencyMS *int64 `json:"routing_latency_ms"`
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
	// ContentSentBeforeFailure is set with a turn failure class: whether the
	// client had already received model output when the turn failed.
	ContentSentBeforeFailure *bool `json:"content_sent_before_failure"`
	// FailureDetail is the specific cause behind failure_class, such as the
	// protocol code of a reply the Router could not use.
	FailureDetail *string `json:"failure_detail"`
	// UpstreamAttempts is how many times the cell's provider hop sent the
	// call, null when it did not say. Above 1, Envoy retried it on the same
	// route before any response reached the Router, so the counts and cost
	// here are the final attempt's alone; what the earlier attempts cost, if
	// anything, is unknown, and so is the call's total.
	UpstreamAttempts *int `json:"upstream_attempts"`
	// CostComplete says whether this row covers every attempt of the call:
	// true for a single attempt, false when Envoy retried it (the earlier
	// attempts' charge is unknown), null when the attempt count is.
	CostComplete *bool `json:"cost_complete"`

	StopReason       *string `json:"stop_reason"`
	NativeStopReason *string `json:"native_stop_reason"`
	// ProviderStopReason and the stop_details type and category say why
	// the provider stopped beyond the neutral reason (llmprotocol.StopDetails),
	// so a refused turn names the classifier that refused it.
	ProviderStopReason  *string `json:"provider_stop_reason"`
	StopDetailsType     *string `json:"stop_details_type"`
	StopDetailsCategory *string `json:"stop_details_category"`
	// UpstreamProvider is the provider that served the call, as the upstream
	// named it; ProviderOrder is the route the dispatched body pinned.
	UpstreamProvider *string  `json:"upstream_provider"`
	ProviderOrder    []string `json:"provider_order"`
	// The provider's own ids for the call (upstream_call_ids.go), null when
	// the upstream stated none: UpstreamGenerationID is the x-generation-id
	// header or the response's id (OpenRouter's generation id), and
	// UpstreamRequestID a provider request id header, which
	// UpstreamRequestIDHeader names.
	UpstreamGenerationID    *string `json:"upstream_generation_id"`
	UpstreamRequestID       *string `json:"upstream_request_id"`
	UpstreamRequestIDHeader *string `json:"upstream_request_id_header"`
	// PrivacyMode is the data mode a Rayline ARC turn was served under
	// (standard, privacy or zdr), null outside ARC; a private turn also
	// carries its allowed set's size and digest, never the set
	// (rayline_arc_privacy.go).
	PrivacyMode             *string `json:"privacy_mode"`
	PrivacyAllowedArmsCount *int    `json:"privacy_allowed_arms_count"`
	PrivacyAllowedArmsHash  *string `json:"privacy_allowed_arms_hash"`
	// The provider object's data-use members as the dispatched body carried
	// them, null where it carried none.
	ProviderDataCollection *string `json:"provider_data_collection"`
	ProviderZDR            *bool   `json:"provider_zdr"`
	ProviderAllowFallbacks *bool   `json:"provider_allow_fallbacks"`
	// UpstreamServiceTier is the service tier the upstream says it served
	// the call at (service_tier, or Anthropic's usage.service_tier), in its
	// spelling; null when the response did not say.
	UpstreamServiceTier *string `json:"upstream_service_tier"`
	// RequestedServiceTier is the service_tier the dispatched body carried:
	// the arm's configured tier (providers.models[].service_tier), which
	// replaces a client's, or the client's own forwarded; null when the body
	// carried none.
	RequestedServiceTier *string `json:"requested_service_tier"`
	// FlexResend marks the two calls of a discount-endpoint turn that was
	// re-sent on its standard twin: flex_attempt on the refused or timed-out
	// call, standard_resend on the re-send. Null on every other call.
	FlexResend *string `json:"flex_resend"`
	// EndpointResend marks the calls of an ARC turn whose picked OpenRouter
	// endpoint failed before any byte and was re-sent on the arm's next
	// endpoints (rayline_arc_endpoint_pick.go): endpoint_attempt on each
	// refused call the cell went on from, endpoint_resend on the last call,
	// whose answer the client got whatever its status: a failed last call
	// carries its failure_class like any failed call. Null on every other
	// call.
	EndpointResend *string `json:"endpoint_resend"`
	// ProviderEndpoint is the endpoint tag the call is known to be served
	// by, null when OpenRouter could choose; ProviderEndpointsTried are the
	// turn's endpoints dispatched so far when it picked one of several;
	// SessionKeySource is where the key that picks came from (header,
	// fallback or none) and EndpointSource where the arm's endpoints did
	// (feed, or none when no routing-price feed was in effect). Null outside
	// ARC.
	ProviderEndpoint       *string  `json:"provider_endpoint"`
	ProviderEndpointsTried []string `json:"provider_endpoints_tried"`
	SessionKeySource       *string  `json:"session_key_source"`
	EndpointSource         *string  `json:"endpoint_source"`

	// The rate-card price: Cost is cache-aware, from the rates beside it,
	// which come from the table PricingSnapshot names. Cost is null whenever
	// the call could not be priced; Pricing says why.
	Pricing                 string   `json:"pricing"`
	PricingSnapshot         string   `json:"pricing_snapshot"`
	Cost                    *float64 `json:"cost"`
	Currency                *string  `json:"currency"`
	PricingPromptPer1M      *float64 `json:"pricing_prompt_per_1m"`
	PricingCachedInputPer1M *float64 `json:"pricing_cached_input_per_1m"`
	PricingCacheWritePer1M  *float64 `json:"pricing_cache_write_per_1m"`
	PricingCompletionPer1M  *float64 `json:"pricing_completion_per_1m"`
	ProviderReportedCost    *float64 `json:"provider_reported_cost"`
	// PricingSource is where the rates beside Cost came from: the live
	// routing price feed, or the cell's configured rate card (the table
	// PricingSnapshot names). PricingFeedFallback says why a call the feed
	// could have priced was priced from config, null when the feed priced
	// it or is not configured. The feed identity and the endpoint and tier
	// the call was billed at are set when the feed priced it; PriceTier is
	// standard for config rates and null for an unpriced call.
	PricingSource           string  `json:"pricing_source"`
	PricingFeedFallback     *string `json:"pricing_feed_fallback"`
	PricingFeedStoreVersion *string `json:"pricing_feed_store_version"`
	PricingFeedAsOf         *string `json:"pricing_feed_as_of"`
	PricingFeedSHA256       *string `json:"pricing_feed_sha256"`
	PricingEndpointTag      *string `json:"pricing_endpoint_tag"`
	PriceTier               *string `json:"price_tier"`
	// ProviderReportedCurrency is the unit of the reported charge, set only
	// where the unit is known: OpenRouter states usage.cost in credits of one
	// US dollar each. A charge from any other upstream has no stated unit,
	// and its currency stays null.
	ProviderReportedCurrency *string  `json:"provider_reported_currency"`
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
	// ThinkingRefused names a steer refused on this call (ADR 0129); the
	// level in force is then the one the provider already saw, not the
	// level the policy drew.
	ThinkingRefused *string `json:"thinking_refused"`
	// The Stage-1 exploration record of the call's decision (pathfinder#5436),
	// all null when the decision carried none: the served action's
	// propensity, the tau the decision used, whether the draw left greedy,
	// and the boundary's decision key (on a held turn, the key of the
	// boundary that chose its model), which joins the call to its draw.
	RouterPropensity       *float64 `json:"router_propensity"`
	RouterTau              *float64 `json:"router_tau"`
	RouterExplored         *bool    `json:"router_explored"`
	ExplorationDecisionKey *string  `json:"exploration_decision_key"`
	// DecisionMode is a delta cell's (ADR 0155): whether the call's action
	// was decided on a turn_delta (delta) or a full request (sync). Absent
	// on a sync cell.
	DecisionMode *string `json:"decision_mode,omitempty"`
	// RouterTemperature and RouterFloorApplied are a temperature-shape
	// record's (pathfinder#5637): the softmax temperature the decision used,
	// and whether the new-model floor put mass in its distribution. Null on
	// a legacy tau record and when the decision carried none.
	RouterTemperature  *float64 `json:"router_temperature"`
	RouterFloorApplied *bool    `json:"router_floor_applied"`

	// The served identity of an ARC call (x-vsr-served-model and
	// x-vsr-policy-cost-weight), null outside ARC and when unknown:
	// ServedModel is the provider model id the call was dispatched to, read
	// back off the final provider body, which a pass-through bill resolves
	// its card from; PolicyCostWeight is the stage-one cost weight the
	// service echoed for the decision.
	ServedModel      *string  `json:"served_model"`
	PolicyCostWeight *float64 `json:"policy_cost_weight"`
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
		PricingSource:   usagePricingSourceConfig,
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
	if record.ProviderReportedCost != nil && ctx.DispatchedToOpenRouter {
		record.ProviderReportedCurrency = nonEmpty(openRouterChargeCurrency)
	}
	record.RequestID = nonEmpty(ctx.RequestID)
	record.Model = nonEmpty(ctx.RequestModel)
	record.Streaming = ctx.IsStreamingResponse
	record.Truncated = ctx.StreamingAborted
	record.FailureClass = nonEmpty(ctx.ResponseFailureClass)
	record.ContentSentBeforeFailure = ctx.ContentSentBeforeFailure
	record.FailureDetail = nonEmpty(ctx.ResponseFailureDetail)
	// A re-sent turn (flex twin or another endpoint) is read whole before the
	// response pipeline sees it, and an unreachable re-send is answered with
	// synthetic headers, so its "first byte" is not one: no TTFT is stated.
	resent := (ctx.FlexResend != nil && ctx.FlexResend.served()) || len(ctx.RaylineARCEndpointsTried) > 1
	if ctx.TTFTRecorded && !resent {
		ttft := int64(ctx.TTFTSeconds * 1000)
		record.TTFTMS = &ttft
	}
	if ctx.RoutingLatency > 0 {
		routing := ctx.RoutingLatency.Milliseconds()
		record.RoutingLatencyMS = &routing
	}
	record.UpstreamGenerationID = nonEmpty(ctx.UpstreamCallIDs.generationID)
	record.UpstreamRequestID = nonEmpty(ctx.UpstreamCallIDs.requestID)
	record.UpstreamRequestIDHeader = nonEmpty(ctx.UpstreamCallIDs.requestIDHeader)
	if ctx.UpstreamAttempts > 0 {
		attempts := ctx.UpstreamAttempts
		complete := attempts == 1
		record.UpstreamAttempts = &attempts
		record.CostComplete = &complete
	}
	if len(ctx.DispatchedProviderOrder) > 0 {
		record.ProviderOrder = append([]string(nil), ctx.DispatchedProviderOrder...)
	}
	record.ProviderDataCollection = nonEmpty(ctx.DispatchedProviderPrivacy.dataCollection)
	record.ProviderZDR = ctx.DispatchedProviderPrivacy.zdr
	record.ProviderAllowFallbacks = ctx.DispatchedProviderPrivacy.allowFallbacks
	if ctx.VSRRaylineARC != nil {
		record.PrivacyMode = nonEmpty(privacyModeLabel(ctx.RaylineARCPrivacyMode))
		if ctx.RaylineARCPrivacyMode != "" {
			count := len(ctx.RaylineARCPrivacyAllowedArms)
			record.PrivacyAllowedArmsCount = &count
			record.PrivacyAllowedArmsHash = nonEmpty(privacyAllowedArmsHash(ctx.RaylineARCPrivacyAllowedArms))
		}
	}
	record.RequestedServiceTier = nonEmpty(ctx.DispatchedServiceTier)
	if ctx.FlexResend.served() {
		record.FlexResend = nonEmpty(flexResendStandard)
	}
	record.ProviderEndpoint = nonEmpty(ctx.RaylineARCDispatchedEndpoint)
	record.SessionKeySource = nonEmpty(ctx.RaylineARCSessionKeySource)
	record.EndpointSource = nonEmpty(ctx.RaylineARCEndpointSource)
	if len(ctx.RaylineARCEndpointsTried) > 0 {
		record.ProviderEndpointsTried = append([]string(nil), ctx.RaylineARCEndpointsTried...)
	}
	if len(ctx.RaylineARCEndpointsTried) > 1 {
		// A refused re-send overwrites this with endpoint_attempt.
		record.EndpointResend = nonEmpty(endpointResendServed)
	}
	stop := responseStopTelemetry(ctx)
	record.ProviderStopReason = nonEmpty(stop.providerStop)
	record.StopDetailsType = nonEmpty(stop.detailsType)
	record.StopDetailsCategory = nonEmpty(stop.detailsCategory)
	if response := attributedResponse(ctx); response != nil {
		record.StopReason = nonEmpty(string(response.StopReason))
		record.NativeStopReason = nonEmpty(response.SourceStopReason)
		record.UpstreamProvider = nonEmpty(response.UpstreamProvider)
		record.UpstreamServiceTier = nonEmpty(response.UpstreamServiceTier)
		record.ReasoningTokens = response.Usage.OutputReasoning.Value
	} else if state := ctx.SemanticStreamState; state != nil {
		// A cut stream has no reconstructed response; what it did say is
		// on the stream state.
		record.UpstreamProvider = nonEmpty(state.provider)
		record.UpstreamServiceTier = nonEmpty(state.serviceTier)
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
		if identity, ok := raylineARCServedIdentityOf(ctx); ok {
			record.ServedModel = nonEmpty(identity.model)
			record.PolicyCostWeight = identity.costWeight
		}
		if decision := ctx.VSRSelectedDecision; trace.PolicyActionID != "" && decision != nil &&
			decision.Algorithm != nil && decision.Algorithm.RaylineARC != nil &&
			decision.Algorithm.RaylineARC.PolicyService != nil {
			record.PolicyPackageSHA256 = nonEmpty(decision.Algorithm.RaylineARC.PolicyService.PackageSHA256)
		}
		explored := policyExplorationUsage(trace)
		record.RouterPropensity, record.RouterTau = explored.propensity, explored.tau
		record.RouterExplored, record.ExplorationDecisionKey = explored.explored, explored.decisionKey
		record.RouterTemperature, record.RouterFloorApplied = explored.temperature, explored.floorApplied
		decided := map[string]interface{}{}
		addPolicyDecisionModeFields(decided, trace)
		if mode, ok := decided["decision_mode"].(string); ok {
			record.DecisionMode = &mode
		}
	}
	if thinking := ctx.RaylineARCThinking; thinking != nil {
		record.ThinkingLevelInForce = nonEmpty(thinking.LevelInForce)
		record.ThinkingRefused = nonEmpty(thinking.Refused)
		record.ThinkingControlSHA256 = nonEmpty(thinking.ControlInForce)
	}
}

// pricingSnapshotCache holds the snapshot id of the last config asked about.
// A config is not changed in place once loaded, and a reload installs a new
// one, so the id is computed once per config rather than once per call.
type pricingSnapshotCache struct {
	config *config.RouterConfig
	id     string
}

var lastPricingSnapshot atomic.Pointer[pricingSnapshotCache]

// pricingSnapshot names the rate table this Router prices with.
func (r *OpenAIRouter) pricingSnapshot() string {
	var cfg *config.RouterConfig
	if r != nil {
		cfg = r.Config
	}
	if cached := lastPricingSnapshot.Load(); cached != nil && cached.config == cfg {
		return cached.id
	}
	id := cfg.PricingSnapshotID()
	lastPricingSnapshot.Store(&pricingSnapshotCache{config: cfg, id: id})
	return id
}

// priceUsageRecord prices the call, when it can be priced, and records the
// cost metric: at the live routing price when the call's ARC decision
// carried the feed and the feed can name the call's price, otherwise from
// the rate card.
func (r *OpenAIRouter) priceUsageRecord(record *llmUsageRecord, ctx *RequestContext, model string, usage responseUsageMetrics) {
	// A call the provider stated no usage for has nothing to price.
	if responseUsageSource(usage) == usageSourceUnknown {
		record.Pricing = usagePricingNoUsage
		return
	}
	feedPrice, fallback := feedSettlementPrice(ctx, record.UpstreamProvider, usage)
	record.PricingFeedFallback = nonEmpty(fallback)
	var rates modelpricing.Rates
	if feedPrice != nil {
		rates = feedSettlementRates(feedPrice)
	} else {
		if r == nil || r.Config == nil {
			return
		}
		pricing, ok := r.Config.GetFullModelPricing(model)
		if !ok {
			return
		}
		rates = modelPricingRates(pricing)
	}
	if !responseUsageHasPricableBreakdown(usage) {
		record.Pricing = usagePricingBreakdownUnavailable
		return
	}
	cost := modelpricing.Cost(modelPricingUsage(usage), rates)
	metrics.RecordModelCost(model, rates.Currency, cost)
	cacheWriteRate := rates.EffectiveCacheWritePer1M()
	record.Pricing = usagePricingPriced
	record.Cost = &cost
	record.Currency = nonEmpty(rates.Currency)
	record.PricingPromptPer1M = &rates.PromptPer1M
	record.PricingCachedInputPer1M = &rates.CachedInputPer1M
	record.PricingCacheWritePer1M = &cacheWriteRate
	record.PricingCompletionPer1M = &rates.CompletionPer1M
	if feedPrice != nil {
		recordFeedPrice(record, feedPrice)
	} else {
		record.PriceTier = nonEmpty(raylinearc.PriceTierStandard)
	}
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
		tag := strings.Split(recordType.Field(index).Tag.Get("json"), ",")
		name := tag[0]
		member := value.Field(index)
		switch {
		case slices.Contains(tag[1:], "omitempty") && member.Kind() == reflect.Pointer && member.IsNil():
			// A member that exists only on some cells is absent elsewhere,
			// so their lines are as before it.
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
