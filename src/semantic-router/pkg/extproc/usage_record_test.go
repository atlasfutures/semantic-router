package extproc

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/usagerecords"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// These tests read the llm_usage line through the reporting entry points
// only, so they compile against any Router that writes that line and say the
// same thing about it.

func usageRecordRouter() *OpenAIRouter {
	cacheWrite := 3.75
	return &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"priced-model": {Pricing: config.ModelPricing{
					Currency: "USD", PromptPer1M: 3, CompletionPer1M: 15,
					CachedInputPer1M: 0.3, CacheWritePer1M: &cacheWrite,
				}},
			},
		},
	}}
}

func usageRecordJSON(t *testing.T, fields map[string]interface{}) string {
	t.Helper()
	encoded, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

// A call nobody priced and nobody counted is null on every unknown member.
// Before the record existed this line said cost 0.0 and prompt_tokens 0: a
// free call with no input.
func TestUsageRecordWritesNullNotZeroForUnknowns(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "rt_unknown-000", RequestModel: "unpriced-model"}

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, responseUsageMetrics{
		invalid: true, invalidReason: "authoritative_usage_missing",
	})

	fields := findLogEvent(t, logs, "llm_usage")
	for _, key := range []string{
		"cost", "currency", "prompt_tokens", "cached_prompt_tokens", "cache_write_tokens",
		"completion_tokens", "total_tokens", "provider_reported_cost",
	} {
		value, present := fields[key]
		if !present || value != nil {
			t.Errorf("%s = %v (present %v), want null", key, value, present)
		}
	}
	encoded := usageRecordJSON(t, fields)
	for _, want := range []string{`"cost":null`, `"prompt_tokens":null`, `"provider_reported_cost":null`} {
		if !strings.Contains(encoded, want) {
			t.Errorf("serialized record lacks %s: %s", want, encoded)
		}
	}
}

// Every row names the rate table it was priced, or not priced, under.
func TestUsageRecordNamesItsPricingSnapshot(t *testing.T) {
	router := usageRecordRouter()
	for _, model := range []string{"priced-model", "unpriced-model"} {
		logs := captureLogs(t)
		ctx := &RequestContext{RequestID: "rt_snapshot-" + model, RequestModel: model}
		router.reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())
		fields := findLogEvent(t, logs, "llm_usage")
		if got, _ := fields["pricing_snapshot"].(string); got != router.Config.PricingSnapshotID() ||
			!strings.HasPrefix(got, config.PricingSnapshotPrefix) {
			t.Fatalf("%s: pricing_snapshot = %v, want %s", model, fields["pricing_snapshot"], router.Config.PricingSnapshotID())
		}
	}
}

// A priced call carries the cache-aware rate-card price and, beside it, what
// the provider said it charged. Neither stands in for the other.
func TestUsageRecordCarriesRateCardAndProviderCost(t *testing.T) {
	logs := captureLogs(t)
	charged, upstream, byok := 0.0123, 0.0119, false
	ctx := &RequestContext{RequestID: "rt_priced-000", RequestModel: "priced-model"}
	usage := responseUsageMetrics{
		promptTokens: 1000, promptTokensReported: true,
		cachedPromptTokens: 400, cachedPromptTokensReported: true,
		cacheWriteTokens: 100, cacheWriteTokensReported: true,
		completionTokens: 200, completionTokensReported: true,
		totalTokens: 1200, totalTokensReported: true,
		providerCost: llmprotocol.ProviderCost{Charged: &charged, UpstreamInference: &upstream, BYOK: &byok},
	}

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, usage)

	fields := findLogEvent(t, logs, "llm_usage")
	// 500 fresh x 3 + 400 cached x 0.3 + 100 written x 3.75 + 200 out x 15, per 1M.
	want := (500*3 + 400*0.3 + 100*3.75 + 200*15) / 1e6
	if got, _ := fields["cost"].(float64); got < want-1e-12 || got > want+1e-12 {
		t.Fatalf("cost = %v, want the cache-aware %v", fields["cost"], want)
	}
	if fields["pricing"] != usagePricingPriced || fields["currency"] != "USD" {
		t.Fatalf("pricing = %v, currency = %v", fields["pricing"], fields["currency"])
	}
	if fields["provider_reported_cost"] != charged ||
		fields["provider_reported_upstream_inference_cost"] != upstream ||
		fields["provider_reported_byok"] != false || fields["provider_reported_currency"] != nil {
		t.Fatalf("provider cost members = %v / %v / %v", fields["provider_reported_cost"],
			fields["provider_reported_upstream_inference_cost"], fields["provider_reported_byok"])
	}
}

// A provider that billed a turn the Router could not count still said what it
// charged.
func TestUsageRecordKeepsProviderCostOnAnUncountedTurn(t *testing.T) {
	logs := captureLogs(t)
	charged := 0.0042
	ctx := &RequestContext{RequestID: "rt_uncounted-000", RequestModel: "priced-model"}

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, responseUsageMetrics{
		invalid: true, providerCost: llmprotocol.ProviderCost{Charged: &charged},
	})

	fields := findLogEvent(t, logs, "llm_usage")
	if fields["provider_reported_cost"] != charged || fields["cost"] != nil || fields["prompt_tokens"] != nil {
		t.Fatalf("an uncounted turn lost its charge or invented counts: %v", fields)
	}
}

// A streamed turn with no usable count still gets its row, so every call the
// gateway made has one to join.
func TestUncountedStreamStillWritesAUsageRecord(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "rt_stream_uncounted", RequestModel: "priced-model", IsStreamingResponse: true}

	usageRecordRouter().reportSemanticStreamingUsage(ctx, time.Second, invalidResponseTerminalUsage("stream_cut"))

	fields := findLogEvent(t, logs, "llm_usage")
	if fields["request_id"] != "rt_stream_uncounted" || fields["prompt_tokens"] != nil || fields["cost"] != nil {
		t.Fatalf("uncounted stream row = %v", fields)
	}
}

// The row names the ARC decision that produced the call, so it stands on its
// own without a join to the selection line.
func TestUsageRecordCarriesARCAttribution(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{
		RequestID: "rt_arc-000", RequestModel: "priced-model",
		DispatchedProviderOrder: []string{"novita"},
		VSRRaylineARC: &selection.RaylineARCTrace{
			EpisodeIDHash: "episode-hash", SelectedArm: 1, TurnIndex: 7,
			ArtifactID: "artifact-hash", ArtifactRevision: "revision-hash",
			PolicyActionID: "action-1", PolicyArmID: "arm-1",
		},
		RaylineARCThinking: &raylineARCThinkingTrace{LevelInForce: "low", ControlInForce: "control-digest"},
		VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{
			RaylineARC: &config.RaylineARCAlgorithmConfig{PolicyService: &config.RaylineARCPolicyServiceConfig{
				PackageSHA256: "package-digest",
			}},
		}},
	}

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())

	fields := findLogEvent(t, logs, "llm_usage")
	want := map[string]interface{}{
		"episode_id_hash": "episode-hash", "turn_index": uint64(7), "selected_arm": int64(1),
		"artifact_id_hash": "artifact-hash", "artifact_revision_hash": "revision-hash",
		"policy_package_sha256": "package-digest", "policy_action_id": "action-1", "policy_arm_id": "arm-1",
		"thinking_level_in_force": "low", "thinking_control_sha256": "control-digest",
	}
	for key, value := range want {
		if fields[key] != value {
			t.Errorf("%s = %#v, want %#v", key, fields[key], value)
		}
	}
	if encoded := usageRecordJSON(t, fields); !strings.Contains(encoded, `"provider_order":["novita"]`) {
		t.Errorf("provider_order not carried: %s", encoded)
	}
}

// A call outside ARC has no ARC attribution; every member says so with null.
func TestUsageRecordWithoutARCLeavesAttributionNull(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "rt_plain-000", RequestModel: "priced-model"}

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())

	fields := findLogEvent(t, logs, "llm_usage")
	for _, key := range []string{"episode_id_hash", "turn_index", "selected_arm", "policy_package_sha256", "thinking_control_sha256"} {
		if value, present := fields[key]; !present || value != nil {
			t.Errorf("%s = %v (present %v), want null", key, value, present)
		}
	}
}

// Two context-recovery calls make one turn. The turn's charge is their sum,
// and unknown when either left its charge unstated.
func TestContextRecoveryChargesBothCalls(t *testing.T) {
	first, second := 0.25, 0.5
	merged := mergeContextRecoveryProviderCost(
		llmprotocol.ProviderCost{Charged: &first}, llmprotocol.ProviderCost{Charged: &second},
	)
	if merged.Charged == nil || *merged.Charged != 0.75 {
		t.Fatalf("charged = %v, want 0.75", merged.Charged)
	}
	partial := mergeContextRecoveryProviderCost(llmprotocol.ProviderCost{Charged: &first}, llmprotocol.ProviderCost{})
	if partial.Charged != nil {
		t.Fatalf("charged = %v, want unknown when one call stated none", *partial.Charged)
	}
}

type capturedUsageRecords struct{ records [][]byte }

func (sink *capturedUsageRecords) Publish(record []byte) {
	sink.records = append(sink.records, append([]byte(nil), record...))
}

// The durable record and the log line are one value: one record per call,
// with the same members, and null where the line is null.
func TestDurableUsageRecordMatchesTheLogLine(t *testing.T) {
	logs := captureLogs(t)
	sink := &capturedUsageRecords{}
	t.Cleanup(usagerecords.Install(sink))
	ctx := &RequestContext{RequestID: "rt_durable-000", RequestModel: "unpriced-model"}

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, responseUsageMetrics{invalid: true})

	if len(sink.records) != 1 {
		t.Fatalf("durable records = %d, want exactly one for the call", len(sink.records))
	}
	var durable map[string]interface{}
	if err := json.Unmarshal(sink.records[0], &durable); err != nil {
		t.Fatal(err)
	}
	line := findLogEvent(t, logs, "llm_usage")
	delete(line, "event")
	if len(durable) != len(line) {
		t.Fatalf("durable record has %d members, log line %d", len(durable), len(line))
	}
	for key, value := range line {
		stored, present := durable[key]
		if !present || (value == nil) != (stored == nil) {
			t.Errorf("%s: log %v, durable %v (present %v)", key, value, stored, present)
		}
	}
	if durable["request_id"] != "rt_durable-000" || durable["record_schema"] != llmUsageRecordSchema ||
		durable["cost"] != nil || durable["pricing_snapshot"] == nil {
		t.Fatalf("durable record = %s", sink.records[0])
	}
}

// A reloaded config is a new config: its rows name its own snapshot, not the
// one cached for the config before it.
func TestPricingSnapshotFollowsAConfigReload(t *testing.T) {
	before := usageRecordRouter()
	after := usageRecordRouter()
	after.Config.ModelConfig["priced-model"] = config.ModelParams{Pricing: config.ModelPricing{PromptPer1M: 4}}
	for range 2 {
		if got := before.pricingSnapshot(); got != before.Config.PricingSnapshotID() {
			t.Fatalf("before reload: %s, want %s", got, before.Config.PricingSnapshotID())
		}
		if got := after.pricingSnapshot(); got != after.Config.PricingSnapshotID() || got == before.pricingSnapshot() {
			t.Fatalf("after reload: %s, want %s", got, after.Config.PricingSnapshotID())
		}
	}
}

// A cut stream whose counts are estimated keeps the charge the upstream did
// state before the cut.
func TestEstimatedCutStreamKeepsTheStatedCharge(t *testing.T) {
	charged := 0.0031
	ctx := attributedStreamContext("rt_cut_charged", "partial answer text")
	ctx.StreamingAborted = true
	ctx.SemanticStreamState.usage = llmprotocol.Usage{
		State: llmprotocol.UsageUnavailable, ProviderCost: llmprotocol.ProviderCost{Charged: &charged},
	}

	usage := truncatedStreamUsage(ctx, invalidResponseTerminalUsage("stream_cut"))

	if !usage.estimated || usage.providerCost.Charged == nil || *usage.providerCost.Charged != charged {
		t.Fatalf("estimated = %v, charge = %v; want an estimate that keeps %v", usage.estimated, usage.providerCost.Charged, charged)
	}
}

// A provider charge of an unexpected shape costs the record its charge, never
// the client its paid completion or the call its usage row.
func TestOddlyTypedProviderChargeStillServesTheCompletion(t *testing.T) {
	for _, fragment := range []string{
		`"cost":"0.0012"`, `"cost":{"amount":1}`, `"is_byok":"false"`,
		`"cost_details":[1]`, `"cost_details":{"upstream_inference_cost":"0.001"}`,
	} {
		t.Run(fragment, func(t *testing.T) {
			logs := captureLogs(t)
			router := usageRecordRouter()
			ctx := &RequestContext{
				RequestID: "rt_odd_charge", RequestModel: "priced-model",
				SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
			}
			body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
				`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
				`"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4,` + fragment + `}}`)

			response := router.handleNonStreamingResponseBody(body, ctx, time.Second)

			if immediate := response.GetImmediateResponse(); immediate != nil {
				t.Fatalf("the completion was refused with %d", immediate.GetStatus().GetCode())
			}
			fields := findLogEvent(t, logs, "llm_usage")
			if fields["prompt_tokens"] != int64(3) || fields["cost"] == nil {
				t.Fatalf("usage row lost its counts or price: %v", fields)
			}
		})
	}
}

// The charge's currency is recorded only where its unit is known: a call
// dispatched to OpenRouter, which states usage.cost in USD credits.
func TestProviderChargeCurrencyComesFromTheDispatch(t *testing.T) {
	charged := 0.002
	for _, tc := range []struct {
		name       string
		openRouter bool
		charge     *float64
		want       interface{}
	}{
		{"openrouter charge", true, &charged, "USD"},
		{"charge of unknown unit", false, &charged, nil},
		{"openrouter without a charge", true, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logs := captureLogs(t)
			ctx := &RequestContext{RequestID: "rt_currency", RequestModel: "priced-model", DispatchedToOpenRouter: tc.openRouter}
			usage := streamingFlagUsage()
			usage.providerCost.Charged = tc.charge
			usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, usage)
			if got := findLogEvent(t, logs, "llm_usage")["provider_reported_currency"]; got != tc.want {
				t.Fatalf("provider_reported_currency = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDispatchRecordsWhetherItWentToOpenRouter(t *testing.T) {
	router := usageRecordRouter()
	for _, tc := range []struct {
		profile *config.ProviderProfile
		want    bool
	}{
		{&config.ProviderProfile{Type: "openrouter"}, true},
		{&config.ProviderProfile{BaseURL: "https://openrouter.ai/api/v1"}, true},
		{&config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com"}, false},
	} {
		ctx := &RequestContext{RequestID: "rt_dispatch", Headers: map[string]string{}}
		router.buildProviderDispatchResponse(&providerDispatch{logicalModel: "priced-model", profile: tc.profile}, ctx)
		if ctx.DispatchedToOpenRouter != tc.want {
			t.Fatalf("%+v: DispatchedToOpenRouter = %v, want %v", tc.profile, ctx.DispatchedToOpenRouter, tc.want)
		}
	}
}

// A request with no resolved model writes no usage row on any path; the
// uncounted-stream row follows the same rule.
func TestUncountedStreamWithoutAModelWritesNoRow(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "rt_stream_no_model", IsStreamingResponse: true}

	usageRecordRouter().reportSemanticStreamingUsage(ctx, time.Second, invalidResponseTerminalUsage("stream_cut"))

	for _, entry := range logs.All() {
		if entry.ContextMap()["event"] == "llm_usage" {
			t.Fatalf("a request with no model wrote a usage row: %v", entry.ContextMap())
		}
	}
}

// A cache hit calls no provider. A cached body that still holds its original
// call's charge must not report that charge again on every hit.
func TestCacheHitReportsNoProviderCharge(t *testing.T) {
	logs := captureLogs(t)
	body := []byte(`{"id":"chatcmpl_cached","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"a cached answer"}}],` +
		`"usage":{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16,"cost":0.0042,"is_byok":false,` +
		`"cost_details":{"upstream_inference_cost":0.0042}}}`)
	ctx := cacheHitContext("rt_cache_charged", false)
	ctx.DispatchedToOpenRouter = true

	usageRecordRouter().reportCacheHitTelemetry(ctx, body, time.Millisecond)

	fields := findLogEvent(t, logs, "llm_usage")
	if fields["from_cache"] != true {
		t.Fatalf("this must be the cache-hit row: %v", fields)
	}
	for _, key := range []string{
		"provider_reported_cost", "provider_reported_upstream_inference_cost",
		"provider_reported_byok", "provider_reported_currency",
	} {
		if fields[key] != nil {
			t.Errorf("%s = %v on a cache hit, want null", key, fields[key])
		}
	}
}
