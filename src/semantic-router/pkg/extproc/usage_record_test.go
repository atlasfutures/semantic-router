package extproc

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
		fields["provider_reported_byok"] != false {
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
