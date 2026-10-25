package extproc

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func generationPolicy(callerValue bool, fixedWeight *float64) *config.RaylineARCPolicyServiceConfig {
	policy := &config.RaylineARCPolicyServiceConfig{
		TrustTurnSignalHeaders: true,
		GenerationRouting:      &config.RaylineARCGenerationRoutingConfig{CallerValue: callerValue},
	}
	if fixedWeight != nil {
		policy.DecisionsV2 = &config.RaylineARCDecisionsV2Config{CostWeight: fixedWeight}
	}
	return policy
}

func sageWeight() *float64 {
	weight := 1 / 4.5
	return &weight
}

// The contract's generation fixtures: arc-1.0 reads value_usd and
// temperature; prices are a Decisions member; a tier's V cannot move.
func TestGenerationRoutingReadsTheContractFixtures(t *testing.T) {
	valid := readContractFixture(t, "request_valid_generation_arc10.json")
	routing, stripped, refusal := readRaylineGenerationRouting(valid.Body, generationPolicy(true, nil))
	if refusal != nil || routing == nil || *routing.valueUSD != 3 || *routing.costWeight != 1.0/3 || *routing.temperature != 0.2 {
		t.Fatalf("arc-1.0: %+v %+v", routing, refusal)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(stripped, &body); err != nil {
		t.Fatal(err)
	}
	if _, present := body["routing"]; present || body["messages"] == nil {
		t.Fatalf("stripped body %s", stripped)
	}

	prices := readContractFixture(t, "request_invalid_generation_prices.json")
	if _, _, refusal := readRaylineGenerationRouting(prices.Body, generationPolicy(true, nil)); refusal == nil || refusal.code != prices.Expect.Code {
		t.Fatalf("prices on generation: %+v", refusal)
	}
	// The tier fixture names rayline/arc-sage; the gateway sends a tier's
	// turns to the tier's own cell, which knows its V from its config.
	tier := readContractFixture(t, "request_invalid_tier_value_override_generation.json")
	if _, _, refusal := readRaylineGenerationRouting(tier.Body, generationPolicy(false, sageWeight())); refusal == nil || refusal.code != tier.Expect.Code {
		t.Fatalf("tier V override: %+v", refusal)
	}
}

func TestGenerationRoutingTierAndConfigurableRules(t *testing.T) {
	body := func(routing string) []byte {
		return []byte(`{"model":"rayline/arc-dev","messages":[{"role":"user","content":"x"}],"routing":` + routing + `}`)
	}
	for name, tc := range map[string]struct {
		policy *config.RaylineARCPolicyServiceConfig
		body   []byte
		header string
		code   string
	}{
		"tier restates its own V":   {generationPolicy(false, sageWeight()), body(`{"value_usd":4.5}`), "", ""},
		"tier T only":               {generationPolicy(false, sageWeight()), body(`{"temperature":1}`), "", ""},
		"tier moved V":              {generationPolicy(false, sageWeight()), body(`{"value_usd":0.1}`), "", "tier_value_immutable"},
		"cell with no fixed V":      {generationPolicy(false, nil), body(`{"value_usd":1}`), "", "tier_value_immutable"},
		"configurable out of range": {generationPolicy(true, nil), body(`{"value_usd":5.5}`), "", "value_out_of_bounds"},
		"configurable bool V":       {generationPolicy(true, nil), body(`{"value_usd":true}`), "", "value_invalid"},
		"unknown member":            {generationPolicy(true, nil), body(`{"speed":1}`), "", "routing_unknown_member"},
		"exclusive":                 {generationPolicy(true, nil), body(`{"exclusive":false}`), "", "prices_not_supported"},
		"not an object":             {generationPolicy(true, nil), body(`[]`), "", "routing_invalid"},
		"negative T":                {generationPolicy(true, nil), body(`{"temperature":-1}`), "", "temperature_invalid"},
	} {
		t.Run(name, func(t *testing.T) {
			routing, _, refusal := readRaylineGenerationRouting(tc.body, tc.policy)
			switch {
			case tc.code == "" && refusal != nil:
				t.Fatalf("refused %+v", refusal)
			case tc.code != "" && (refusal == nil || refusal.code != tc.code || refusal.status != 400):
				t.Fatalf("refusal %+v, want %s", refusal, tc.code)
			case tc.code == "" && !tc.policy.GenerationRouting.CallerValue && routing.costWeight != nil:
				t.Fatal("a tier cell took a caller cost weight")
			}
		})
	}
	// No block: the body is untouched and nothing is read.
	plain := []byte(`{"model":"rayline/arc-dev","messages":[{"role":"user","content":"x"}]}`)
	if routing, out, refusal := readRaylineGenerationRouting(plain, generationPolicy(true, nil)); routing != nil || refusal != nil || string(out) != string(plain) {
		t.Fatal("a body without routing changed")
	}
}

// At the body phase: the block is removed before anything reads the body,
// the turn's V and T reach the context and its coalescing key, and a refusal
// is a 400 carrying the contract's code in the client's protocol.
func TestApplyGenerationRoutingStripsAndRefuses(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
		Name: "arc",
		Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{
			PolicyService: generationPolicy(true, nil),
		}},
	}}}}}
	ctx := &RequestContext{Headers: map[string]string{}}
	body, refusal := router.applyRaylineGenerationRouting(
		[]byte(`{"model":"rayline/arc-dev","messages":[{"role":"user","content":"x"}],"routing":{"value_usd":2,"temperature":0.5}}`), ctx)
	if refusal != nil || strings.Contains(string(body), "routing") || ctx.RaylineGenerationRouting == nil ||
		*ctx.RaylineGenerationRouting.costWeight != 0.5 || !strings.Contains(ctx.RaylineRoutesControlsKey, "v=2") {
		t.Fatalf("body %s ctx %+v key %q refusal %v", body, ctx.RaylineGenerationRouting, ctx.RaylineRoutesControlsKey, refusal)
	}
	refusedCtx := &RequestContext{Headers: map[string]string{}}
	_, refusal = router.applyRaylineGenerationRouting(
		[]byte(`{"model":"rayline/arc-dev","messages":[{"role":"user","content":"x"}],"routing":{"value_usd":9}}`), refusedCtx)
	if refusal == nil || immediateStatusCode(t, refusal) != 400 || refusedCtx.ImmediateProtocolError == nil ||
		refusedCtx.ImmediateProtocolError.Code != "value_out_of_bounds" {
		t.Fatalf("refusal %v %+v", refusal, refusedCtx.ImmediateProtocolError)
	}
	// A cell without generation_routing reads nothing.
	plainRouter := routesRouter(true)
	untouched := []byte(`{"model":"m","messages":[],"routing":{"value_usd":9}}`)
	if out, refusal := plainRouter.applyRaylineGenerationRouting(untouched, &RequestContext{}); refusal != nil || string(out) != string(untouched) {
		t.Fatal("a cell without generation_routing read the block")
	}
}

// The turn's own cost weight is the decide request's, and a decision echoed
// at another weight is not served.
func TestTurnCostWeightIsSentAndChecked(t *testing.T) {
	fixture := newDecisionsV2Fixture(t, nil)
	weight := 0.5
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectOnWith(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}),
		func(arcContext *selection.RaylineARCSelectionContext) { arcContext.TurnCostWeight = &weight })
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	point := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["selection"].(map[string]any)["operating_point"].(map[string]any)
	if point["cost_weight"] != 0.5 {
		t.Fatalf("operating_point %v", point)
	}

	mismatched := newDecisionsV2Fixture(t, nil)
	other := 2.0
	mismatched.fake.echoCostWeight = &other
	state, _ = raylinearc.NewEpisodeState(2)
	if _, err := mismatched.selectOnWith(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}),
		func(arcContext *selection.RaylineARCSelectionContext) { arcContext.TurnCostWeight = &weight }); err == nil {
		t.Fatal("a decision at another cost weight was served")
	}
}

// routing.temperature is the hosted caller T, only on a trusted cell; what
// the turn applied is published.
func TestGenerationTemperatureIsTheHostedCallerT(t *testing.T) {
	temperature := 0.8
	routing := &raylineGenerationRouting{temperature: &temperature}
	with := raylineGenerationSignalHeaders(map[string]string{"x-rayline-privacy-mode": "false"}, routing, generationPolicy(true, nil))
	if with[raylineARCCallerTemperatureSignal] != "0.8" || with["x-rayline-privacy-mode"] != "false" {
		t.Fatalf("signal headers %v", with)
	}
	untrusted := generationPolicy(true, nil)
	untrusted.TrustTurnSignalHeaders = false
	if got := raylineGenerationSignalHeaders(map[string]string{}, routing, untrusted); len(got) != 0 {
		t.Fatalf("an untrusted cell took the caller T: %v", got)
	}

	// End to end through the hosted rules: an eligible turn on a cell that
	// honours caller T draws at it.
	fixture := newTemperatureFixture(t, fixtureExplorationTemperature(), nil)
	got := selectTemperatureTurn(t, fixture, raylineGenerationSignalHeaders(eligibleHeaders(), routing, generationPolicy(true, nil)))
	explore := got.sent["exploration"].(map[string]any)
	if explore["source"] != "caller" || explore["temperature"] != 0.8 {
		t.Fatalf("exploration %v", explore)
	}

	value := 2.0
	applied := &RequestContext{RaylineGenerationRouting: &raylineGenerationRouting{valueUSD: &value, temperature: &temperature}, VSRRaylineARC: got.trace}
	if in, out := raylineGenerationRoutingStatus(applied); !slices.Equal(in, []string{"value_usd", "temperature"}) || len(out) != 0 {
		t.Fatalf("applied %v unapplied %v", in, out)
	}
	// A privacy turn sends no exploration: temperature is unapplied.
	private := selectTemperatureTurn(t, newTemperatureFixture(t, fixtureExplorationTemperature(), nil),
		raylineGenerationSignalHeaders(withHeader(eligibleHeaders(), raylineARCPrivacyModeHeader, "privacy"), routing, generationPolicy(true, nil)))
	unapplied := &RequestContext{RaylineGenerationRouting: &raylineGenerationRouting{temperature: &temperature}, VSRRaylineARC: private.trace}
	in, out := raylineGenerationRoutingStatus(unapplied)
	if len(in) != 0 || !slices.Equal(out, []string{"temperature"}) {
		t.Fatalf("applied %v unapplied %v", in, out)
	}
	unapplied.VSRCacheHit = false
	set := servedHeaders(unapplied)
	if set[headers.VSRRoutingUnapplied] != "temperature" || set[headers.VSRRoutingApplied] != "" {
		t.Fatalf("headers %v", set)
	}
}

func servedHeaders(ctx *RequestContext) map[string]string {
	set := map[string]string{}
	for _, option := range buildResponseHeaderMutation(ctx, true).SetHeaders {
		set[option.Header.Key] = string(option.Header.RawValue)
	}
	return set
}

// Configurable ARC needs decisions_v2 and the whole public V range; a tier
// cell needs nothing more.
func TestGenerationRoutingReadinessRequirements(t *testing.T) {
	var required raylinearc.PolicyServiceRequirements
	generationRoutingRequirements(&required, generationPolicy(true, nil))
	if !slices.Contains(required.Capabilities, raylinearc.PolicyCapabilityDecisionsV2) ||
		!slices.Contains(required.CostWeights, 0.2) || !slices.Contains(required.CostWeights, 10.0) {
		t.Fatalf("requirements %+v", required)
	}
	var tier raylinearc.PolicyServiceRequirements
	generationRoutingRequirements(&tier, generationPolicy(false, sageWeight()))
	if len(tier.Capabilities) != 0 || len(tier.CostWeights) != 0 {
		t.Fatalf("a tier cell requires %+v", tier)
	}
}
