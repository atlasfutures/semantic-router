package extproc

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fixtureExplorationTemperature is a hosted cell's exploration in the
// temperature shape (pathfinder#5637): a default T with its calibration, the
// cap, and the caller's T honoured.
func fixtureExplorationTemperature() *config.RaylineARCExplorationTemperatureConfig {
	value := func(v float64) *float64 { return &v }
	return &config.RaylineARCExplorationTemperatureConfig{
		Temperature: value(0.4), TemperatureMax: value(1.5), CallerTemperature: true,
		Salt: "vsr-explore-2026-10",
		Calibration: &config.RaylineARCExplorationBasis{
			R: value(0.05), PackageSHA256: strings.Repeat("a", 64), ReadoutPin: "qwen3.8-27b-arc-encoder@depth8",
		},
	}
}

func newTemperatureFixture(t *testing.T, explore *config.RaylineARCExplorationTemperatureConfig, shape func(*config.RaylineARCPolicyServiceConfig)) *policySelectorFixture {
	t.Helper()
	return newExplorationFixture(t, nil, func(policy *config.RaylineARCPolicyServiceConfig) {
		policy.ExplorationTemperature = explore
		if shape != nil {
			shape(policy)
		}
	})
}

func withHeader(headers map[string]string, name, value string) map[string]string {
	copied := map[string]string{}
	for key, existing := range headers {
		copied[key] = existing
	}
	copied[name] = value
	return copied
}

// raylinearcTraceResult is one turn's sent selection and its trace.
type raylinearcTraceResult struct {
	sent  map[string]any
	trace *selection.RaylineARCTrace
}

func selectTemperatureTurn(t *testing.T, fixture *policySelectorFixture, headers map[string]string) *raylinearcTraceResult {
	t.Helper()
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}), headers)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	return &raylinearcTraceResult{sent: sentSelection(t, fixture), trace: result.RaylineARC}
}

// An eligible hosted turn with no caller T sends the cell's default in the
// temperature shape, exactly the members #5637 names: no tau, no basis.
func TestHostedTurnSendsTheDefaultTemperature(t *testing.T) {
	got := selectTemperatureTurn(t, newTemperatureFixture(t, fixtureExplorationTemperature(), nil), eligibleHeaders())
	want := map[string]any{
		"temperature": 0.4, "salt": "vsr-explore-2026-10", "surface": "hosted", "source": "default",
		"training_data": "allowed",
		"calibration":   map[string]any{"r": 0.05, "package_sha256": strings.Repeat("a", 64), "readout_pin": "qwen3.8-27b-arc-encoder@depth8"},
	}
	if explore := got.sent["exploration"]; !reflect.DeepEqual(explore, want) {
		t.Fatalf("exploration %v, want %v", explore, want)
	}
	if got.trace.PolicyExplorationExcluded != "" || !got.trace.PolicyExplorationSent {
		t.Fatalf("trace excluded %q sent %v", got.trace.PolicyExplorationExcluded, got.trace.PolicyExplorationSent)
	}
}

// The gateway's caller T replaces the default on an eligible turn: source
// caller, no calibration. The service clamps it to the cap. A cell that does
// not honour caller T sends its default whatever the header says.
func TestHostedTurnSendsTheCallerTemperatureWhenHonoured(t *testing.T) {
	headers := withHeader(eligibleHeaders(), raylineARCExplorationTemperatureHeader, "2.5")
	got := selectTemperatureTurn(t, newTemperatureFixture(t, fixtureExplorationTemperature(), nil), headers)
	want := map[string]any{
		"temperature": 2.5, "salt": "vsr-explore-2026-10", "surface": "hosted", "source": "caller", "training_data": "allowed",
	}
	if explore := got.sent["exploration"]; !reflect.DeepEqual(explore, want) {
		t.Fatalf("exploration %v, want %v", explore, want)
	}

	off := fixtureExplorationTemperature()
	off.CallerTemperature = false
	got = selectTemperatureTurn(t, newTemperatureFixture(t, off, nil), headers)
	explore := got.sent["exploration"].(map[string]any)
	if explore["source"] != "default" || explore["temperature"] != 0.4 {
		t.Fatalf("a cell that does not honour caller T sent %v", explore)
	}
}

// Fail closed: privacy mode, ZDR, an opted-out or unstated user, an
// untrusted gateway, an eval cell and a malformed caller T all send no
// exploration -- not the default, and never the caller's T.
func TestHostedExplorationFailsClosed(t *testing.T) {
	caller := withHeader(eligibleHeaders(), raylineARCExplorationTemperatureHeader, "1")
	for name, tc := range map[string]struct {
		headers map[string]string
		shape   func(*config.RaylineARCPolicyServiceConfig)
		want    string
	}{
		"privacy mode":        {withHeader(caller, raylineARCPrivacyModeHeader, "privacy"), nil, explorationExcludedPrivacyMode},
		"zdr":                 {withHeader(caller, raylineARCPrivacyModeHeader, "zdr"), nil, explorationExcludedPrivacyMode},
		"privacy unstated":    {withHeader(caller, raylineARCPrivacyModeHeader, ""), nil, explorationExcludedPrivacyUnknown},
		"training not stated": {withHeader(caller, raylineARCTrainingDataHeader, ""), nil, explorationExcludedTrainingUnknown},
		"opted out":           {withHeader(caller, raylineARCTrainingDataHeader, "not_allowed"), nil, explorationExcludedTrainingOptOut},
		"untrusted gateway": {caller, func(policy *config.RaylineARCPolicyServiceConfig) {
			policy.TrustTurnSignalHeaders = false
		}, explorationExcludedUntrustedHeaders},
		"eval cell": {caller, func(policy *config.RaylineARCPolicyServiceConfig) {
			policy.EvalCell = true
		}, explorationExcludedEvalCell},
		"negative caller T": {withHeader(eligibleHeaders(), raylineARCExplorationTemperatureHeader, "-1"), nil, explorationExcludedCallerTemperature},
		"NaN caller T":      {withHeader(eligibleHeaders(), raylineARCExplorationTemperatureHeader, "NaN"), nil, explorationExcludedCallerTemperature},
		"word caller T":     {withHeader(eligibleHeaders(), raylineARCExplorationTemperatureHeader, "hot"), nil, explorationExcludedCallerTemperature},
	} {
		t.Run(name, func(t *testing.T) {
			got := selectTemperatureTurn(t, newTemperatureFixture(t, fixtureExplorationTemperature(), tc.shape), tc.headers)
			if _, sent := got.sent["exploration"]; sent {
				t.Fatalf("exploration sent: %v", got.sent["exploration"])
			}
			if got.trace.PolicyExplorationExcluded != tc.want {
				t.Fatalf("excluded %q, want %q", got.trace.PolicyExplorationExcluded, tc.want)
			}
			assertNoExplorationUsage(t, usageLine(t, got.trace))
		})
	}
}

// The service's temperature-shape record lands on the selection log without
// the legacy tau members, and llm_usage gains router_temperature and
// router_floor_applied (#5637, "llm_usage gains").
func TestTemperatureRecordReachesTheLogAndUsage(t *testing.T) {
	got := selectTemperatureTurn(t, newTemperatureFixture(t, fixtureExplorationTemperature(), nil), eligibleHeaders())
	record := got.trace.PolicyExploration
	if record == nil || record.Temperature == nil || *record.Temperature != 0.4 {
		t.Fatalf("record %+v", record)
	}
	fields := map[string]interface{}{}
	addPolicyExplorationFields(fields, got.trace)
	logged := fields["exploration"].(map[string]interface{})
	for _, key := range []string{"surface", "source", "training_data", "requested_temperature", "temperature", "normalization", "pi_softmax", "floor"} {
		if _, present := logged[key]; !present {
			t.Errorf("logged record lacks %s", key)
		}
	}
	for _, key := range []string{"requested_tau", "basis"} {
		if _, present := logged[key]; present {
			t.Errorf("logged record carries the legacy %s", key)
		}
	}
	line := usageLine(t, got.trace)
	if line["router_temperature"] != 0.4 || line["router_floor_applied"] != true || line["router_tau"] != nil ||
		line["router_propensity"] == nil {
		t.Fatalf("llm_usage %v %v %v %v", line["router_temperature"], line["router_floor_applied"], line["router_tau"], line["router_propensity"])
	}
}

// A cell whose exploration may run needs a service that advertises the
// temperature shape; one that can never explore needs nothing.
func TestTemperatureExplorationRequiresItsCapability(t *testing.T) {
	policy := &config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true, ExplorationTemperature: fixtureExplorationTemperature()}
	if got := raylineARCPolicyServiceRequirements(policy).Capabilities; !slices.Equal(got, []string{raylinearc.PolicyCapabilityExplorationTemperature}) {
		t.Fatalf("capabilities %v", got)
	}
	policy.EvalCell = true
	if got := raylineARCPolicyServiceRequirements(policy).Capabilities; len(got) != 0 {
		t.Fatalf("an eval cell requires %v", got)
	}
}

// The caller T is a turn signal: never forwarded upstream, and two requests
// that differ in it never coalesce.
func TestCallerTemperatureIsAStrippedTurnSignal(t *testing.T) {
	if !slices.Contains(raylineARCTurnSignalHeadersForRemoval(), raylineARCExplorationTemperatureHeader) {
		t.Fatal("x-rayline-exploration-temperature is forwarded upstream")
	}
	arcConfig := &config.RaylineARCAlgorithmConfig{}
	first := &RequestContext{Headers: map[string]string{raylineARCExplorationTemperatureHeader: "1"}}
	second := &RequestContext{Headers: map[string]string{raylineARCExplorationTemperatureHeader: "2"}}
	if raylineARCTurnInputs(arcConfig, first, nil) == raylineARCTurnInputs(arcConfig, second, nil) {
		t.Fatal("requests with different caller temperatures coalesce")
	}
}

// The wire carries one shape at a time, reads either strictly, and refuses a
// member carrying both.
func TestExplorationRequestWireShapes(t *testing.T) {
	legacy, err := json.Marshal(raylinearc.PolicyExplorationRequest{
		Tau: 0.02, Salt: "s",
		Basis: raylinearc.PolicyExplorationBasis{R: 0.05, PackageSHA256: "p", ReadoutPin: "r"},
	})
	if err != nil || string(legacy) != `{"tau":0.02,"basis":{"r":0.05,"package_sha256":"p","readout_pin":"r"},"salt":"s"}` {
		t.Fatalf("legacy %s %v", legacy, err)
	}
	decisions, err := json.Marshal(raylinearc.PolicyExplorationRequest{Salt: "s", Temperature: &raylinearc.PolicyExplorationTemperature{
		Temperature: 0.7, Surface: "decisions", Source: "caller", TrainingData: "not_allowed",
	}})
	if err != nil || string(decisions) != `{"temperature":0.7,"salt":"s","surface":"decisions","source":"caller","training_data":"not_allowed"}` {
		t.Fatalf("temperature %s %v", decisions, err)
	}
	for _, body := range []string{string(legacy), string(decisions)} {
		var decoded raylinearc.PolicyExplorationRequest
		if err := json.Unmarshal([]byte(body), &decoded); err != nil {
			t.Fatalf("decode %s: %v", body, err)
		}
		if again, _ := json.Marshal(decoded); string(again) != body {
			t.Fatalf("round trip %s -> %s", body, again)
		}
	}
	var both raylinearc.PolicyExplorationRequest
	if err := json.Unmarshal([]byte(`{"tau":0.02,"temperature":1,"salt":"s"}`), &both); err == nil {
		t.Fatal("a member carrying both shapes decoded")
	}
	if err := json.Unmarshal([]byte(`{"temperature":1,"salt":"s","surface":"hosted","source":"default","training_data":"allowed","extra":1}`), &both); err == nil {
		t.Fatal("an unknown temperature-shape member decoded")
	}
}
