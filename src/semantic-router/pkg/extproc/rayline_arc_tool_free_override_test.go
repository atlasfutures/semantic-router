package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A request is tool-free when its tools are absent, null or an empty array,
// read off the decide request each ingress codec builds.
func TestPolicyRequestToolFreePerCodec(t *testing.T) {
	input := []json.RawMessage{json.RawMessage(`{"role":"user","content":"hi"}`)}
	for _, tc := range []struct {
		name, format, body string
		toolFree           bool
	}{
		{"anthropic no tools", policyFormatAnthropic, `{"messages":[{"role":"user","content":"hi"}]}`, true},
		{"anthropic empty tools", policyFormatAnthropic, `{"tools":[],"messages":[{"role":"user","content":"hi"}]}`, true},
		{"anthropic null tools", policyFormatAnthropic, `{"tools":null,"messages":[{"role":"user","content":"hi"}]}`, true},
		{"anthropic tools", policyFormatAnthropic, `{"tools":[{"name":"bash","input_schema":{}}],"messages":[{"role":"user","content":"hi"}]}`, false},
		{"chat no tools", policyFormatOpenAI, `{"messages":[{"role":"user","content":"hi"}]}`, true},
		{"chat empty tools", policyFormatOpenAI, `{"tools":[ ],"messages":[{"role":"user","content":"hi"}]}`, true},
		{"chat tools", policyFormatOpenAI, `{"tools":[{"type":"function","function":{"name":"f"}}],"messages":[{"role":"user","content":"hi"}]}`, false},
		{"responses no tools", policyFormatResponses, `{"input":"hi"}`, true},
		{"responses empty tools", policyFormatResponses, `{"input":"hi","tools":[]}`, true},
		{"responses built-in tool", policyFormatResponses, `{"input":"hi","tools":[{"type":"web_search"}]}`, false},
	} {
		request, _, _, err := policyClientRequestOf(&selection.RaylineARCSelectionContext{
			RequestFormat: tc.format, RawRequest: []byte(tc.body), PolicyInput: input,
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if got := policyRequestToolFree(request); got != tc.toolFree {
			t.Errorf("%s: tool-free = %v, want %v", tc.name, got, tc.toolFree)
		}
	}
	if policyRequestToolFree(raylinearc.PolicyClientRequest{Tools: json.RawMessage(`{"name":"f"}`)}) {
		t.Error("tools that are not an array read as none")
	}
}

// newToolFreeFixture is a sage cell at V 1 whose package declares a sage
// tool-free override, against a service that lists the capability or not.
func newToolFreeFixture(t *testing.T, advertised bool) *policySelectorFixture {
	t.Helper()
	fixture := newPolicySelectorFixture(t, "")
	weight := 1.0
	fixture.decision.Algorithm.RaylineARC.PolicyService.DecisionsV2 = &config.RaylineARCDecisionsV2Config{
		Name: "sage", CostWeight: &weight,
	}
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	scorer.toolFreeTier = true
	if advertised {
		scorer.setAdvertised([]string{raylinearc.PolicyCapabilityToolFreeOverride})
	}
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    scorer,
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	return fixture
}

// The service applies the override from the request's tools and its
// operating_point name, so both must reach it on every codec: the client's
// tools as sent ([] for a Responses request that sent none) and the cell's
// tier name.
func TestToolFreeOverrideInputsReachTheServicePerCodec(t *testing.T) {
	items := []json.RawMessage{json.RawMessage(`{"type":"message","role":"user","content":"go"}`)}
	for _, tc := range []struct {
		name, format, body, tools string
	}{
		{"anthropic none", policyFormatAnthropic, `{"model":"auto","messages":[{"role":"user","content":"go"}]}`, ""},
		{"anthropic tools", policyFormatAnthropic,
			`{"model":"auto","tools":[{"name":"bash","input_schema":{}}],"messages":[{"role":"user","content":"go"}]}`,
			`[{"name":"bash","input_schema":{}}]`},
		{"chat empty", policyFormatOpenAI, `{"model":"auto","tools":[],"messages":[{"role":"user","content":"go"}]}`, `[]`},
		{"chat tools", policyFormatOpenAI,
			`{"model":"auto","tools":[{"type":"function","function":{"name":"f"}}],"messages":[{"role":"user","content":"go"}]}`,
			`[{"type":"function","function":{"name":"f"}}]`},
		{"responses none", policyFormatResponses, `{"model":"auto","input":"go"}`, `[]`},
		{"responses tools", policyFormatResponses, `{"model":"auto","input":"go","tools":[{"type":"web_search"}]}`,
			`[{"type":"web_search"}]`},
	} {
		fixture := newToolFreeFixture(t, true)
		state, _ := raylinearc.NewEpisodeState(2)
		arcContext := &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), State: state, RawRequest: []byte(tc.body), RequestFormat: tc.format,
		}
		if tc.format == policyFormatResponses {
			arcContext.PolicyInput = items
		}
		if _, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
			DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs, RaylineARC: arcContext,
		}); err != nil {
			t.Fatalf("%s: select: %v", tc.name, err)
		}
		var sent struct {
			Request struct {
				Tools json.RawMessage `json:"tools"`
			} `json:"request"`
			Selection struct {
				OperatingPoint struct {
					Name string `json:"name"`
				} `json:"operating_point"`
			} `json:"selection"`
			ResponseFields []string `json:"response_fields"`
		}
		if err := json.Unmarshal(fixture.fake.receivedBodies()[0], &sent); err != nil {
			t.Fatal(err)
		}
		if got := string(sent.Request.Tools); got != tc.tools && !(tc.tools == "" && got == "null") {
			t.Errorf("%s: request.tools %s, want %s", tc.name, got, tc.tools)
		}
		if sent.Selection.OperatingPoint.Name != "sage" {
			t.Errorf("%s: operating_point.name %q, want sage", tc.name, sent.Selection.OperatingPoint.Name)
		}
		if !slices.Contains(sent.ResponseFields, raylinearc.PolicyCapabilityToolFreeOverride) {
			t.Errorf("%s: response_fields %v do not ask for the verdict", tc.name, sent.ResponseFields)
		}
	}
}

// llm_usage tool_free_override is the service's verdict: applied, or its
// reason; null when the service stated none or the cell could not ask. VSR
// never infers it from the request.
func TestToolFreeOverrideRecordsTheServiceVerdict(t *testing.T) {
	reason := "carries_tools"
	for _, tc := range []struct {
		name       string
		advertised bool
		verdict    *raylinearc.PolicyToolFreeOverrideVerdict
		want       *string
	}{
		{"applied", true, &raylinearc.PolicyToolFreeOverrideVerdict{Applied: true}, ptrString(toolFreeOverrideApplied)},
		{"not applied", true, &raylinearc.PolicyToolFreeOverrideVerdict{Reason: &reason}, &reason},
		{"no verdict", true, nil, nil},
		{"not advertised", false, &raylinearc.PolicyToolFreeOverrideVerdict{Applied: true}, nil},
	} {
		fixture := newToolFreeFixture(t, tc.advertised)
		fixture.fake.toolFreeVerdict = tc.verdict
		state, _ := raylinearc.NewEpisodeState(2)
		// A tool-free request: the verdict, not VSR's reading, decides.
		result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
		if err != nil {
			t.Fatalf("%s: select: %v", tc.name, err)
		}
		asked := slices.Contains(fixture.fake.received()[0].ResponseFields, raylinearc.PolicyCapabilityToolFreeOverride)
		if asked != tc.advertised {
			t.Fatalf("%s: asked %v", tc.name, asked)
		}
		record := llmUsageRecord{}
		addUsageRecordARCAttribution(&record, &RequestContext{VSRRaylineARC: result.RaylineARC})
		if (record.ToolFreeOverride == nil) != (tc.want == nil) ||
			(tc.want != nil && *record.ToolFreeOverride != *tc.want) {
			t.Fatalf("%s: llm_usage tool_free_override %v, want %v", tc.name, record.ToolFreeOverride, tc.want)
		}
		fields := map[string]interface{}{}
		addPolicyDecisionsV2Fields(fields, result.RaylineARC)
		if tc.advertised && fields["request_tool_free"] != true {
			t.Fatalf("%s: logged request_tool_free %v", tc.name, fields["request_tool_free"])
		}
		if tc.want != nil && fields["tool_free_override"] != *tc.want {
			t.Fatalf("%s: logged %v", tc.name, fields["tool_free_override"])
		}
	}
	record := llmUsageRecord{}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if value, present := decoded["tool_free_override"]; !present || value != nil {
		t.Fatalf("llm_usage tool_free_override = %v (present %v), want null", value, present)
	}
}

func ptrString(value string) *string { return &value }

// A cell asks for the verdict only when its package declares an override
// for its decisions_v2 tier.
func TestPolicyCellHasToolFreeOverrideForItsTier(t *testing.T) {
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	stageOne := `{"stage_one": {"offered": [{"model": "glm-5.3-flash", "effort": null}, {"model": "kimi-k3", "effort": null}],` +
		` "operating_bounds": {"cost_weight": [0.2, 10.0]}, "presets": {"sage": 0.25, "swift": 5},` +
		` "tool_free_overrides": {"sage": {"v_multiplier": 0.5, "excluded_models": ["kimi-k3"]}}},`
	body := []byte(strings.Replace(string(raw), "{", stageOne, 1))
	path := filepath.Join(t.TempDir(), "package.json")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	for name, want := range map[string]bool{"sage": true, "swift": false, "": false} {
		policy := &config.RaylineARCPolicyServiceConfig{
			PackageAlias: policyTestAlias, PackageSHA256: hex.EncodeToString(sum[:]), PackageManifest: path,
			DecisionsV2: &config.RaylineARCDecisionsV2Config{Name: name},
		}
		if got := policyCellHasToolFreeOverride(policy); got != want {
			t.Errorf("tier %q: %v, want %v", name, got, want)
		}
	}
}
