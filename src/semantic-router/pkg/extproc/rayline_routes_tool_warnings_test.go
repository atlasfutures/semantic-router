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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// toolsPolicy is a policy_service block whose package states
// encoding_profile.tool_definitions and harness_shell as given ("" leaves
// harness_shell absent, which excludes the shell).
func toolsPolicy(t *testing.T, toolDefinitions, harnessShell string) *config.RaylineARCPolicyServiceConfig {
	t.Helper()
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	replacement := `"tool_definitions": "` + toolDefinitions + `"`
	if harnessShell != "" {
		replacement += `, "harness_shell": "` + harnessShell + `"`
	}
	body := []byte(strings.Replace(string(raw), `"tool_definitions": "include_recorded"`, replacement, 1))
	path := filepath.Join(t.TempDir(), "package.json")
	if err := os.WriteFile(path, body, 0o600); err != nil { //nolint:gosec // a t.TempDir() path
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	return &config.RaylineARCPolicyServiceConfig{
		PackageAlias: policyTestAlias, PackageSHA256: hex.EncodeToString(sum[:]), PackageManifest: path,
		Bindings: policyTestBindings(), ModelSchedule: config.RaylineARCModelScheduleTaskTurnCompaction,
	}
}

// Whether the encoding included the request's tools: the service's
// encoding.tool_definitions_status when it states one; else the package's
// encoding profile, include_recorded meaning encoded; else unknown.
func TestRouteToolsEncoded(t *testing.T) {
	present, absent, unknown := "present", "absent", "unknown"
	recorded := toolsPolicy(t, "include_recorded", "include")
	recordedV2 := toolsPolicy(t, "include_recorded", "include_v2")
	excludedTools := toolsPolicy(t, "exclude", "include")
	excludedShell := toolsPolicy(t, "include_recorded", "")
	for name, tc := range map[string]struct {
		status *string
		policy *config.RaylineARCPolicyServiceConfig
		want   *bool
	}{
		"present wins over an excluding package":          {&present, excludedTools, ptrTo(true)},
		"absent wins over an including package":           {&absent, recorded, ptrTo(false)},
		"missing, shell include and recorded tools":       {nil, recorded, ptrTo(true)},
		"missing, shell include_v2 and recorded tools":    {nil, recordedV2, ptrTo(true)},
		"missing, tool_definitions exclude: unknown":      {nil, excludedTools, nil},
		"missing, harness shell excluded: unknown":        {nil, excludedShell, nil},
		"unknown says nothing: falls back to the package": {&unknown, recorded, ptrTo(true)},
		"unknown on an excluded shell: unknown":           {&unknown, excludedShell, nil},
		"missing, no policy service: unknown":             {nil, nil, nil},
	} {
		got := routeToolsEncoded(tc.status, tc.policy)
		if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
			t.Errorf("%s: %v, want %v", name, deref(got), deref(tc.want))
		}
	}
}

// The tool warning follows the encoding: encoded drops it, not encoded
// states tools_not_encoded (also over the names-only account), unknown keeps
// the configured account; other warnings stand.
func TestRaylineRoutesToolWarnings(t *testing.T) {
	names := "tool_schemas_not_encoded: tool names influenced this route; their descriptions and schemas were not encoded"
	other := "exploration_untracked: x"
	for name, tc := range map[string]struct {
		in      []string
		encoded *bool
		want    []string
	}{
		"encoded drops tools_not_encoded":   {[]string{raylineRoutesToolsNotEncoded, other}, ptrTo(true), []string{other}},
		"encoded drops the names warning":   {[]string{names}, ptrTo(true), []string{}},
		"absent states tools_not_encoded":   {[]string{names, other}, ptrTo(false), []string{raylineRoutesToolsNotEncoded, other}},
		"absent keeps tools_not_encoded":    {[]string{raylineRoutesToolsNotEncoded}, ptrTo(false), []string{raylineRoutesToolsNotEncoded}},
		"no tools declared, nothing to add": {[]string{other}, ptrTo(false), []string{other}},
		"unknown keeps today's account":     {[]string{names}, nil, []string{names}},
	} {
		if got := raylineRoutesToolWarnings(tc.in, tc.encoded); !slices.Equal(got, tc.want) {
			t.Errorf("%s: %v, want %v", name, got, tc.want)
		}
	}
}

// The status decodes when stated and stays nil when not.
func TestPolicyEncodingToolDefinitionsStatusDecodes(t *testing.T) {
	var stated, unstated raylinearc.PolicyEncoding
	if err := json.Unmarshal([]byte(`{"tool_definitions_status":"present"}`), &stated); err != nil || stated.ToolDefinitionsStatus == nil || *stated.ToolDefinitionsStatus != "present" {
		t.Fatalf("stated: %v %v", err, stated.ToolDefinitionsStatus)
	}
	if err := json.Unmarshal([]byte(`{}`), &unstated); err != nil || unstated.ToolDefinitionsStatus != nil {
		t.Fatalf("unstated: %v %v", err, unstated.ToolDefinitionsStatus)
	}
}

func ptrTo[T any](value T) *T { return &value }

func deref(value *bool) any {
	if value == nil {
		return nil
	}
	return *value
}

// The decide request asks for the tool status only of a service whose
// readiness listing advertises it (an older one refuses the field), and the
// answer reaches the trace.
func TestToolDefinitionsStatusIsAskedOnlyWhenAdvertised(t *testing.T) {
	for _, advertised := range []bool{false, true} {
		fixture := newDecisionsV2Fixture(t, nil)
		capabilities := []string{
			raylinearc.PolicyCapabilityDecisionsV2, raylinearc.PolicyCapabilityTierPoints,
			raylinearc.PolicyCapabilityResponseFields, raylinearc.PolicyCapabilityExploration,
		}
		if advertised {
			capabilities = append(capabilities, raylinearc.PolicyCapabilityToolDefinitionsStatus)
		}
		fixture.fake.advertised = capabilities
		armed := fixture.selector.armed.Load()
		scorer := armed.scorer.(*policyServiceScorer)
		if err := requireRaylineARCPolicyServing(context.Background(), armed.policy,
			fixture.decision.Algorithm.RaylineARC.PolicyService, scorer.setAdvertised); err != nil {
			t.Fatalf("probe: %v", err)
		}
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
		if err != nil {
			t.Fatalf("advertised=%v: %v", advertised, err)
		}
		fields, _ := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["response_fields"].([]any)
		asked := slices.Contains(fields, any(raylinearc.PolicyCapabilityToolDefinitionsStatus))
		status := result.RaylineARC.PolicyToolDefinitionsStatus
		if asked != advertised || (status != nil) != advertised {
			t.Fatalf("advertised=%v: asked %v, status %v", advertised, asked, status)
		}
	}
}
