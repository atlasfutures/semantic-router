package extproc

import (
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

// toolsAlgorithm is a policy-service algorithm whose package states
// encoding_profile.tool_definitions as given.
func toolsAlgorithm(t *testing.T, toolDefinitions string) *config.AlgorithmConfig {
	t.Helper()
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(strings.Replace(string(raw), `"tool_definitions": "include_recorded"`, `"tool_definitions": "`+toolDefinitions+`"`, 1))
	path := filepath.Join(t.TempDir(), "package.json")
	if err := os.WriteFile(path, body, 0o600); err != nil { //nolint:gosec // a t.TempDir() path
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	decision := policyTestDecision(config.RaylineARCModelScheduleTaskTurnCompaction)
	decision.Algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{
		PackageAlias: policyTestAlias, PackageSHA256: hex.EncodeToString(sum[:]), PackageManifest: path,
		Bindings: policyTestBindings(), ModelSchedule: config.RaylineARCModelScheduleTaskTurnCompaction,
	}
	return decision.Algorithm
}

// Whether the encoding included the request's tools: the service's
// encoding.tool_definitions_status when it states one; else the package's
// encoding profile, include_recorded meaning encoded; else unknown.
func TestRouteToolsEncoded(t *testing.T) {
	present, absent, other := "present", "absent", "unknown"
	recorded, excluded := toolsAlgorithm(t, "include_recorded"), toolsAlgorithm(t, "exclude")
	for name, tc := range map[string]struct {
		status    *string
		algorithm *config.AlgorithmConfig
		want      *bool
	}{
		"present wins over an excluding package":      {&present, excluded, ptrTo(true)},
		"absent wins over an including package":       {&absent, recorded, ptrTo(false)},
		"missing, package includes recorded tools":    {nil, recorded, ptrTo(true)},
		"missing, package excludes: unknown":          {nil, excluded, nil},
		"an unstated value falls back to the package": {&other, recorded, ptrTo(true)},
		"missing, no policy service: unknown":         {nil, nil, nil},
	} {
		got := routeToolsEncoded(tc.status, tc.algorithm)
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
