package extproc

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Whether the encoding included the request's tools: the service's
// encoding.tool_definitions_status when it states present or absent; else
// unknown, whatever the package profile says.
func TestRouteToolsEncoded(t *testing.T) {
	present, absent, unknown := "present", "absent", "unknown"
	for name, tc := range map[string]struct {
		status *string
		want   *bool
	}{
		"present":  {&present, ptrTo(true)},
		"absent":   {&absent, ptrTo(false)},
		"unknown":  {&unknown, nil},
		"unstated": {nil, nil},
	} {
		got := routeToolsEncoded(tc.status)
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

// The decide request asks for the tool status only on a /v1/routes lookup
// and only of a service whose readiness listing advertises it (an older one
// refuses the field), and the answer reaches the trace. A routed turn never
// asks, so a mixed-version pool's older replica never sees the field on it.
func TestToolDefinitionsStatusIsAskedOnlyWhenAdvertised(t *testing.T) {
	for _, tc := range []struct{ advertised, lookup bool }{
		{false, false}, {false, true}, {true, false}, {true, true},
	} {
		fixture := newDecisionsV2Fixture(t, nil)
		capabilities := []string{
			raylinearc.PolicyCapabilityDecisionsV2, raylinearc.PolicyCapabilityTierPoints,
			raylinearc.PolicyCapabilityResponseFields, raylinearc.PolicyCapabilityExploration,
		}
		if tc.advertised {
			capabilities = append(capabilities, raylinearc.PolicyCapabilityToolDefinitionsStatus)
		}
		fixture.fake.advertised = capabilities
		armed := fixture.selector.armed.Load()
		scorer := armed.scorer.(*policyServiceScorer)
		if err := requireRaylineARCPolicyServing(context.Background(), armed.policy,
			fixture.decision.Algorithm.RaylineARC.PolicyService, nil, scorer.setAdvertised); err != nil {
			t.Fatalf("probe: %v", err)
		}
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOnWith(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}),
			func(arcContext *selection.RaylineARCSelectionContext) { arcContext.RoutesLookup = tc.lookup })
		if err != nil {
			t.Fatalf("%+v: %v", tc, err)
		}
		fields, _ := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["response_fields"].([]any)
		asked := slices.Contains(fields, any(raylinearc.PolicyCapabilityToolDefinitionsStatus))
		status := result.RaylineARC.PolicyToolDefinitionsStatus
		want := tc.advertised && tc.lookup
		if asked != want || (status != nil) != want {
			t.Fatalf("%+v: asked %v, status %v, want %v", tc, asked, status, want)
		}
	}
}

// toolStatusSelector answers like tierStubSelector and states the service's
// tool-definitions status on the trace.
type toolStatusSelector struct {
	tierStubSelector
	status    *string
	offerKind string
}

func (s *toolStatusSelector) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	result, err := s.tierStubSelector.Select(ctx, selCtx)
	if err == nil {
		result.RaylineARC.PolicyToolDefinitionsStatus = s.status
		result.RaylineARC.OfferKind = s.offerKind
	}
	return result, err
}

// A lookup's tools warning follows the decision's encoding end to end: the
// service's present drops it, absent states it, and no statement keeps the
// configured account. A held turn encoded nothing, whatever else says.
func TestRaylineRoutesLookupSettlesTheToolWarning(t *testing.T) {
	body := []byte(`{"model":"rayline/arc-dev","max_tokens":64,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"name":"search","input_schema":{"type":"object"}}]}`)
	present, absent := "present", "absent"
	for name, tc := range map[string]struct {
		status    *string
		offerKind string
		warns     bool
	}{
		"present":       {&present, "", false},
		"absent":        {&absent, "", true},
		"unstated":      {nil, "", true},
		"capacity hold": {&present, "capacity_hold", true},
		"media hold":    {&present, "media_hold", true},
	} {
		router, _ := oneCellRouter(t, "main")
		selector := &toolStatusSelector{
			tierStubSelector: tierStubSelector{stubARCSelector: stubARCSelector{arm: 0, workers: oneCellWorkers()}},
			status:           tc.status,
			offerKind:        tc.offerKind,
		}
		registry := selection.NewRegistry()
		registry.Register(selection.MethodRaylineARC, selector)
		router.RecipeModelSelectors[config.DefaultRecipeName] = registry
		router.ModelSelector = registry
		response := router.handleRaylineRoutesAPI(body, routesContext(nil))
		if status := immediateStatusCode(t, response); status != 200 {
			t.Fatalf("%s: status %d", name, status)
		}
		var envelope struct {
			Warnings []string `json:"warnings"`
		}
		if err := json.Unmarshal(response.GetImmediateResponse().Body, &envelope); err != nil {
			t.Fatal(err)
		}
		if got := routesWarns(envelope.Warnings, raylineRoutesToolsNotEncodedPrefix); got != tc.warns {
			t.Errorf("%s: warnings %v, want tools_not_encoded %v", name, envelope.Warnings, tc.warns)
		}
	}
}
