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

// The service's split is read only when both members are stated and sum to
// token_count; anything else is no statement.
func TestPolicyEncodeSplit(t *testing.T) {
	split := func(token int, retained, appended *int) raylinearc.PolicyEncoding {
		return raylinearc.PolicyEncoding{TokenCount: token, RetainedPrefixTokens: retained, AppendedTokens: appended}
	}
	for name, tc := range map[string]struct {
		encoding           raylinearc.PolicyEncoding
		retained, appended int
		ok                 bool
	}{
		"warm append":      {split(100, ptrTo(90), ptrTo(10)), 90, 10, true},
		"rebuilt":          {split(100, ptrTo(0), ptrTo(100)), 0, 100, true},
		"repeat":           {split(100, ptrTo(100), ptrTo(0)), 100, 0, true},
		"unstated":         {split(100, nil, nil), 0, 0, false},
		"retained only":    {split(100, ptrTo(90), nil), 0, 0, false},
		"appended only":    {split(100, nil, ptrTo(10)), 0, 0, false},
		"sum short":        {split(100, ptrTo(90), ptrTo(5)), 0, 0, false},
		"negative":         {split(100, ptrTo(-1), ptrTo(101)), 0, 0, false},
		"over token count": {split(100, ptrTo(100), ptrTo(1)), 0, 0, false},
	} {
		retained, appended, ok := policyEncodeSplit(tc.encoding)
		if retained != tc.retained || appended != tc.appended || ok != tc.ok {
			t.Errorf("%s: %d/%d %v, want %d/%d %v", name, retained, appended, ok, tc.retained, tc.appended, tc.ok)
		}
	}
}

// The members decode when stated and stay nil when not, so an older
// service's response decodes as before.
func TestPolicyEncodingSessionReuseDecodes(t *testing.T) {
	var stated, unstated raylinearc.PolicyEncoding
	if err := json.Unmarshal([]byte(`{"retained_prefix_tokens":9,"appended_tokens":3}`), &stated); err != nil ||
		stated.RetainedPrefixTokens == nil || *stated.RetainedPrefixTokens != 9 ||
		stated.AppendedTokens == nil || *stated.AppendedTokens != 3 {
		t.Fatalf("stated: %v %+v", err, stated)
	}
	if err := json.Unmarshal([]byte(`{}`), &unstated); err != nil ||
		unstated.RetainedPrefixTokens != nil || unstated.AppendedTokens != nil {
		t.Fatalf("unstated: %v %+v", err, unstated)
	}
}

// A /v1/routes lookup asks a service that lists session_reuse for the split
// and carries it on the trace: a warm append's retained tokens, none for a
// rebuilt session. An older service (not listing it) is never asked, and a
// service that states no split leaves the trace as before; a routed turn
// never asks.
func TestSessionReuseReachesTheTraceOnlyWhenAdvertisedAndStated(t *testing.T) {
	for name, tc := range map[string]struct {
		advertised, lookup bool
		reuse              *fakeSessionReuse
		asked              bool
		retained, appended int
	}{
		"warm append":        {true, true, &fakeSessionReuse{"appended", &[2]int{3000, 412}}, true, 3000, 412},
		"rebuilt":            {true, true, &fakeSessionReuse{"rebuilt", &[2]int{0, 3412}}, true, 0, 3412},
		"created":            {true, true, &fakeSessionReuse{"created", &[2]int{0, 3412}}, true, 0, 3412},
		"split unreported":   {true, true, &fakeSessionReuse{"appended", nil}, true, 0, 0},
		"older service":      {false, true, &fakeSessionReuse{"appended", &[2]int{3000, 412}}, false, 0, 0},
		"routed turn":        {true, false, &fakeSessionReuse{"appended", &[2]int{3000, 412}}, false, 0, 0},
		"older, routed turn": {false, false, &fakeSessionReuse{"appended", &[2]int{3000, 412}}, false, 0, 0},
	} {
		fixture := newDecisionsV2Fixture(t, nil)
		capabilities := []string{
			raylinearc.PolicyCapabilityDecisionsV2, raylinearc.PolicyCapabilityTierPoints,
			raylinearc.PolicyCapabilityResponseFields, raylinearc.PolicyCapabilityExploration,
		}
		if tc.advertised {
			capabilities = append(capabilities, raylinearc.PolicyCapabilitySessionReuse)
		}
		fixture.fake.advertised = capabilities
		fixture.fake.sessionReuse = tc.reuse
		armed := fixture.selector.armed.Load()
		scorer := armed.scorer.(*policyServiceScorer)
		if err := requireRaylineARCPolicyServing(context.Background(), armed.policy,
			fixture.decision.Algorithm.RaylineARC.PolicyService, nil, scorer.setAdvertised); err != nil {
			t.Fatalf("%s: probe: %v", name, err)
		}
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOnWith(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}),
			func(arcContext *selection.RaylineARCSelectionContext) { arcContext.RoutesLookup = tc.lookup })
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		fields, _ := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["response_fields"].([]any)
		asked := slices.Contains(fields, any(raylinearc.PolicyCapabilitySessionReuse))
		trace := result.RaylineARC
		if asked != tc.asked || trace.RetainedPrefixTokens != tc.retained || trace.AppendedTokens != tc.appended ||
			trace.CachedPrefixTokens != 0 {
			t.Errorf("%s: asked %v, retained %d, appended %d, cached %d; want %v, %d, %d, 0", name, asked,
				trace.RetainedPrefixTokens, trace.AppendedTokens, trace.CachedPrefixTokens, tc.asked, tc.retained, tc.appended)
		}
	}
}

// The selection log's split is the service's own when it stated one, and
// today's derivation from the session action otherwise.
func TestPolicyTokenSplitReadsTheStatedSplit(t *testing.T) {
	retained, appended := policyTokenSplit(&selection.RaylineARCTrace{
		SessionAction: "appended", SerializedTokens: 900, RetainedPrefixTokens: 800, AppendedTokens: 100,
	})
	if retained != 800 || appended != 100 {
		t.Fatalf("stated: %v/%v, want 800/100", retained, appended)
	}
	retained, appended = policyTokenSplit(&selection.RaylineARCTrace{
		SessionAction: "appended", SerializedTokens: 900, RetainedPrefixTokens: 800, AppendedTokens: 100,
		TokenCountsUnknown: true,
	})
	if retained != nil || appended != nil {
		t.Fatalf("unknown counts: %v/%v, want nil/nil", retained, appended)
	}
}

// reuseSelector answers like tierStubSelector with the encoder split a
// policy-service selection carries on its trace.
type reuseSelector struct {
	tierStubSelector
	serialized, retained, appended int
	action                         string
}

func (s *reuseSelector) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	result, err := s.tierStubSelector.Select(ctx, selCtx)
	if err == nil {
		result.RaylineARC.SerializedTokens = s.serialized
		result.RaylineARC.RetainedPrefixTokens = s.retained
		result.RaylineARC.AppendedTokens = s.appended
		result.RaylineARC.SessionAction = s.action
	}
	return result, err
}

// The lookup's usage reports the split end to end: a warm append's retained
// tokens are its cache_read_tokens, a rebuilt or unstated split reads 0.
func TestRaylineRoutesLookupReportsTheEncodersReuse(t *testing.T) {
	body := []byte(`{"model":"rayline/arc-dev","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	for name, tc := range map[string]struct {
		selector  reuseSelector
		cacheRead float64
	}{
		"warm append": {reuseSelector{serialized: 3412, retained: 3000, appended: 412, action: "appended"}, 3000},
		"rebuilt":     {reuseSelector{serialized: 3412, retained: 0, appended: 3412, action: "rebuilt"}, 0},
		"unstated":    {reuseSelector{serialized: 3412, action: "appended"}, 0},
	} {
		router, _ := oneCellRouter(t, "main")
		selector := tc.selector
		selector.tierStubSelector = tierStubSelector{stubARCSelector: stubARCSelector{arm: 0, workers: oneCellWorkers()}}
		registry := selection.NewRegistry()
		registry.Register(selection.MethodRaylineARC, &selector)
		router.RecipeModelSelectors[config.DefaultRecipeName] = registry
		router.ModelSelector = registry
		response := router.handleRaylineRoutesAPI(body, routesContext(nil))
		if status := immediateStatusCode(t, response); status != 200 {
			t.Fatalf("%s: status %d", name, status)
		}
		var envelope struct {
			Usage map[string]float64 `json:"usage"`
		}
		if err := json.Unmarshal(response.GetImmediateResponse().Body, &envelope); err != nil {
			t.Fatal(err)
		}
		if envelope.Usage["cache_read_tokens"] != tc.cacheRead || envelope.Usage["encoded_input_tokens"] != 3412 {
			t.Errorf("%s: usage %v, want cache_read_tokens %v of 3412", name, envelope.Usage, tc.cacheRead)
		}
	}
}
