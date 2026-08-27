package conformance

import (
	"strings"
	"testing"
)

// compareCase is one comparator table row.
type compareCase struct {
	name string
	cmp  Comparison
	want Payload
	got  Payload
	// wantPaths are the mismatch paths the comparison must report, in order.
	wantPaths []string
}

func runCompareCases(t *testing.T, tests []compareCase) {
	t.Helper()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Compare(tt.cmp, tt.want, tt.got)
			if err != nil {
				t.Fatalf("Compare() error = %v", err)
			}

			got := make([]string, 0, len(result.Mismatches))
			for _, m := range result.Mismatches {
				got = append(got, m.Path)
			}
			if !equalStrings(got, tt.wantPaths) {
				t.Fatalf("mismatch paths = %v, want %v\n%v", got, tt.wantPaths, result.Err())
			}
			if result.Pass() != (len(tt.wantPaths) == 0) {
				t.Errorf("Pass() = %v, want %v", result.Pass(), len(tt.wantPaths) == 0)
			}
		})
	}
}

func TestCompareExactModes(t *testing.T) {
	runCompareCases(t, []compareCase{
		{
			name: "exact accepts identical bytes and headers",
			cmp:  Comparison{Mode: ModeExact},
			want: Payload{Status: 200, Headers: map[string]string{"Content-Type": "application/json"}, Body: []byte(`{"a":1,"b":2}`)},
			got:  Payload{Status: 200, Headers: map[string]string{"content-type": "application/json"}, Body: []byte(`{"a":1,"b":2}`)},
		},
		{
			name:      "exact rejects reserialized bytes",
			cmp:       Comparison{Mode: ModeExact},
			want:      Payload{Body: []byte(`{"a":1,"b":2}`)},
			got:       Payload{Body: []byte(`{"b":2,"a":1}`)},
			wantPaths: []string{"body"},
		},
		{
			name:      "exact rejects a missing header",
			cmp:       Comparison{Mode: ModeExact},
			want:      Payload{Headers: map[string]string{"Anthropic-Version": "2023-06-01"}, Body: []byte(`{}`)},
			got:       Payload{Body: []byte(`{}`)},
			wantPaths: []string{"Anthropic-Version"},
		},
		{
			name: "exact-except ignores the declared model patch and auth header",
			cmp:  Comparison{Mode: ModeExactExcept, Exclude: []string{"/model", "Authorization"}},
			want: Payload{Headers: map[string]string{"Authorization": "Bearer client"}, Body: []byte(`{"model":"client-alias","stream":false}`)},
			got:  Payload{Headers: map[string]string{"Authorization": "Bearer provider"}, Body: []byte(`{"stream":false,"model":"provider-model"}`)},
		},
		{
			name:      "exact-except still fails on an unknown field the router dropped",
			cmp:       Comparison{Mode: ModeExactExcept, Exclude: []string{"/model"}},
			want:      Payload{Body: []byte(`{"model":"a","x_conformance_extension":{"keep":true}}`)},
			got:       Payload{Body: []byte(`{"model":"b"}`)},
			wantPaths: []string{"/x_conformance_extension"},
		},
		{
			name:      "exact-except reports a nested value change with its pointer",
			cmp:       Comparison{Mode: ModeExactExcept},
			want:      Payload{Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
			got:       Payload{Body: []byte(`{"messages":[{"role":"user","content":"hello"}]}`)},
			wantPaths: []string{"/messages/0/content"},
		},
		{
			name:      "exact-except reports an array length change once",
			cmp:       Comparison{Mode: ModeExactExcept},
			want:      Payload{Body: []byte(`{"tools":[1,2]}`)},
			got:       Payload{Body: []byte(`{"tools":[1]}`)},
			wantPaths: []string{"/tools"},
		},
		{
			name: "exact-except escapes a pointer token containing a slash",
			cmp:  Comparison{Mode: ModeExactExcept},
			want: Payload{Body: []byte(`{"a/b":1}`)},
			got:  Payload{Body: []byte(`{"a/b":1}`)},
		},
	})
}

func TestCompareSemanticAndRejectModes(t *testing.T) {
	runCompareCases(t, []compareCase{
		{
			name: "semantic ignores key order and whitespace",
			cmp:  Comparison{Mode: ModeSemantic},
			want: Payload{Body: []byte("{\n  \"a\": 1,\n  \"b\": [1, 2]\n}")},
			got:  Payload{Body: []byte(`{"b":[1,2],"a":1}`)},
		},
		{
			name: "semantic matches a volatile id by type",
			cmp:  Comparison{Mode: ModeSemantic, Volatile: []string{"/id", "/created"}},
			want: Payload{Body: []byte(`{"id":"resp_fixture","created":1,"model":"m"}`)},
			got:  Payload{Body: []byte(`{"id":"resp_9f2","created":1756300000,"model":"m"}`)},
		},
		{
			name:      "semantic fails when a volatile field disappears",
			cmp:       Comparison{Mode: ModeSemantic, Volatile: []string{"/id"}},
			want:      Payload{Body: []byte(`{"id":"resp_fixture","model":"m"}`)},
			got:       Payload{Body: []byte(`{"model":"m"}`)},
			wantPaths: []string{"/id"},
		},
		{
			name:      "semantic fails when a volatile field changes JSON type",
			cmp:       Comparison{Mode: ModeSemantic, Volatile: []string{"/id"}},
			want:      Payload{Body: []byte(`{"id":"resp_fixture"}`)},
			got:       Payload{Body: []byte(`{"id":42}`)},
			wantPaths: []string{"/id"},
		},
		{
			name:      "semantic rejects a body whose encoding changed",
			cmp:       Comparison{Mode: ModeSemantic},
			want:      Payload{Body: []byte("data: {}\n\n"), Stream: true},
			got:       Payload{Body: []byte(`{}`)},
			wantPaths: []string{"body"},
		},
		{
			name: "reject matches the declared status, header, and error body",
			cmp: Comparison{
				Mode: ModeReject,
				Reject: RejectSpec{
					Status:  400,
					Headers: map[string]string{"Content-Type": "application/json"},
					Body:    map[string]any{"error": map[string]any{"type": "invalid_request_error", "code": "unsupported_tool_result"}},
				},
			},
			got: Payload{
				Status:  400,
				Headers: map[string]string{"Content-Type": "application/json"},
				Body:    []byte(`{"error":{"code":"unsupported_tool_result","type":"invalid_request_error"}}`),
			},
		},
		{
			name:      "reject fails on the wrong status and error code",
			cmp:       Comparison{Mode: ModeReject, Reject: RejectSpec{Status: 400, Body: map[string]any{"error": map[string]any{"code": "unsupported_tool_result"}}}},
			got:       Payload{Status: 502, Body: []byte(`{"error":{"code":"upstream_error"}}`)},
			wantPaths: []string{"status", "/error/code"},
		},
	})
}

func TestCompareErrors(t *testing.T) {
	tests := []struct {
		name    string
		cmp     Comparison
		want    Payload
		got     Payload
		wantErr string
	}{
		{
			name:    "unknown mode",
			cmp:     Comparison{Mode: "almost-exact"},
			wantErr: `unknown comparison mode "almost-exact"`,
		},
		{
			name:    "unparsable observed body",
			cmp:     Comparison{Mode: ModeExactExcept},
			want:    Payload{Body: []byte(`{}`)},
			got:     Payload{Body: []byte(`{`)},
			wantErr: "observed body",
		},
		{
			name:    "exclusion pointer into an array element",
			cmp:     Comparison{Mode: ModeExactExcept, Exclude: []string{"/tools/0"}},
			want:    Payload{Body: []byte(`{"tools":[1]}`)},
			got:     Payload{Body: []byte(`{"tools":[1]}`)},
			wantErr: "addresses an array element",
		},
		{
			name:    "exclusion that is neither a pointer nor a header",
			cmp:     Comparison{Mode: ModeExactExcept, Exclude: []string{"~model"}},
			want:    Payload{Body: []byte(`{"model":"a"}`)},
			got:     Payload{Body: []byte(`{"model":"b"}`)},
			wantErr: "", // treated as a header name, so the body difference still fails
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Compare(tt.cmp, tt.want, tt.got)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Compare() error = %v, want none", err)
				}
				if result.Pass() {
					t.Error("Compare() passed, want a mismatch")
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Compare() error = %v, want it to contain %q", err, tt.wantErr)
			}
		})
	}
}

func TestCaseComparisonResolution(t *testing.T) {
	c := &Case{
		ID: "unit-01",
		Expectation: Expectation{
			ProviderRequest: ModeExactExcept,
			ClientResponse:  ModeSemantic,
			AllowedPatches:  []string{"/model", "Authorization"},
		},
		Fixtures: &Fixtures{Compare: CompareTuning{ExcludeExtra: []string{"/id"}, Volatile: []string{"/created"}}},
	}

	request, err := c.Comparison(BoundaryProviderRequest)
	if err != nil {
		t.Fatalf("Comparison(provider-request) error = %v", err)
	}
	if !equalStrings(request.Exclude, []string{"/model", "Authorization", "/id"}) {
		t.Errorf("provider-request exclusions = %v", request.Exclude)
	}

	response, err := c.Comparison(BoundaryClientResponse)
	if err != nil {
		t.Fatalf("Comparison(client-response) error = %v", err)
	}
	if !equalStrings(response.Volatile, []string{"/created"}) {
		t.Errorf("client-response volatile = %v", response.Volatile)
	}
	if len(response.Exclude) != 0 {
		t.Errorf("semantic boundary resolved exclusions %v, want none", response.Exclude)
	}
}

func TestCaseComparisonRejectsExclusionsUnderExact(t *testing.T) {
	c := &Case{
		ID:          "unit-01",
		Expectation: Expectation{ProviderRequest: ModeExact, ClientResponse: ModeExact, AllowedPatches: []string{"/model"}},
	}

	if _, err := c.Comparison(BoundaryProviderRequest); err == nil || !strings.Contains(err.Error(), "admits no exclusions") {
		t.Fatalf("Comparison() error = %v, want an exact-with-exclusions error", err)
	}
}

func TestCaseComparisonRequiresRejectStatus(t *testing.T) {
	c := &Case{
		ID:          "unit-01",
		Expectation: Expectation{ProviderRequest: ModeReject, ClientResponse: ModeReject},
	}

	if _, err := c.Comparison(BoundaryClientResponse); err == nil || !strings.Contains(err.Error(), "no reject_status") {
		t.Fatalf("Comparison() error = %v, want a missing reject_status error", err)
	}
}

func TestCompareFidelity(t *testing.T) {
	c := &Case{
		ID: "unit-01",
		Expectation: Expectation{Fidelity: map[string]FidelityAction{
			"/model":                 ActionPatched,
			"/tools/0/cache_control": ActionOmitted,
			"/response.completed":    ActionSynthesized,
		}},
	}

	tests := []struct {
		name      string
		got       map[string]FidelityAction
		wantPaths []string
	}{
		{
			name: "identical ledger",
			got: map[string]FidelityAction{
				"/model":                 ActionPatched,
				"/tools/0/cache_control": ActionOmitted,
				"/response.completed":    ActionSynthesized,
			},
		},
		{
			name: "a silent drop is reported as a tier change",
			got: map[string]FidelityAction{
				"/model":                 ActionPatched,
				"/tools/0/cache_control": ActionPreserved,
				"/response.completed":    ActionSynthesized,
			},
			wantPaths: []string{"/tools/0/cache_control"},
		},
		{
			name: "a missing and an undeclared entry are both reported",
			got: map[string]FidelityAction{
				"/model":              ActionPatched,
				"/response.completed": ActionSynthesized,
				"/system":             ActionOmitted,
			},
			wantPaths: []string{"/system", "/tools/0/cache_control"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := c.CompareFidelity(tt.got)
			if err != nil {
				t.Fatalf("CompareFidelity() error = %v", err)
			}

			got := make([]string, 0, len(result.Mismatches))
			for _, m := range result.Mismatches {
				got = append(got, m.Path)
			}
			if !equalStrings(got, tt.wantPaths) {
				t.Fatalf("mismatch paths = %v, want %v\n%v", got, tt.wantPaths, result.Err())
			}
		})
	}
}

func TestCompareFidelityRejectsUnknownAction(t *testing.T) {
	c := &Case{ID: "unit-01", Expectation: Expectation{Fidelity: map[string]FidelityAction{"/model": ActionPatched}}}

	if _, err := c.CompareFidelity(map[string]FidelityAction{"/model": "shortened"}); err == nil ||
		!strings.Contains(err.Error(), "unknown fidelity action") {
		t.Fatalf("CompareFidelity() error = %v, want an unknown-action error", err)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
