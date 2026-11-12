package extproc

import "testing"

func stringPointer(value string) *string { return &value }

// Every llm_usage record says why upstream_provider is what it is. A failure
// that reached no provider is "unserved", with upstream_provider still null,
// so the flash-1017 fallback metric (which counts non-null providers other
// than DeepInfra) never counts a failure as a fallback. A served call the
// upstream did not attribute is "unnamed": the gap a record cannot close.
func TestUsageRecordAlwaysStatesTheUpstreamProvider(t *testing.T) {
	sent := true
	notSent := false
	cases := []struct {
		name          string
		record        llmUsageRecord
		wantState     string
		wantAttempted *string
	}{
		{
			name: "served and named",
			record: llmUsageRecord{UsageSource: "upstream", UpstreamProvider: stringPointer("Anthropic"),
				ProviderOrder: []string{"anthropic", "azure/global"}},
			wantState: upstreamProviderNamed, wantAttempted: stringPointer("anthropic"),
		},
		{
			// The 53 Claude Standard upstream_status_400 turns (10-22..11-11).
			name: "failed before any provider answered",
			record: llmUsageRecord{UsageSource: usageSourceUnknown, FailureClass: stringPointer(turnFailureUpstreamError),
				ContentSentBeforeFailure: &notSent, ProviderOrder: []string{"azure/us", "azure/eu"}},
			wantState: upstreamProviderUnserved, wantAttempted: stringPointer("azure"),
		},
		{
			name: "provider error names the pinned slug",
			record: llmUsageRecord{UsageSource: usageSourceUnknown, FailureClass: stringPointer(turnFailureUpstreamError),
				UpstreamErrorPinnedProviderSlug: stringPointer("novita"), ProviderOrder: []string{"phala", "novita"}},
			wantState: upstreamProviderUnserved, wantAttempted: stringPointer("novita"),
		},
		{
			name: "failure after content was sent is not unserved",
			record: llmUsageRecord{UsageSource: usageSourceUnknown, FailureClass: stringPointer(turnFailureUpstreamError),
				ContentSentBeforeFailure: &sent},
			wantState: upstreamProviderUnnamed,
		},
		{
			name:      "served with usage but no provider named",
			record:    llmUsageRecord{UsageSource: "upstream", ProviderEndpoint: stringPointer("deepinfra/fp8/flex")},
			wantState: upstreamProviderUnnamed, wantAttempted: stringPointer("deepinfra"),
		},
		{
			name:      "open route, nothing known",
			record:    llmUsageRecord{UsageSource: "upstream"},
			wantState: upstreamProviderUnnamed,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			record := tc.record
			record.setUpstreamProviderState()
			if record.UpstreamProviderState != tc.wantState {
				t.Fatalf("upstream_provider_state = %q, want %q", record.UpstreamProviderState, tc.wantState)
			}
			switch {
			case tc.wantAttempted == nil && record.AttemptedProvider != nil:
				t.Fatalf("attempted_provider = %q, want null", *record.AttemptedProvider)
			case tc.wantAttempted != nil && (record.AttemptedProvider == nil || *record.AttemptedProvider != *tc.wantAttempted):
				t.Fatalf("attempted_provider = %v, want %q", record.AttemptedProvider, *tc.wantAttempted)
			}
			if tc.wantState != upstreamProviderNamed && record.UpstreamProvider != nil {
				t.Fatal("upstream_provider must stay null unless the upstream named it")
			}
			fields := record.fields()
			if fields["upstream_provider_state"] != tc.wantState {
				t.Fatalf("log line upstream_provider_state = %v, want %q", fields["upstream_provider_state"], tc.wantState)
			}
		})
	}
}

// The real unusable-response path writes the state on its llm_usage line.
func TestUnusableResponseUsageLineStatesNamedProvider(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	router.handleNonStreamingResponseBody([]byte(emptyChatCompletionBody), emptyCompletionContext(), 0)
	fields := findLogEvent(t, logs, "llm_usage")
	if got := fields["upstream_provider_state"]; got != upstreamProviderNamed {
		t.Fatalf("upstream_provider_state = %v, want named", got)
	}
}
