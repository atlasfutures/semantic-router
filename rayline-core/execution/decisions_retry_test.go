package execution

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRetryAdviceReadsOnlyDeltaSecondsAndExactBooleans(t *testing.T) {
	seconds := func(n int) *time.Duration { d := time.Duration(n) * time.Second; return &d }
	yes, no := true, false
	for _, tc := range []struct {
		name, retryAfter, shouldRetry string
		wantDelay                     *time.Duration
		wantRetry                     *bool
	}{
		{name: "absent"},
		{name: "delta seconds", retryAfter: "7", wantDelay: seconds(7)},
		{name: "padded", retryAfter: "  12 ", wantDelay: seconds(12)},
		{name: "zero", retryAfter: "0", wantDelay: seconds(0)},
		{name: "at bound", retryAfter: "300", wantDelay: seconds(300)},
		{name: "clamped", retryAfter: "3600", wantDelay: seconds(300)},
		{name: "overflow clamped", retryAfter: "99999999999999999999999", wantDelay: seconds(300)},
		{name: "http date ignored", retryAfter: "Wed, 21 Oct 2026 07:28:00 GMT"},
		{name: "negative ignored", retryAfter: "-5"},
		{name: "fractional ignored", retryAfter: "1.5"},
		{name: "signed ignored", retryAfter: "+5"},
		{name: "should retry true", shouldRetry: "true", wantRetry: &yes},
		{name: "should retry false", shouldRetry: "false", wantRetry: &no},
		{name: "should retry padded", shouldRetry: " false ", wantRetry: &no},
		{name: "should retry other case ignored", shouldRetry: "True"},
		{name: "should retry other value ignored", shouldRetry: "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			header := http.Header{}
			if tc.retryAfter != "" {
				header.Set("Retry-After", tc.retryAfter)
			}
			if tc.shouldRetry != "" {
				header.Set("x-should-retry", tc.shouldRetry)
			}
			delay, retry := retryAdvice(header)
			if (delay == nil) != (tc.wantDelay == nil) || delay != nil && *delay != *tc.wantDelay {
				t.Errorf("Retry-After %q: got %v want %v", tc.retryAfter, delay, tc.wantDelay)
			}
			if (retry == nil) != (tc.wantRetry == nil) || retry != nil && *retry != *tc.wantRetry {
				t.Errorf("x-should-retry %q: got %v want %v", tc.shouldRetry, retry, tc.wantRetry)
			}
		})
	}
}

// The core's 502 carries the service's retry advice, and its message keeps
// the format callers already parse.
func TestDecisionFailureForwardsServiceRetryAdvice(t *testing.T) {
	fixture, _, url := signalsSetup(t)
	body := `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`
	for _, tc := range []struct {
		name, code    string
		status        int
		headers       map[string]string
		wantMessage   string
		wantRetry     string
		wantRetryHint string
	}{
		{name: "overloaded", status: http.StatusServiceUnavailable, headers: map[string]string{"Retry-After": "5", "x-should-retry": "true"},
			wantMessage: "Decisions request refused (HTTP 503)", wantRetry: "5", wantRetryHint: "true"},
		{name: "deterministic", status: http.StatusBadRequest, code: "policy_unknown_preset", headers: map[string]string{"x-should-retry": "false"},
			wantMessage: "Decisions request refused (HTTP 400, policy_unknown_preset)", wantRetryHint: "false"},
		{name: "date ignored", status: http.StatusTooManyRequests, headers: map[string]string{"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT", "x-should-retry": "maybe"},
			wantMessage: "Decisions request refused (HTTP 429)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture.answer = func(w http.ResponseWriter) bool {
				for name, value := range tc.headers {
					w.Header().Set(name, value)
				}
				w.WriteHeader(tc.status)
				if tc.code != "" {
					_, _ = w.Write([]byte(`{"error":{"code":"` + tc.code + `"}}`))
				}
				return true
			}
			status, header, response := executeWithHeaders(t, url, body, "retry-advice-"+tc.name)
			if status != http.StatusBadGateway || !strings.Contains(string(response), "decision_failed") || !strings.Contains(string(response), tc.wantMessage) {
				t.Fatalf("status %d: %s", status, response)
			}
			if got := header.Get("Retry-After"); got != tc.wantRetry {
				t.Errorf("Retry-After %q want %q", got, tc.wantRetry)
			}
			if got := header.Get("x-should-retry"); got != tc.wantRetryHint {
				t.Errorf("x-should-retry %q want %q", got, tc.wantRetryHint)
			}
		})
	}
}

// The current (non-v2) API returns the same refusal type and message, with
// the advice attached; a transport failure carries none.
func TestCurrentRefusalCarriesRetryAdviceAndTransportFailureNone(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "test-key")
	config := currentTestConfig(t)
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "9")
		w.Header().Set("x-should-retry", "true")
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	config.Decisions.URL = endpoint.URL
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.Decide(t.Context(), currentTestInput())
	refusal, ok := err.(*DecisionsRefusal)
	if !ok || err.Error() != "Decisions request refused (HTTP 503)" || refusal.RetryAfter == nil || *refusal.RetryAfter != 9*time.Second || refusal.ShouldRetry == nil || !*refusal.ShouldRetry {
		t.Fatalf("refusal %#v: %v", err, err)
	}
	endpoint.Close()
	_, err = provider.Decide(t.Context(), currentTestInput())
	if err == nil || err.Error() != "Decisions request failed" {
		t.Fatalf("transport failure %v", err)
	}
	if _, ok := err.(*DecisionsRefusal); ok {
		t.Fatal("transport failure reported as a refusal")
	}
}
