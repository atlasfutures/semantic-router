package raylinearc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// A release names the package and the episode only, and its outcome reads
// the service's answer: either success status is released, 404 or 405 a
// service that predates it, anything else a failure.
func TestPolicyServiceClientReleaseOutcomes(t *testing.T) {
	cases := []struct {
		status int
		body   string
		want   string
	}{
		{http.StatusOK, `{}`, PolicyReleaseReleased},
		{http.StatusNoContent, ``, PolicyReleaseReleased},
		{http.StatusNotFound, `{"detail":"Not Found"}`, PolicyReleaseUnsupported},
		{http.StatusMethodNotAllowed, `{"detail":"Method Not Allowed"}`, PolicyReleaseUnsupported},
		{http.StatusInternalServerError, `{"error":"internal"}`, PolicyReleaseFailed},
	}
	for _, tc := range cases {
		var got PolicyReleaseRequest
		var path string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			path = r.URL.Path
			body, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(body, &got); err != nil {
				t.Error(err)
			}
			w.WriteHeader(tc.status)
			_, _ = w.Write([]byte(tc.body))
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		episode := strings.Repeat("a", 64)
		err := client.Release(context.Background(), PolicyReleaseRequest{
			Package: PolicyPackageRef{Alias: "alias", PackageSHA256: strings.Repeat("b", 64)}, EpisodeIDHash: episode,
		})
		server.Close()
		if outcome := PolicyReleaseOutcomeOf(err); outcome != tc.want {
			t.Errorf("status %d: outcome %q (%v), want %q", tc.status, outcome, err, tc.want)
		}
		if path != "/v1/rayline/arc/policy/release" || got.SchemaVersion != PolicyReleaseRequestSchema ||
			got.EpisodeIDHash != episode || got.Package.Alias != "alias" {
			t.Errorf("status %d: sent %s %+v", tc.status, path, got)
		}
	}
}

// A 503 is a service that kept the episode for a retry: the release is
// asked once more, and only once.
func TestPolicyServiceClientReleaseRetriesA503Once(t *testing.T) {
	for _, tc := range []struct {
		name     string
		statuses []int
		want     string
		calls    int
	}{
		{"recovers", []int{http.StatusServiceUnavailable, http.StatusOK}, PolicyReleaseReleased, 2},
		{"still down", []int{http.StatusServiceUnavailable, http.StatusServiceUnavailable, http.StatusOK}, PolicyReleaseFailed, 2},
		{"other failure", []int{http.StatusInternalServerError, http.StatusOK}, PolicyReleaseFailed, 1},
	} {
		calls := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			status := tc.statuses[calls]
			calls++
			w.WriteHeader(status)
			if status == http.StatusServiceUnavailable {
				_, _ = w.Write([]byte(`{"error":"backend_unavailable"}`))
			}
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		err := client.Release(context.Background(), PolicyReleaseRequest{EpisodeIDHash: strings.Repeat("a", 64)})
		server.Close()
		if outcome := PolicyReleaseOutcomeOf(err); outcome != tc.want || calls != tc.calls {
			t.Errorf("%s: outcome %q after %d calls, want %q after %d", tc.name, outcome, calls, tc.want, tc.calls)
		}
	}
}

// The worst outcome of a turn's releases is the turn's.
func TestWorsePolicyReleaseOutcome(t *testing.T) {
	outcome := PolicyReleaseNone
	for _, next := range []string{PolicyReleaseReleased, PolicyReleaseUnsupported, PolicyReleaseReleased, PolicyReleaseFailed, PolicyReleaseReleased} {
		outcome = worsePolicyReleaseOutcome(outcome, next)
	}
	if outcome != PolicyReleaseFailed {
		t.Fatalf("outcome %q", outcome)
	}
}
