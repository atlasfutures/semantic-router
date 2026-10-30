package raylinearc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Each decide response carries the httptrace of the exchange that returned
// it: the first call dials a fresh connection, and the second reuses it
// after it sat idle in the pool, with no dial phases.
func TestPolicyServiceDecideRecordsHTTPTiming(t *testing.T) {
	body := readPolicyFixture(t, "decision_response.v1.json")
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write(body)
	}))
	defer server.Close()
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: 5 * time.Second})

	first, err := client.Decide(context.Background(), PolicyDecisionRequest{})
	if err != nil {
		t.Fatal(err)
	}
	timing := first.HTTPTiming
	if timing == nil {
		t.Fatal("first decide carries no HTTP timing")
	}
	if timing.Reused || timing.WasIdle {
		t.Fatalf("first call reused=%v was_idle=%v, want a fresh connection", timing.Reused, timing.WasIdle)
	}
	if timing.ConnectMillis <= 0 || timing.FirstByteMillis <= 0 || timing.WroteMillis <= 0 {
		t.Fatalf("first call phases = %+v, want connect, wrote and first byte recorded", *timing)
	}
	if timing.WroteMillis > timing.FirstByteMillis {
		t.Fatalf("wrote %.3f ms after first byte %.3f ms", timing.WroteMillis, timing.FirstByteMillis)
	}

	time.Sleep(5 * time.Millisecond)
	second, err := client.Decide(context.Background(), PolicyDecisionRequest{})
	if err != nil {
		t.Fatal(err)
	}
	timing = second.HTTPTiming
	if timing == nil || !timing.Reused || !timing.WasIdle || timing.IdleMillis <= 0 {
		t.Fatalf("second call timing = %+v, want a reused idle connection", timing)
	}
	if timing.DNSMillis != 0 || timing.ConnectMillis != 0 || timing.TLSMillis != 0 {
		t.Fatalf("reused call dial phases = %+v, want zero", *timing)
	}
}

// A nil trace reads as no timing rather than a panic.
func TestPolicyHTTPTraceSnapshotIsNilSafe(t *testing.T) {
	var trace *policyHTTPTrace
	if trace.snapshot() != nil {
		t.Fatal("nil trace produced a timing")
	}
}
