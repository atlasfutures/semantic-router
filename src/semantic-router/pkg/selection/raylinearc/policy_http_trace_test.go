package raylinearc

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// Each decide response carries the httptrace of the exchange that returned
// it: the first call dials a fresh connection, and the second reuses it
// after it sat idle in the pool, with no dial phases.
func TestPolicyServiceDecideRecordsHTTPTiming(t *testing.T) {
	body := readPolicyFixture(t, "decision_response.v1.json")
	var received atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		sent, _ := io.ReadAll(request.Body)
		received.Store(int64(len(sent)))
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
	if timing.RequestBytes <= 0 || int64(timing.RequestBytes) != received.Load() || timing.Proto != "HTTP/1.1" {
		t.Fatalf("request_bytes %d (server read %d) proto %q, want equal and HTTP/1.1",
			timing.RequestBytes, received.Load(), timing.Proto)
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

// An exchange negotiated over HTTP/2 records that protocol.
func TestPolicyServiceDecideRecordsHTTP2Proto(t *testing.T) {
	body := readPolicyFixture(t, "decision_response.v1.json")
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write(body)
	}))
	server.EnableHTTP2 = true
	server.StartTLS()
	defer server.Close()
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: 5 * time.Second})
	client.http = server.Client()

	response, err := client.Decide(context.Background(), PolicyDecisionRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if timing := response.HTTPTiming; timing == nil || timing.Proto != "HTTP/2.0" || timing.RequestBytes <= 0 {
		t.Fatalf("timing = %+v, want HTTP/2.0 and the request size", timing)
	}
}

// A refused exchange's timing still carries the request size and protocol.
func TestPolicyServiceRefusalRecordsRequestBytesAndProto(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusConflict)
		_, _ = writer.Write([]byte(`{"error":"session_busy"}`))
	}))
	defer server.Close()
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: 5 * time.Second})

	_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.HTTPTiming == nil {
		t.Fatalf("err = %v, want a refusal with its timing", err)
	}
	if failure.HTTPTiming.RequestBytes <= 0 || failure.HTTPTiming.Proto != "HTTP/1.1" {
		t.Fatalf("refusal timing = %+v, want the request size and HTTP/1.1", *failure.HTTPTiming)
	}
}

// A nil trace reads as no timing rather than a panic.
func TestPolicyHTTPTraceSnapshotIsNilSafe(t *testing.T) {
	var trace *policyHTTPTrace
	if trace.snapshot() != nil {
		t.Fatal("nil trace produced a timing")
	}
}
