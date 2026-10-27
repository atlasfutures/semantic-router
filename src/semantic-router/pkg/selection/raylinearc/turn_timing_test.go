package raylinearc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func requireTimingFields(t *testing.T, fields map[string]interface{}, names ...string) {
	t.Helper()
	for _, name := range names {
		if _, ok := fields[name]; !ok {
			t.Errorf("%s missing from %v", name, fields)
		}
	}
}

func refuseTimingFields(t *testing.T, fields map[string]interface{}, names ...string) {
	t.Helper()
	for _, name := range names {
		if value, ok := fields[name]; ok {
			t.Errorf("%s = %v, want it left out", name, value)
		}
	}
}

// A decide through a context that carries the turn's timing records the
// call's marshal, connection, upload, server and decode spans, and a second
// call on the kept-alive connection reports it reused and dials nothing.
func TestTurnTimingRecordsThePolicyCall(t *testing.T) {
	body := readPolicyFixture(t, "decision_response.v1.json")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Millisecond)
		_, _ = w.Write(body)
	}))
	t.Cleanup(server.Close)
	client := NewPolicyServiceClient(PolicyServiceConfig{
		BaseURL: server.URL, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: 5 * time.Second,
	})

	first := NewTurnTiming()
	if _, err := client.Decide(WithTurnTiming(context.Background(), first), sessionDecisionRequest(t)); err != nil {
		t.Fatal(err)
	}
	fields := first.Seal(time.Second)
	requireTimingFields(t, fields, "decide_marshal_ms", "decide_build_ms", "connect_ms", "wrote_request_ms",
		"first_byte_ms", "http_ms", "decode_ms", "decide_request_bytes", "decide_response_bytes", "total_ms")
	// An IP literal resolves nothing, and plain HTTP shakes no hands.
	refuseTimingFields(t, fields, "dns_ms", "tls_ms", "session_start_ms", "episode_load_ms")
	if fields["reused_conn"] != false || fields["policy_calls"] != 1 || fields["policy_operation"] != "decide" {
		t.Fatalf("fields = %v", fields)
	}
	if fields["first_byte_ms"].(float64) < 2 {
		t.Fatalf("first_byte_ms = %v, want the server's 2 ms in it", fields["first_byte_ms"])
	}
	if fields["decide_response_bytes"] != len(body) || fields["decide_request_bytes"].(int) <= 0 {
		t.Fatalf("byte counts = %v / %v", fields["decide_request_bytes"], fields["decide_response_bytes"])
	}
	if fields["total_ms"] != 1000.0 {
		t.Fatalf("total_ms = %v", fields["total_ms"])
	}
	if again := first.Seal(time.Second); again != nil {
		t.Fatalf("a sealed turn logged again: %v", again)
	}

	second := NewTurnTiming()
	if _, err := client.Decide(WithTurnTiming(context.Background(), second), sessionDecisionRequest(t)); err != nil {
		t.Fatal(err)
	}
	fields = second.Seal(time.Second)
	if fields["reused_conn"] != true {
		t.Fatalf("reused_conn = %v on a kept-alive connection", fields["reused_conn"])
	}
	refuseTimingFields(t, fields, "connect_ms", "dns_ms")
}

// A refused call still records its HTTP spans, and decodes nothing; a turn
// without a timing records nothing anywhere.
func TestTurnTimingRecordsAFailedCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"backend_unavailable"}`))
	}))
	t.Cleanup(server.Close)
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: 5 * time.Second})
	timing := NewTurnTiming()
	if _, err := client.Decide(WithTurnTiming(context.Background(), timing), sessionDecisionRequest(t)); err == nil {
		t.Fatal("a refused decide succeeded")
	}
	fields := timing.Seal(0)
	requireTimingFields(t, fields, "wrote_request_ms", "first_byte_ms", "http_ms", "policy_calls")
	refuseTimingFields(t, fields, "decode_ms")

	// The service unreachable: the call is counted, with no server span.
	server.Close()
	timing = NewTurnTiming()
	if _, err := client.Decide(WithTurnTiming(context.Background(), timing), sessionDecisionRequest(t)); err == nil {
		t.Fatal("a decide to a closed server succeeded")
	}
	fields = timing.Seal(0)
	if fields["policy_calls"] != 1 {
		t.Fatalf("policy_calls = %v", fields["policy_calls"])
	}
	refuseTimingFields(t, fields, "first_byte_ms", "http_ms", "decode_ms")

	if TurnTimingFrom(context.Background()) != nil || TurnTimingFrom(nil) != nil { //nolint:staticcheck // a nil context is the case under test
		t.Fatal("a bare context carried a timing")
	}
	var none *TurnTiming
	none.Set(SpanParse, time.Millisecond)
	none.AddSessionStart(time.Millisecond)
	if none.Seal(0) != nil {
		t.Fatal("a nil timing logged")
	}
}

// A sessioned turn times its session lookup and the Modal session start, and
// the line carries no token, key, secret or session id.
func TestTurnTimingTimesTheSessionStartAndLogsNoSecret(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	backend := testSessionBackend(t, fake, time.Second)
	timing := NewTurnTiming()
	timing.SetTurn(episodeHash("a"), 1200)
	timing.SetSessionAction("created")
	turn := backend.Bind(nil, false)
	if _, err := turn.Decide(WithTurnTiming(context.Background(), timing), sessionDecisionRequest(t)); err != nil {
		t.Fatal(err)
	}
	session, _ := turn.Session()
	fields := timing.Seal(time.Millisecond)
	requireTimingFields(t, fields, "session_lookup_ms", "session_start_ms", "session_starts", "http_ms")
	if fields["session_starts"] != 1 || fields["episode_id_hash"] != episodeHash("a") ||
		fields["context_tokens"] != 1200 || fields["session_action"] != "created" {
		t.Fatalf("fields = %v", fields)
	}
	encoded, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{fakeModalKey, fakeModalSecret, session.Token, session.ID, "secret-material"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("the timing line carries %q: %s", secret, encoded)
		}
	}

	// A persisted session is looked up and never started.
	timing = NewTurnTiming()
	if _, err := backend.Bind(session, false).Decide(WithTurnTiming(context.Background(), timing), sessionDecisionRequest(t)); err != nil {
		t.Fatal(err)
	}
	fields = timing.Seal(0)
	requireTimingFields(t, fields, "session_lookup_ms")
	refuseTimingFields(t, fields, "session_start_ms", "session_starts")
}

// Records from many goroutines, and after the seal, are race-free; a record
// after the seal changes nothing.
func TestTurnTimingIsSafeForConcurrentRecords(t *testing.T) {
	timing := NewTurnTiming()
	var wait sync.WaitGroup
	for index := 0; index < 8; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for span := TurnSpan(0); span < turnSpanCount; span++ {
				if span != SpanSessionStart {
					timing.Set(span, time.Millisecond)
				}
			}
			timing.AddSessionStart(time.Millisecond)
		}()
	}
	wait.Wait()
	fields := timing.Seal(0)
	if fields["session_starts"] != 8 || fields["session_start_ms"] != 8.0 || fields["parse_ms"] != 1.0 {
		t.Fatalf("fields = %v", fields)
	}
	timing.Set(SpanParse, time.Hour)
	if timing.Seal(0) != nil {
		t.Fatal("a sealed timing logged twice")
	}
}

func TestTurnMillisKeepsATenthOfAMillisecond(t *testing.T) {
	for duration, want := range map[time.Duration]float64{
		0: 0, 49 * time.Microsecond: 0, 150 * time.Microsecond: 0.2, 1234567 * time.Nanosecond: 1.2, 2 * time.Second: 2000,
	} {
		if got := turnMillis(duration); got != want {
			t.Errorf("turnMillis(%v) = %v, want %v", duration, got, want)
		}
	}
}
