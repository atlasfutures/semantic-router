package extproc

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// slowStream answers its headers after headerDelay, then writes frames, one
// every gap, flushing each.
func slowStream(headerDelay, gap time.Duration, frames int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(headerDelay):
		case <-r.Context().Done():
			return
		}
		w.Header().Set("content-type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		for i := 0; i < frames; i++ {
			select {
			case <-time.After(gap):
			case <-r.Context().Done():
				return
			}
			_, _ = w.Write([]byte("data: {\"n\":1}\n\n"))
			w.(http.Flusher).Flush()
		}
	}))
}

func twinCall(url string) *flexTwinCall {
	return &flexTwinCall{url: url, headers: http.Header{}, body: []byte("{}")}
}

// A re-send whose headers come within the budget is read to its end even when
// the generation runs past the budget: the budget bounds the wait for an
// answer, not the answer (a 300 s budget cut every longer re-sent generation).
func TestFlexTwinBodyOutlivesTheBudget(t *testing.T) {
	server := slowStream(0, 40*time.Millisecond, 8) // about 320 ms of body
	defer server.Close()
	ctx := &RequestContext{RequestID: "rt_body_outlives"}
	start := time.Now()
	result := callFlexTwin(ctx, twinCall(server.URL), 1<<20, start.Add(100*time.Millisecond), start.Add(5*time.Second))
	if result.err != nil || result.status != http.StatusOK || !result.delivered {
		t.Fatalf("want the whole body, got status %d delivered %v err %v", result.status, result.delivered, result.err)
	}
	if got := strings.Count(string(result.body), "data: "); got != 8 {
		t.Fatalf("want 8 frames, got %d", got)
	}
}

// Headers later than the budget are still a re-send out of time: a 504 the
// pipeline settles as the twin's failure, as before.
func TestFlexTwinHeadersPastTheBudgetAreOutOfTime(t *testing.T) {
	server := slowStream(400*time.Millisecond, 0, 1)
	defer server.Close()
	ctx := &RequestContext{RequestID: "rt_headers_late"}
	start := time.Now()
	result := callFlexTwin(ctx, twinCall(server.URL), 1<<20, start.Add(100*time.Millisecond), start.Add(5*time.Second))
	if result.status != http.StatusGatewayTimeout || !errors.Is(result.err, context.DeadlineExceeded) {
		t.Fatalf("want 504 with DeadlineExceeded, got status %d err %v", result.status, result.err)
	}
	if elapsed := time.Since(start); elapsed > 350*time.Millisecond {
		t.Fatalf("the budget did not cut the wait: %v", elapsed)
	}
}

// The body is still bounded, by the turn's stream deadline: a body cut there
// keeps what arrived and reports the read error, which terminalFlexTwinResult
// turns into standard_resend_body_cut.
func TestFlexTwinBodyStopsAtTheStreamDeadline(t *testing.T) {
	server := slowStream(0, 50*time.Millisecond, 40) // about 2 s of body
	defer server.Close()
	ctx := &RequestContext{RequestID: "rt_body_deadline"}
	start := time.Now()
	result := callFlexTwin(ctx, twinCall(server.URL), 1<<20, start.Add(100*time.Millisecond), start.Add(300*time.Millisecond))
	if !result.delivered || result.status != http.StatusOK || !errors.Is(result.err, context.DeadlineExceeded) {
		t.Fatalf("want a delivered 200 cut by the deadline, got status %d delivered %v err %v",
			result.status, result.delivered, result.err)
	}
	if n := strings.Count(string(result.body), "data: "); n == 0 || n >= 40 {
		t.Fatalf("want a partial body, got %d frames", n)
	}
}

func TestFlexTwinBodyDeadlineIsTheTurnsStreamDeadline(t *testing.T) {
	start := time.Now().Add(-time.Minute)
	ctx := &RequestContext{StartTime: start}
	r := &OpenAIRouter{Config: &config.RouterConfig{}}
	r.Config.ResponseStreamDeadlineSec = 590
	if got := r.flexTwinBodyDeadline(ctx); !got.Equal(start.Add(590 * time.Second)) {
		t.Fatalf("want start+590s, got %v", got.Sub(start))
	}
	r.Config.ResponseStreamDeadlineSec = -1
	if got := r.flexTwinBodyDeadline(ctx); !got.IsZero() {
		t.Fatalf("an off stream deadline leaves the body unbounded, got %v", got)
	}
	if got := (&OpenAIRouter{Config: &config.RouterConfig{}}).flexTwinBodyDeadline(&RequestContext{}); !got.IsZero() {
		t.Fatalf("no start time, no deadline; got %v", got)
	}
}
