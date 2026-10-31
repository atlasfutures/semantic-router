package raylinearc

import (
	"context"
	"crypto/tls"
	"net/http/httptrace"
	"sync"
	"time"
)

// PolicyHTTPTiming is one policy-service HTTP exchange as net/http/httptrace
// saw it: whether the connection was reused and how long it sat idle, and
// the client-side phases in milliseconds from the moment the request was
// handed to the transport. DNS, connect and TLS are zero on a reused
// connection. A phase that never happened reads zero.
type PolicyHTTPTiming struct {
	Reused          bool
	WasIdle         bool
	IdleMillis      float64
	GotConnMillis   float64
	DNSMillis       float64
	ConnectMillis   float64
	TLSMillis       float64
	WroteMillis     float64
	FirstByteMillis float64
}

// policyHTTPTrace collects the httptrace callbacks of one exchange. The
// transport may call them from its dial goroutines, so every write holds the
// lock, and the caller reads a copy once the exchange is over.
type policyHTTPTrace struct {
	mu           sync.Mutex
	start        time.Time
	dnsStart     time.Time
	connectStart time.Time
	tlsStart     time.Time
	timing       PolicyHTTPTiming
}

func sinceMillis(from, to time.Time) float64 {
	if from.IsZero() || to.Before(from) {
		return 0
	}
	return float64(to.Sub(from)) / float64(time.Millisecond)
}

// withPolicyHTTPTrace attaches a trace to ctx; the clock starts now.
func withPolicyHTTPTrace(ctx context.Context) (context.Context, *policyHTTPTrace) {
	trace := &policyHTTPTrace{start: time.Now()}
	record := func(apply func(now time.Time)) {
		now := time.Now()
		trace.mu.Lock()
		apply(now)
		trace.mu.Unlock()
	}
	client := &httptrace.ClientTrace{
		GotConn: func(info httptrace.GotConnInfo) {
			record(func(now time.Time) {
				trace.timing.Reused = info.Reused
				trace.timing.WasIdle = info.WasIdle
				trace.timing.IdleMillis = float64(info.IdleTime) / float64(time.Millisecond)
				trace.timing.GotConnMillis = sinceMillis(trace.start, now)
			})
		},
		DNSStart: func(httptrace.DNSStartInfo) {
			record(func(now time.Time) { trace.dnsStart = now })
		},
		DNSDone: func(httptrace.DNSDoneInfo) {
			record(func(now time.Time) { trace.timing.DNSMillis = sinceMillis(trace.dnsStart, now) })
		},
		ConnectStart: func(string, string) {
			record(func(now time.Time) {
				if trace.connectStart.IsZero() {
					trace.connectStart = now
				}
			})
		},
		ConnectDone: func(_, _ string, err error) {
			record(func(now time.Time) {
				if err == nil && trace.timing.ConnectMillis == 0 {
					trace.timing.ConnectMillis = sinceMillis(trace.connectStart, now)
				}
			})
		},
		TLSHandshakeStart: func() {
			record(func(now time.Time) { trace.tlsStart = now })
		},
		TLSHandshakeDone: func(tls.ConnectionState, error) {
			record(func(now time.Time) { trace.timing.TLSMillis = sinceMillis(trace.tlsStart, now) })
		},
		WroteRequest: func(httptrace.WroteRequestInfo) {
			record(func(now time.Time) { trace.timing.WroteMillis = sinceMillis(trace.start, now) })
		},
		GotFirstResponseByte: func() {
			record(func(now time.Time) { trace.timing.FirstByteMillis = sinceMillis(trace.start, now) })
		},
	}
	return httptrace.WithClientTrace(ctx, client), trace
}

// snapshot is the exchange as recorded so far; nil-safe.
func (trace *policyHTTPTrace) snapshot() *PolicyHTTPTiming {
	if trace == nil {
		return nil
	}
	trace.mu.Lock()
	defer trace.mu.Unlock()
	timing := trace.timing
	return &timing
}
