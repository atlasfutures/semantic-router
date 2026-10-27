package raylinearc

import (
	"context"
	"crypto/tls"
	"math"
	"net/http/httptrace"
	"sync"
	"time"
)

// TurnSpan names one timed span of an ARC turn inside the cell.
type TurnSpan int

// The spans of one turn, in the order they happen. A span that did not happen
// is left out of the log line, never reported as zero.
const (
	SpanHeadersToBody TurnSpan = iota
	SpanParse
	SpanEpisodeLoad
	SpanEpisodeCommit
	SpanSessionLookup
	SpanSessionStart
	SpanDecidePrepare
	SpanDecideMarshal
	SpanDNS
	SpanConnect
	SpanTLS
	SpanWroteRequest
	SpanFirstByte
	SpanHTTP
	SpanDecode
	SpanPolicy
	turnSpanCount
)

var turnSpanFields = [turnSpanCount]string{
	SpanHeadersToBody: "headers_to_body_ms",
	SpanParse:         "parse_ms",
	SpanEpisodeLoad:   "episode_load_ms",
	SpanEpisodeCommit: "episode_commit_ms",
	SpanSessionLookup: "session_lookup_ms",
	SpanSessionStart:  "session_start_ms",
	SpanDecidePrepare: "decide_prepare_ms",
	SpanDecideMarshal: "decide_marshal_ms",
	SpanDNS:           "dns_ms",
	SpanConnect:       "connect_ms",
	SpanTLS:           "tls_ms",
	SpanWroteRequest:  "wrote_request_ms",
	SpanFirstByte:     "first_byte_ms",
	SpanHTTP:          "http_ms",
	SpanDecode:        "decode_ms",
	SpanPolicy:        "policy_ms",
}

// TurnTiming collects one turn's cell-side spans for the
// rayline_arc_turn_timing line: names and numbers only, never content,
// tokens or keys. Every method is safe on a nil receiver and for concurrent
// use; once Seal has run, later records are dropped, so a call that outlives
// the turn (an off-path append) changes nothing already logged.
//
// The per-call spans (marshal, the HTTP trace, decode) and byte counts
// describe the turn's last policy-service call; policy_calls counts them.
type TurnTiming struct {
	mu            sync.Mutex
	sealed        bool
	spans         [turnSpanCount]time.Duration
	has           uint32
	bodyBytes     int
	requestBytes  int
	responseBytes int
	policyCalls   int
	sessionStarts int
	reusedKnown   bool
	reused        bool
	operation     string
	episodeIDHash string
	contextTokens int
	sessionAction string
}

// NewTurnTiming is an empty timing for one turn.
func NewTurnTiming() *TurnTiming { return &TurnTiming{} }

type turnTimingKey struct{}

// WithTurnTiming carries the turn's timing to the policy-service client.
func WithTurnTiming(ctx context.Context, timing *TurnTiming) context.Context {
	if timing == nil {
		return ctx
	}
	return context.WithValue(ctx, turnTimingKey{}, timing)
}

// TurnTimingFrom is the timing a context carries, nil for none.
func TurnTimingFrom(ctx context.Context) *TurnTiming {
	if ctx == nil {
		return nil
	}
	timing, _ := ctx.Value(turnTimingKey{}).(*TurnTiming)
	return timing
}

// Set records a span, replacing any earlier value.
func (timing *TurnTiming) Set(span TurnSpan, duration time.Duration) {
	if timing == nil || span < 0 || span >= turnSpanCount {
		return
	}
	timing.mu.Lock()
	defer timing.mu.Unlock()
	if !timing.sealed {
		timing.set(span, duration)
	}
}

func (timing *TurnTiming) set(span TurnSpan, duration time.Duration) {
	timing.spans[span] = duration
	timing.has |= 1 << uint(span)
}

// SetBodyBytes records the request body's size.
func (timing *TurnTiming) SetBodyBytes(bytes int) {
	if timing == nil {
		return
	}
	timing.mu.Lock()
	defer timing.mu.Unlock()
	if !timing.sealed {
		timing.bodyBytes = bytes
	}
}

// SetTurn records the turn's identity facts the router already holds: the
// hashed episode id and its context token estimate (zero when unknown).
func (timing *TurnTiming) SetTurn(episodeIDHash string, contextTokens int) {
	if timing == nil {
		return
	}
	timing.mu.Lock()
	defer timing.mu.Unlock()
	if !timing.sealed {
		timing.episodeIDHash, timing.contextTokens = episodeIDHash, contextTokens
	}
}

// SetSessionAction records the service's bounded session action.
func (timing *TurnTiming) SetSessionAction(action string) {
	if timing == nil {
		return
	}
	timing.mu.Lock()
	defer timing.mu.Unlock()
	if !timing.sealed {
		timing.sessionAction = action
	}
}

// AddSessionStart adds one Modal session start.
func (timing *TurnTiming) AddSessionStart(duration time.Duration) {
	if timing == nil {
		return
	}
	timing.mu.Lock()
	defer timing.mu.Unlock()
	if timing.sealed {
		return
	}
	timing.sessionStarts++
	timing.set(SpanSessionStart, timing.spans[SpanSessionStart]+duration)
}

// policyCallTiming is one policy-service call's measurements. The httptrace
// hooks can run on the transport's own goroutines (a dial abandoned by a
// cancelled request finishes after Do returns), so they write under a lock
// of their own and the call copies them out under it.
type policyCallTiming struct {
	mu           sync.Mutex
	start        time.Time
	dnsStart     time.Time
	dns          time.Duration
	connectStart time.Time
	connect      time.Duration
	tlsStart     time.Time
	tls          time.Duration
	wrote        time.Time
	firstByte    time.Time
	reusedKnown  bool
	reused       bool
}

func (call *policyCallTiming) clientTrace() *httptrace.ClientTrace {
	stamp := func(apply func(now time.Time)) {
		now := time.Now()
		call.mu.Lock()
		apply(now)
		call.mu.Unlock()
	}
	return &httptrace.ClientTrace{
		GotConn: func(info httptrace.GotConnInfo) {
			stamp(func(time.Time) { call.reusedKnown, call.reused = true, info.Reused })
		},
		DNSStart: func(httptrace.DNSStartInfo) { stamp(func(now time.Time) { call.dnsStart = now }) },
		DNSDone: func(httptrace.DNSDoneInfo) {
			stamp(func(now time.Time) {
				if !call.dnsStart.IsZero() {
					call.dns += now.Sub(call.dnsStart)
				}
			})
		},
		ConnectStart: func(string, string) {
			stamp(func(now time.Time) {
				if call.connectStart.IsZero() {
					call.connectStart = now
				}
			})
		},
		ConnectDone: func(string, string, error) {
			stamp(func(now time.Time) {
				if !call.connectStart.IsZero() {
					call.connect = now.Sub(call.connectStart)
				}
			})
		},
		TLSHandshakeStart: func() { stamp(func(now time.Time) { call.tlsStart = now }) },
		TLSHandshakeDone: func(tls.ConnectionState, error) {
			stamp(func(now time.Time) {
				if !call.tlsStart.IsZero() {
					call.tls = now.Sub(call.tlsStart)
				}
			})
		},
		WroteRequest: func(httptrace.WroteRequestInfo) {
			stamp(func(now time.Time) {
				if call.wrote.IsZero() {
					call.wrote = now
				}
			})
		},
		GotFirstResponseByte: func() {
			stamp(func(now time.Time) {
				if call.firstByte.IsZero() {
					call.firstByte = now
				}
			})
		},
	}
}

// recordPolicyCall replaces the per-call spans with this call's. finished is
// when the response body was read; zero when the call failed before it.
func (timing *TurnTiming) recordPolicyCall(
	operation string,
	marshal time.Duration,
	requestBytes int,
	call *policyCallTiming,
	finished time.Time,
	responseBytes int,
) {
	if timing == nil || call == nil {
		return
	}
	call.mu.Lock()
	snapshot := policyCallTiming{
		start: call.start, dns: call.dns, connect: call.connect, tls: call.tls,
		dnsStart: call.dnsStart, connectStart: call.connectStart, tlsStart: call.tlsStart,
		wrote: call.wrote, firstByte: call.firstByte, reusedKnown: call.reusedKnown, reused: call.reused,
	}
	call.mu.Unlock()
	timing.mu.Lock()
	defer timing.mu.Unlock()
	if timing.sealed {
		return
	}
	timing.policyCalls++
	timing.operation = operation
	timing.requestBytes, timing.responseBytes = requestBytes, responseBytes
	timing.reusedKnown, timing.reused = snapshot.reusedKnown, snapshot.reused
	for _, span := range []TurnSpan{SpanDNS, SpanConnect, SpanTLS, SpanWroteRequest, SpanFirstByte, SpanHTTP, SpanDecode} {
		timing.has &^= 1 << uint(span)
	}
	timing.set(SpanDecideMarshal, marshal)
	if !snapshot.dnsStart.IsZero() {
		timing.set(SpanDNS, snapshot.dns)
	}
	if !snapshot.connectStart.IsZero() {
		timing.set(SpanConnect, snapshot.connect)
	}
	if !snapshot.tlsStart.IsZero() {
		timing.set(SpanTLS, snapshot.tls)
	}
	if !snapshot.wrote.IsZero() {
		timing.set(SpanWroteRequest, snapshot.wrote.Sub(snapshot.start))
		if !snapshot.firstByte.IsZero() {
			timing.set(SpanFirstByte, snapshot.firstByte.Sub(snapshot.wrote))
		}
	}
	if !finished.IsZero() {
		timing.set(SpanHTTP, finished.Sub(snapshot.start))
	}
}

// Seal ends the turn: it returns the log fields once, with total_ms, and nil
// on any later call, so a turn is logged at most once.
func (timing *TurnTiming) Seal(total time.Duration) map[string]interface{} {
	if timing == nil {
		return nil
	}
	timing.mu.Lock()
	defer timing.mu.Unlock()
	if timing.sealed {
		return nil
	}
	timing.sealed = true
	fields := make(map[string]interface{}, 28)
	for span := TurnSpan(0); span < turnSpanCount; span++ {
		if timing.has&(1<<uint(span)) != 0 {
			fields[turnSpanFields[span]] = turnMillis(timing.spans[span])
		}
	}
	if timing.has&(1<<uint(SpanDecidePrepare)) != 0 || timing.has&(1<<uint(SpanDecideMarshal)) != 0 {
		fields["decide_build_ms"] = turnMillis(timing.spans[SpanDecidePrepare] + timing.spans[SpanDecideMarshal])
	}
	fields["total_ms"] = turnMillis(total)
	if timing.bodyBytes > 0 {
		fields["body_bytes"] = timing.bodyBytes
	}
	if timing.policyCalls > 0 {
		fields["policy_calls"] = timing.policyCalls
		fields["policy_operation"] = timing.operation
		fields["decide_request_bytes"] = timing.requestBytes
		fields["decide_response_bytes"] = timing.responseBytes
	}
	if timing.reusedKnown {
		fields["reused_conn"] = timing.reused
	}
	if timing.sessionStarts > 0 {
		fields["session_starts"] = timing.sessionStarts
	}
	if timing.episodeIDHash != "" {
		fields["episode_id_hash"] = timing.episodeIDHash
	}
	if timing.contextTokens > 0 {
		fields["context_tokens"] = timing.contextTokens
	}
	if timing.sessionAction != "" {
		fields["session_action"] = timing.sessionAction
	}
	return fields
}

// turnMillis is a duration in milliseconds at 0.1 ms resolution.
func turnMillis(duration time.Duration) float64 {
	return math.Round(float64(duration)/float64(100*time.Microsecond)) / 10
}
