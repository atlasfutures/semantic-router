package raylinearc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// slowModalServer emulates a saturated Modal Server's session start: it
// answers after delay (or once hold is closed), with a new session or with
// code, and every start, repeated or not, is a new session.
type slowModalServer struct {
	server *httptest.Server
	delay  time.Duration
	code   int
	// hold, when set, holds every start until it is closed.
	hold chan struct{}
	// entered receives each start as it arrives.
	entered chan struct{}

	mu         sync.Mutex
	starts     int
	cancelled  int
	issued     []string
	terminated []string
}

func newSlowModalServer(t *testing.T, delay time.Duration) *slowModalServer {
	t.Helper()
	fake := &slowModalServer{delay: delay}
	fake.server = httptest.NewServer(http.HandlerFunc(fake.serve))
	t.Cleanup(fake.server.Close)
	return fake
}

func (fake *slowModalServer) serve(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Modal-Key") != fakeModalKey || r.Header.Get("Modal-Secret") != fakeModalSecret {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	switch r.URL.Path {
	case policySessionStartPath:
		fake.mu.Lock()
		fake.starts++
		hold, entered := fake.hold, fake.entered
		fake.mu.Unlock()
		if entered != nil {
			entered <- struct{}{}
		}
		if hold == nil {
			hold = make(chan struct{})
			time.AfterFunc(fake.delay, func() { close(hold) })
		}
		select {
		case <-hold:
		case <-r.Context().Done():
			fake.mu.Lock()
			fake.cancelled++
			fake.mu.Unlock()
			return
		}
		if fake.code != 0 {
			w.WriteHeader(fake.code)
			return
		}
		fake.mu.Lock()
		token := fmt.Sprintf("tok.%04d.secret-material", len(fake.issued))
		fake.issued = append(fake.issued, token)
		fake.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]string{"session_id": "sd-" + token[4:8], "token": token})
	case policySessionTerminatePath:
		fake.mu.Lock()
		fake.terminated = append(fake.terminated, strings.TrimPrefix(r.Header.Get("Modal-Authorization"), "Bearer "))
		fake.mu.Unlock()
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func reapBackend(t *testing.T, url string, config PolicySessionConfig) (*PolicySessionBackend, chan string) {
	t.Helper()
	client := NewPolicyServiceClient(PolicyServiceConfig{
		BaseURL: url, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: time.Second,
	})
	if config.StartTimeout == 0 {
		config.StartTimeout = 100 * time.Millisecond
	}
	if config.ReapTimeout == 0 {
		config.ReapTimeout = 5 * time.Second
	}
	backend, err := NewPolicySessionBackend(client, config)
	if err != nil {
		t.Fatal(err)
	}
	outcomes := make(chan string, 16)
	backend.onReap = func(outcome string) { outcomes <- outcome }
	return backend, outcomes
}

func nextReap(t *testing.T, outcomes chan string) string {
	t.Helper()
	select {
	case outcome := <-outcomes:
		return outcome
	case <-time.After(5 * time.Second):
		t.Fatal("no reap outcome")
		return ""
	}
}

func noReap(t *testing.T, outcomes chan string) {
	t.Helper()
	select {
	case outcome := <-outcomes:
		t.Fatalf("unexpected reap outcome %q", outcome)
	case <-time.After(300 * time.Millisecond):
	}
}

func startFailsClosed(t *testing.T, backend *PolicySessionBackend, ctx context.Context) {
	t.Helper()
	var failure *PolicyServiceError
	if _, err := backend.Start(ctx); !errors.As(err, &failure) || failure.Class != PolicySessionStartClass {
		t.Fatalf("err = %v, want class %s", err, PolicySessionStartClass)
	}
}

// assertReapedTheOne checks that the one start's own delayed session, and no
// other, was ended, with no second start.
func (fake *slowModalServer) assertReapedTheOne(t *testing.T) {
	t.Helper()
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.starts != 1 || len(fake.issued) != 1 {
		t.Fatalf("starts %d, sessions %d, want 1 and 1", fake.starts, len(fake.issued))
	}
	if len(fake.terminated) != 1 || fake.terminated[0] != fake.issued[0] {
		t.Fatalf("terminated %d sessions, want exactly the delayed one", len(fake.terminated))
	}
}

// A start past the start timeout fails the turn at the timeout, but its
// request is still awaited: the session its delayed 200 brings is ended.
func TestPolicySessionTimedOutStartIsReaped(t *testing.T) {
	fake := newSlowModalServer(t, 300*time.Millisecond)
	backend, outcomes := reapBackend(t, fake.server.URL, PolicySessionConfig{})
	began := time.Now()
	startFailsClosed(t, backend, context.Background())
	if elapsed := time.Since(began); elapsed > 250*time.Millisecond {
		t.Fatalf("the start waited %s past its timeout", elapsed)
	}
	if outcome := nextReap(t, outcomes); outcome != policySessionReaped {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionReaped)
	}
	fake.assertReapedTheOne(t)
}

// A caller that goes away mid-start is reaped the same way.
func TestPolicySessionCancelledStartIsReaped(t *testing.T) {
	fake := newSlowModalServer(t, 300*time.Millisecond)
	fake.entered = make(chan struct{}, 1)
	backend, outcomes := reapBackend(t, fake.server.URL, PolicySessionConfig{StartTimeout: 5 * time.Second})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-fake.entered
		cancel()
	}()
	startFailsClosed(t, backend, ctx)
	if outcome := nextReap(t, outcomes); outcome != policySessionReaped {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionReaped)
	}
	fake.assertReapedTheOne(t)
}

// A reap timeout shorter than the start timeout does not cut an in-time start
// short.
func TestPolicySessionReapTimeoutCoversTheStartTimeout(t *testing.T) {
	fake := newSlowModalServer(t, 150*time.Millisecond)
	backend, outcomes := reapBackend(t, fake.server.URL, PolicySessionConfig{
		StartTimeout: 2 * time.Second, ReapTimeout: 50 * time.Millisecond,
	})
	if _, err := backend.Start(context.Background()); err != nil {
		t.Fatalf("an in-time start failed: %v", err)
	}
	noReap(t, outcomes)
}

// A start that brings no answer by the reap timeout is failed; nothing is
// ended.
func TestPolicySessionUnansweredReapFails(t *testing.T) {
	fake := newSlowModalServer(t, 2*time.Second)
	backend, outcomes := reapBackend(t, fake.server.URL, PolicySessionConfig{
		StartTimeout: 50 * time.Millisecond, ReapTimeout: 200 * time.Millisecond,
	})
	startFailsClosed(t, backend, context.Background())
	if outcome := nextReap(t, outcomes); outcome != policySessionReapFailed {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionReapFailed)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.starts != 1 || len(fake.terminated) != 0 {
		t.Fatalf("starts %d terminates %d, want 1 and 0", fake.starts, len(fake.terminated))
	}
}

// A delayed refusal started no session: not_created, nothing ended.
func TestPolicySessionDelayedRefusalIsNotCreated(t *testing.T) {
	fake := newSlowModalServer(t, 300*time.Millisecond)
	fake.code = http.StatusTooManyRequests
	backend, outcomes := reapBackend(t, fake.server.URL, PolicySessionConfig{})
	startFailsClosed(t, backend, context.Background())
	if outcome := nextReap(t, outcomes); outcome != policySessionNotCreated {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionNotCreated)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.starts != 1 || len(fake.terminated) != 0 {
		t.Fatalf("starts %d terminates %d, want 1 and 0", fake.starts, len(fake.terminated))
	}
}

// Reaps beyond the bound are dropped: those starts are cancelled, as before.
func TestPolicySessionReapsAreBounded(t *testing.T) {
	fake := newSlowModalServer(t, 0)
	fake.hold = make(chan struct{})
	fake.entered = make(chan struct{}, 5)
	var once sync.Once
	release := func() { once.Do(func() { close(fake.hold) }) }
	t.Cleanup(release) // runs before the server closes (cleanups are LIFO)
	backend, outcomes := reapBackend(t, fake.server.URL, PolicySessionConfig{StartTimeout: 30 * time.Second, MaxReaps: 2})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var starts sync.WaitGroup
	for range 5 {
		starts.Add(1)
		go func() {
			defer starts.Done()
			if _, err := backend.Start(ctx); err == nil {
				t.Error("a held start succeeded")
			}
		}()
	}
	// Every start reaches the server before any caller gives up.
	for range 5 {
		select {
		case <-fake.entered:
		case <-time.After(10 * time.Second):
			t.Fatal("a start never reached the server")
		}
	}
	cancel()
	starts.Wait()
	for range 3 {
		if outcome := nextReap(t, outcomes); outcome != policySessionReapDrop {
			t.Fatalf("reap outcome %q, want %q while the bound is full", outcome, policySessionReapDrop)
		}
	}
	if dropped := backend.reapDropped.Load(); dropped != 3 {
		t.Fatalf("dropped %d reaps, want 3", dropped)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		fake.mu.Lock()
		cancelled := fake.cancelled
		fake.mu.Unlock()
		if cancelled == 3 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("cancelled %d starts, want the 3 dropped", cancelled)
		}
		time.Sleep(10 * time.Millisecond)
	}
	release()
	for range 2 {
		if outcome := nextReap(t, outcomes); outcome != policySessionReaped {
			t.Fatalf("reap outcome %q, want %q", outcome, policySessionReaped)
		}
	}
	noReap(t, outcomes)
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.starts != 5 || len(fake.terminated) != 2 {
		t.Fatalf("starts %d terminates %d, want 5 and 2", fake.starts, len(fake.terminated))
	}
}

// A start that answers in time, a session or a refusal, is never reaped.
func TestPolicySessionAnsweredStartIsNotReaped(t *testing.T) {
	modal := newFakeModalServer(t, 1)
	answered := testSessionBackend(t, modal, time.Second)
	outcomes := make(chan string, 4)
	answered.onReap = func(outcome string) { outcomes <- outcome }
	if _, err := answered.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	modal.mu.Lock()
	modal.startCode = http.StatusGatewayTimeout
	modal.mu.Unlock()
	if _, err := answered.Start(context.Background()); err == nil {
		t.Fatal("a refused start succeeded")
	}
	noReap(t, outcomes)
	if starts, terminates, _ := modal.counts(); starts != 1 || terminates != 0 {
		t.Fatalf("starts %d terminates %d, want 1 and 0", starts, terminates)
	}
}

// A refusal whose body could not be read is still a refusal: not reaped.
func TestPolicySessionUnreadableRefusalIsNotReaped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write(bytes.Repeat([]byte("x"), maxPolicySessionStartBytes+1))
	}))
	t.Cleanup(server.Close)
	backend, outcomes := reapBackend(t, server.URL, PolicySessionConfig{StartTimeout: time.Second})
	startFailsClosed(t, backend, context.Background())
	noReap(t, outcomes)
}
