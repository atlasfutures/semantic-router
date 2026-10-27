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

// placingModalServer emulates a saturated Modal Server's session start: it
// takes delay to place a session and places it whether or not the caller is
// still waiting, and a repeated start with the same x-idempotency-key waits
// for that placement and answers the session that key created.
type placingModalServer struct {
	server *httptest.Server
	delay  time.Duration
	// create false places nothing; repeat is then the repeated start's answer.
	create bool
	repeat int
	// malformed answers the first start with an empty 200, session or not.
	malformed bool
	// hold, when set, holds every repeated start until it is closed.
	hold chan struct{}
	// entered receives each first start as it arrives.
	entered chan struct{}

	mu         sync.Mutex
	byKey      map[string]string // idempotency key -> token
	placed     map[string]chan struct{}
	stamps     map[string][]string
	attempts   map[string]int
	terminated []string
}

func newPlacingModalServer(t *testing.T, delay time.Duration, create bool) *placingModalServer {
	t.Helper()
	fake := &placingModalServer{
		delay: delay, create: create, byKey: map[string]string{}, placed: map[string]chan struct{}{}, stamps: map[string][]string{}, attempts: map[string]int{},
	}
	fake.server = httptest.NewServer(http.HandlerFunc(fake.serve))
	t.Cleanup(fake.server.Close)
	return fake
}

func (fake *placingModalServer) serve(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Modal-Key") != fakeModalKey || r.Header.Get("Modal-Secret") != fakeModalSecret {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	switch r.URL.Path {
	case policySessionStartPath:
		key := r.Header.Get("x-idempotency-key")
		fake.mu.Lock()
		fake.attempts[key]++
		repeated := fake.attempts[key] > 1
		if !repeated {
			fake.placed[key] = make(chan struct{})
		}
		placed := fake.placed[key]
		fake.stamps[key] = append(fake.stamps[key], r.Header.Get("x-modal-timestamp"))
		hold, entered := fake.hold, fake.entered
		fake.mu.Unlock()
		if !repeated && entered != nil {
			entered <- struct{}{}
		}
		if repeated {
			<-placed
			if hold != nil {
				<-hold
			}
			fake.mu.Lock()
			token, found := fake.byKey[key]
			fake.mu.Unlock()
			if !found {
				w.WriteHeader(fake.repeat)
				return
			}
			fake.answer(w, key, token)
			return
		}
		// Placed regardless of the caller: a cancelled request does not
		// withdraw the start.
		time.Sleep(fake.delay)
		token := ""
		if fake.create {
			token = fmt.Sprintf("tok.%s.secret-material", key[:8])
			fake.mu.Lock()
			fake.byKey[key] = token
			fake.mu.Unlock()
		}
		close(placed)
		if token != "" && !fake.malformed {
			fake.answer(w, key, token)
		}
	case policySessionTerminatePath:
		fake.mu.Lock()
		fake.terminated = append(fake.terminated, strings.TrimPrefix(r.Header.Get("Modal-Authorization"), "Bearer "))
		fake.mu.Unlock()
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (fake *placingModalServer) answer(w http.ResponseWriter, key, token string) {
	_ = json.NewEncoder(w).Encode(map[string]string{"session_id": "sd-" + key[:8], "token": token})
}

func reapBackend(t *testing.T, fake *placingModalServer, maxReaps int) (*PolicySessionBackend, chan string) {
	t.Helper()
	client := NewPolicyServiceClient(PolicyServiceConfig{
		BaseURL: fake.server.URL, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: time.Second,
	})
	backend, err := NewPolicySessionBackend(client, PolicySessionConfig{
		StartTimeout: 100 * time.Millisecond, ReapTimeout: 5 * time.Second, MaxReaps: maxReaps,
	})
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

// A start the cell gave up on while Modal was still placing it: the reaper
// repeats it with the same idempotency key and timestamp and ends exactly the
// session Modal created for it.
func TestPolicySessionTimedOutStartIsReaped(t *testing.T) {
	fake := newPlacingModalServer(t, 300*time.Millisecond, true)
	backend, outcomes := reapBackend(t, fake, 0)
	var failure *PolicyServiceError
	if _, err := backend.Start(context.Background()); !errors.As(err, &failure) || failure.Class != PolicySessionStartClass {
		t.Fatalf("err = %v, want class %s", err, PolicySessionStartClass)
	}
	if outcome := nextReap(t, outcomes); outcome != policySessionReaped {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionReaped)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.byKey) != 1 {
		t.Fatalf("sessions created %d, want 1", len(fake.byKey))
	}
	for key, token := range fake.byKey {
		if fake.attempts[key] != 2 {
			t.Fatalf("starts with the key %d, want 2", fake.attempts[key])
		}
		if stamps := fake.stamps[key]; stamps[0] == "" || stamps[0] != stamps[1] {
			t.Fatalf("timestamps %q, want one repeated", stamps)
		}
		if len(fake.terminated) != 1 || fake.terminated[0] != token {
			t.Fatalf("terminated %d sessions, want exactly the orphan", len(fake.terminated))
		}
	}
}

// A caller that goes away mid-start is reaped the same way.
func TestPolicySessionCancelledStartIsReaped(t *testing.T) {
	fake := newPlacingModalServer(t, 300*time.Millisecond, true)
	fake.entered = make(chan struct{}, 1)
	backend, outcomes := reapBackend(t, fake, 0)
	backend.config.StartTimeout = 5 * time.Second
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-fake.entered
		cancel()
	}()
	if _, err := backend.Start(ctx); err == nil {
		t.Fatal("a cancelled start succeeded")
	}
	if outcome := nextReap(t, outcomes); outcome != policySessionReaped {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionReaped)
	}
}

// A 200 whose session could not be read is reaped too.
func TestPolicySessionMalformedStartIsReaped(t *testing.T) {
	// No delay: the first start answers at once, with an empty 200.
	fake := newPlacingModalServer(t, 0, true)
	fake.malformed = true
	backend, outcomes := reapBackend(t, fake, 0)
	if _, err := backend.Start(context.Background()); err == nil {
		t.Fatal("a start without a session succeeded")
	}
	if outcome := nextReap(t, outcomes); outcome != policySessionReaped {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionReaped)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.terminated) != 1 {
		t.Fatalf("terminated %d sessions, want 1", len(fake.terminated))
	}
}

// A key that started no session: the repeated start is refused, nothing is
// terminated.
func TestPolicySessionReapOfUncreatedStart(t *testing.T) {
	fake := newPlacingModalServer(t, 300*time.Millisecond, false)
	fake.repeat = http.StatusNotFound
	backend, outcomes := reapBackend(t, fake, 0)
	if _, err := backend.Start(context.Background()); err == nil {
		t.Fatal("a start past its timeout succeeded")
	}
	if outcome := nextReap(t, outcomes); outcome != policySessionNotCreated {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionNotCreated)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.terminated) != 0 {
		t.Fatal("a reap terminated a session no start created")
	}
}

// A 5xx on the repeated start leaves the session unknown: failed, not
// not_created.
func TestPolicySessionReapServerErrorIsFailed(t *testing.T) {
	fake := newPlacingModalServer(t, 300*time.Millisecond, false)
	fake.repeat = http.StatusServiceUnavailable
	backend, outcomes := reapBackend(t, fake, 0)
	_, _ = backend.Start(context.Background())
	if outcome := nextReap(t, outcomes); outcome != policySessionReapFailed {
		t.Fatalf("reap outcome %q, want %q", outcome, policySessionReapFailed)
	}
}

// Reaps beyond the bound are dropped, not queued: an overload spawns no more
// than MaxReaps of them.
func TestPolicySessionReapsAreBounded(t *testing.T) {
	fake := newPlacingModalServer(t, 300*time.Millisecond, true)
	fake.hold = make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(fake.hold) }) }
	t.Cleanup(release) // runs before the server closes (cleanups are LIFO)
	backend, outcomes := reapBackend(t, fake, 2)
	var starts sync.WaitGroup
	for range 5 {
		starts.Add(1)
		go func() {
			defer starts.Done()
			_, _ = backend.Start(context.Background())
		}()
	}
	starts.Wait()
	for range 3 {
		if outcome := nextReap(t, outcomes); outcome != policySessionReapDrop {
			t.Fatalf("reap outcome %q, want %q while the bound is full", outcome, policySessionReapDrop)
		}
	}
	if dropped := backend.reapDropped.Load(); dropped != 3 {
		t.Fatalf("dropped %d reaps, want 3", dropped)
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
	if len(fake.terminated) != 2 {
		t.Fatalf("terminated %d sessions, want 2", len(fake.terminated))
	}
}

// A start that answers, a session or a refusal, is never reaped.
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
	client := NewPolicyServiceClient(PolicyServiceConfig{
		BaseURL: server.URL, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: time.Second,
	})
	backend, err := NewPolicySessionBackend(client, PolicySessionConfig{StartTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	outcomes := make(chan string, 1)
	backend.onReap = func(outcome string) { outcomes <- outcome }
	if _, err := backend.Start(context.Background()); err == nil {
		t.Fatal("a refused start succeeded")
	}
	noReap(t, outcomes)
}
