package raylinearc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakePlacement is one start Modal is placing under an idempotency key.
type fakePlacement struct {
	done      chan struct{}
	session   PolicySession
	timestamp string
}

// keyedStart answers a start as Modal does under an idempotency key, when
// the fake is keyed: the first start under a key places one session after
// startSleep, whether or not its caller still waits; every start under the
// key answers with that session once it is placed. It reports whether it
// answered.
func (fake *fakeModalServer) keyedStart(w http.ResponseWriter, r *http.Request, proxied bool) bool {
	fake.mu.Lock()
	if !fake.keyed {
		fake.mu.Unlock()
		return false
	}
	fake.startRequests++
	if !proxied {
		fake.mu.Unlock()
		w.WriteHeader(http.StatusUnauthorized)
		return true
	}
	if fake.startCode != 0 {
		code, pad := fake.startCode, fake.startPad
		fake.mu.Unlock()
		w.WriteHeader(code)
		_, _ = w.Write([]byte(strings.Repeat("x", pad)))
		return true
	}
	key, timestamp := r.Header.Get("x-idempotency-key"), r.Header.Get("x-modal-timestamp")
	if fake.placements == nil {
		fake.placements = map[string]*fakePlacement{}
	}
	placement, found := fake.placements[key]
	if !found {
		placement = &fakePlacement{done: make(chan struct{}), timestamp: timestamp}
		fake.placements[key] = placement
		sleep := fake.startSleep
		go func() {
			time.Sleep(sleep)
			fake.mu.Lock()
			fake.starts++
			fake.nextID++
			placement.session = PolicySession{
				ID: fmt.Sprintf("sd-%04d", fake.nextID), Token: fmt.Sprintf("tok.%04d.secret-material", fake.nextID),
			}
			fake.sessions[placement.session.Token] = 0
			fake.mu.Unlock()
			close(placement.done)
		}()
	} else if placement.timestamp != timestamp {
		fake.violations = append(fake.violations, "a retried start changed its timestamp")
	}
	fake.mu.Unlock()
	select {
	case <-placement.done:
	case <-r.Context().Done():
		return true
	}
	fake.mu.Lock()
	session, garble := placement.session, fake.garbleNext
	fake.garbleNext = false
	fake.mu.Unlock()
	if garble {
		_, _ = w.Write([]byte(`{"session_id":`))
		return true
	}
	_ = json.NewEncoder(w).Encode(map[string]string{"session_id": session.ID, "token": session.Token})
	return true
}

func (fake *fakeModalServer) liveSessions() int {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return len(fake.sessions)
}

func (fake *fakeModalServer) startCounts() (requests, keys int) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return fake.startRequests, len(fake.placements)
}

// startOutcomes records a backend's OnStart outcomes.
type startOutcomes struct {
	mu   sync.Mutex
	seen []string
}

func (outcomes *startOutcomes) record(outcome string) {
	outcomes.mu.Lock()
	defer outcomes.mu.Unlock()
	outcomes.seen = append(outcomes.seen, outcome)
}

func (outcomes *startOutcomes) all() []string {
	outcomes.mu.Lock()
	defer outcomes.mu.Unlock()
	return append([]string(nil), outcomes.seen...)
}

func keyedSessionBackend(t *testing.T, fake *fakeModalServer, startTimeout time.Duration) (*PolicySessionBackend, *startOutcomes) {
	t.Helper()
	fake.mu.Lock()
	fake.keyed = true
	fake.mu.Unlock()
	backend := testSessionBackend(t, fake, startTimeout)
	outcomes := &startOutcomes{}
	backend.config.OnStart = outcomes.record
	backend.reapTimeout, backend.reapBudget = 2*time.Second, 1500*time.Millisecond
	return backend, outcomes
}

func waitForStart(t *testing.T, what string, done func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !done() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A start that lapses is asked once more under its idempotency key, and the
// retry answers with the session the first attempt was placing: the turn is
// decided there, and Modal placed one session, not two.
func TestPolicySessionStartRetryUnderItsKeyRecovers(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startSleep = 150 * time.Millisecond
	backend, outcomes := keyedSessionBackend(t, fake, 100*time.Millisecond)

	turn := backend.Bind(nil, false)
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatalf("the retried start failed: %v", err)
	}
	session, _ := turn.Session()
	requests, keys := fake.startCounts()
	starts, terminates, violations := fake.counts()
	if session == nil || requests != 2 || keys != 1 || starts != 1 || terminates != 0 || len(violations) != 0 {
		t.Fatalf("session %v, start requests %d under %d keys, %d placed, %d ended, violations %v",
			session, requests, keys, starts, terminates, violations)
	}
	if fake.liveSessions() != 1 || fake.containerOf(*session) == -1 {
		t.Fatal("the turn's session is not the one Modal placed")
	}
	if got := outcomes.all(); len(got) != 1 || got[0] != PolicySessionStartRecovered {
		t.Fatalf("outcomes %v", got)
	}
}

// A start whose retry lapses too fails the turn within two start timeouts,
// and the session Modal places for it afterwards is ended by the re-send
// under its key: no session is left holding a slot.
func TestPolicySessionStartThatLapsesTwiceFailsFastAndIsReaped(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startSleep = 400 * time.Millisecond
	backend, outcomes := keyedSessionBackend(t, fake, 100*time.Millisecond)

	began := time.Now()
	_, err := sessionDecide(t, backend.Bind(nil, false))
	elapsed := time.Since(began)
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass {
		t.Fatalf("err = %v, want class %s", err, PolicySessionStartClass)
	}
	if elapsed > 350*time.Millisecond {
		t.Fatalf("the failed start took %s, want about two start timeouts", elapsed)
	}
	waitForStart(t, "the reap", func() bool {
		_, terminates, _ := fake.counts()
		return terminates == 1
	})
	requests, keys := fake.startCounts()
	starts, _, violations := fake.counts()
	if requests != 3 || keys != 1 || starts != 1 || fake.liveSessions() != 0 || len(violations) != 0 {
		t.Fatalf("start requests %d under %d keys, %d placed, %d live, violations %v",
			requests, keys, starts, fake.liveSessions(), violations)
	}
	if got := outcomes.all(); len(got) != 2 || got[0] != PolicySessionStartFailed || got[1] != PolicySessionStartReaped {
		t.Fatalf("outcomes %v", got)
	}
}

// The retry fits inside the turn's deadline or is not made: a turn with too
// little time left fails at its first lapse, and the start is still reaped.
func TestPolicySessionStartRetryIsBoundedByTheTurn(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startSleep = 300 * time.Millisecond
	backend, outcomes := keyedSessionBackend(t, fake, 100*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	began := time.Now()
	_, err := backend.Bind(nil, false).Decide(ctx, sessionDecisionRequest(t))
	if elapsed := time.Since(began); err == nil || elapsed > 250*time.Millisecond {
		t.Fatalf("err %v after %s, want a failure at the first lapse", err, elapsed)
	}
	waitForStart(t, "the reap", func() bool { return fake.liveSessions() == 0 && len(outcomes.all()) == 2 })
	if requests, keys := fake.startCounts(); requests != 2 || keys != 1 {
		t.Fatalf("start requests %d under %d keys, want the first and the reap", requests, keys)
	}
	if got := outcomes.all(); got[0] != PolicySessionStartSkipped || got[1] != PolicySessionStartReaped {
		t.Fatalf("outcomes %v", got)
	}
}

// A start Modal answered with a refusal (a 4xx) placed nothing: it is not
// retried and not reaped. One answered 5xx by Modal's edge may still be
// placed, so it is retried, then reaped.
func TestPolicySessionRefusedStartIsNeitherRetriedNorReaped(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startCode = http.StatusTooManyRequests
	backend, outcomes := keyedSessionBackend(t, fake, time.Second)
	if _, err := sessionDecide(t, backend.Bind(nil, false)); err == nil {
		t.Fatal("a refused start was answered")
	}
	time.Sleep(50 * time.Millisecond)
	if requests, _ := fake.startCounts(); requests != 1 || len(outcomes.all()) != 0 {
		t.Fatalf("start requests %d, outcomes %v", requests, outcomes.all())
	}

	fake.mu.Lock()
	fake.startCode = http.StatusGatewayTimeout
	fake.mu.Unlock()
	if _, err := sessionDecide(t, backend.Bind(nil, false)); err == nil {
		t.Fatal("a start answered 504 was answered")
	}
	waitForStart(t, "the reap", func() bool { return len(outcomes.all()) == 2 })
	if requests, _ := fake.startCounts(); requests < 4 {
		t.Fatalf("start requests %d, want the refusal, two attempts and the reap", requests)
	}
	if got := outcomes.all(); got[0] != PolicySessionStartFailed || got[1] != PolicySessionStartReapFailed {
		t.Fatalf("outcomes %v", got)
	}
}

// A refusal whose body is past what the router reads was answered all the
// same: it is neither retried nor reaped. A caller already gone sends no
// start at all.
func TestPolicySessionStartAnsweredOrAbandonedPlacesNothing(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startCode, fake.startPad = http.StatusTooManyRequests, maxPolicySessionStartBytes+1
	backend, outcomes := keyedSessionBackend(t, fake, time.Second)
	if _, err := sessionDecide(t, backend.Bind(nil, false)); err == nil {
		t.Fatal("a refused start was answered")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := backend.Start(ctx); err == nil {
		t.Fatal("a start for a caller already gone was answered")
	}
	time.Sleep(50 * time.Millisecond)
	if requests, _ := fake.startCounts(); requests != 1 || len(outcomes.all()) != 0 {
		t.Fatalf("start requests %d, outcomes %v, want the one refusal only", requests, outcomes.all())
	}
}

// A start placed after the reap's first re-send lapsed is still reaped: the
// re-send is sent again, under the key, within the reap budget.
func TestPolicySessionLatePlacementIsStillReaped(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startSleep = 600 * time.Millisecond
	backend, outcomes := keyedSessionBackend(t, fake, 100*time.Millisecond)
	backend.reapTimeout, backend.reapBudget = 200*time.Millisecond, 3*time.Second
	if _, err := sessionDecide(t, backend.Bind(nil, false)); err == nil {
		t.Fatal("the lapsed start was answered")
	}
	waitForStart(t, "the reap", func() bool { return len(outcomes.all()) == 2 })
	if got := outcomes.all(); got[1] != PolicySessionStartReaped || fake.liveSessions() != 0 {
		t.Fatalf("outcomes %v, %d live", got, fake.liveSessions())
	}
	if requests, keys := fake.startCounts(); requests < 4 || keys != 1 {
		t.Fatalf("start requests %d under %d keys", requests, keys)
	}
}

// The re-sends in flight are bounded: a start given up on past the bound is
// counted as not reaped rather than held open.
func TestPolicySessionReapsAreBounded(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startSleep = 500 * time.Millisecond
	backend, outcomes := keyedSessionBackend(t, fake, 50*time.Millisecond)
	backend.reaps = make(chan struct{}, 1)
	for range 2 {
		if _, err := sessionDecide(t, backend.Bind(nil, false)); err == nil {
			t.Fatal("the lapsed start was answered")
		}
	}
	waitForStart(t, "the reap", func() bool { return len(outcomes.all()) == 4 })
	counted := map[string]int{}
	for _, outcome := range outcomes.all() {
		counted[outcome]++
	}
	if counted[PolicySessionStartFailed] != 2 || counted[PolicySessionStartReaped] != 1 ||
		counted[PolicySessionStartReapSkipped] != 1 {
		t.Fatalf("outcomes %v", outcomes.all())
	}
}

// A 200 whose session could not be read placed a session whose token never
// arrived: the retry under its key reads it, and the turn is decided there.
func TestPolicySessionStartWithAnUnreadableSessionIsRetried(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.garbleNext = true
	backend, outcomes := keyedSessionBackend(t, fake, time.Second)
	turn := backend.Bind(nil, false)
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatalf("the retried start failed: %v", err)
	}
	session, _ := turn.Session()
	starts, _, _ := fake.counts()
	if session == nil || starts != 1 || fake.liveSessions() != 1 {
		t.Fatalf("session %v, %d placed, %d live", session, starts, fake.liveSessions())
	}
	if got := outcomes.all(); len(got) != 1 || got[0] != PolicySessionStartRecovered {
		t.Fatalf("outcomes %v", got)
	}
}
