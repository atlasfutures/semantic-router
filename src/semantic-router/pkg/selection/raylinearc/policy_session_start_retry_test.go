package raylinearc

import (
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"
)

// retryRecorder collects the backend's start-retry reports.
type retryRecorder struct {
	mu      sync.Mutex
	retries []PolicySessionStartRetry
}

func (recorder *retryRecorder) record(retry PolicySessionStartRetry) {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	recorder.retries = append(recorder.retries, retry)
}

func (recorder *retryRecorder) all() []PolicySessionStartRetry {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	return append([]PolicySessionStartRetry(nil), recorder.retries...)
}

func retryingSessionBackend(
	t *testing.T,
	fake *fakeModalServer,
	startTimeout time.Duration,
) (*PolicySessionBackend, *retryRecorder) {
	t.Helper()
	backend := testSessionBackend(t, fake, startTimeout)
	recorder := &retryRecorder{}
	backend.config.OnStartRetry = recorder.record
	return backend, recorder
}

// A start that hangs (Modal placed it on a container still booting) is given
// up at the start timeout and tried once more under a new idempotency key, so
// Modal opens a new session; the turn is answered there within two start
// timeouts, and the retry is reported once, as recovered.
func TestPolicySessionStartHangRetriesOnceOnANewSession(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	fake.startHangs = 1
	const startTimeout = 200 * time.Millisecond
	backend, recorder := retryingSessionBackend(t, fake, startTimeout)
	turn := backend.Bind(nil, false)

	began := time.Now()
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatalf("the retried start failed the turn: %v", err)
	}
	if elapsed := time.Since(began); elapsed < startTimeout || elapsed > 2*startTimeout+time.Second {
		t.Fatalf("the turn took %s, want one start timeout and the retry", elapsed)
	}
	session, changed := turn.Session()
	if session == nil || !changed {
		t.Fatal("the retried start recorded no session")
	}
	fake.mu.Lock()
	keys := append([]string(nil), fake.startKeys...)
	fake.mu.Unlock()
	if len(keys) != 2 || keys[0] == keys[1] || keys[0] == "" {
		t.Fatalf("start idempotency keys %q, want two distinct", keys)
	}
	retries := recorder.all()
	if len(retries) != 1 || retries[0].Cause != "timeout" || retries[0].Outcome != "recovered" {
		t.Fatalf("retries %+v, want one timeout recovered", retries)
	}
}

// A start that hangs twice fails the turn closed as session_start after two
// start timeouts, never a third attempt.
func TestPolicySessionStartHangingTwiceFailsWithinTwoTimeouts(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startHangs = 2
	const startTimeout = 150 * time.Millisecond
	backend, recorder := retryingSessionBackend(t, fake, startTimeout)

	began := time.Now()
	_, err := sessionDecide(t, backend.Bind(nil, false))
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass {
		t.Fatalf("err = %v, want class %s", err, PolicySessionStartClass)
	}
	if elapsed := time.Since(began); elapsed > 2*startTimeout+time.Second {
		t.Fatalf("the failed start took %s", elapsed)
	}
	fake.mu.Lock()
	attempts, calls := len(fake.startKeys), len(fake.calls)
	fake.mu.Unlock()
	if attempts != 2 || calls != 0 {
		t.Fatalf("start attempts %d decides %d, want 2 and 0", attempts, calls)
	}
	if retries := recorder.all(); len(retries) != 1 || retries[0].Outcome != "failed" {
		t.Fatalf("retries %+v, want one failed", retries)
	}
}

// Only a start that may succeed elsewhere is retried: a 429 or 5xx is, a
// refusal such as a 403 is not.
func TestPolicySessionStartRetriesOnlyRetryableAnswers(t *testing.T) {
	for _, tc := range []struct {
		status   int
		attempts int
	}{
		{http.StatusServiceUnavailable, 2},
		{http.StatusTooManyRequests, 2},
		{http.StatusForbidden, 1},
		{http.StatusNotFound, 1},
	} {
		fake := newFakeModalServer(t, 1)
		fake.startCode = tc.status
		backend, recorder := retryingSessionBackend(t, fake, time.Second)
		_, err := sessionDecide(t, backend.Bind(nil, false))
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass || failure.Status != tc.status {
			t.Fatalf("status %d: err = %v", tc.status, err)
		}
		fake.mu.Lock()
		attempts := len(fake.startKeys)
		fake.mu.Unlock()
		if attempts != tc.attempts {
			t.Fatalf("status %d: %d start attempts, want %d", tc.status, attempts, tc.attempts)
		}
		retries := recorder.all()
		if tc.attempts == 2 && (len(retries) != 1 || retries[0].Cause != "status" || retries[0].Status != tc.status) {
			t.Fatalf("status %d: retries %+v", tc.status, retries)
		}
		if tc.attempts == 1 && len(retries) != 0 {
			t.Fatalf("status %d: a refusal was reported as retried", tc.status)
		}
	}
}

// A lookup whose deadline leaves too little for a second start does not
// start one: the retry is reported as skipped and the first failure stands.
func TestPolicySessionStartRetryRespectsTheLookupDeadline(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startHangs = 2
	const startTimeout = 150 * time.Millisecond
	backend, recorder := retryingSessionBackend(t, fake, startTimeout)
	lookup := backend.BindLookup(nil, time.Now().Add(startTimeout+100*time.Millisecond))

	_, err := sessionDecide(t, lookup)
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass {
		t.Fatalf("err = %v, want class %s", err, PolicySessionStartClass)
	}
	fake.mu.Lock()
	attempts := len(fake.startKeys)
	fake.mu.Unlock()
	if attempts != 1 {
		t.Fatalf("%d start attempts, want 1", attempts)
	}
	if retries := recorder.all(); len(retries) != 1 || retries[0].Outcome != "skipped" {
		t.Fatalf("retries %+v, want one skipped", retries)
	}
}

// A binding that admits each decide takes the slot around the decide only:
// while its session start hangs it holds none, so a full gate does not shed
// another turn on its account.
func TestPolicySessionStartHoldsNoAdmissionSlot(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	fake.startHangs = 1
	const startTimeout = 300 * time.Millisecond
	backend, _ := retryingSessionBackend(t, fake, startTimeout)
	gate := NewAdmissionGate(1)

	var mu sync.Mutex
	acquired, heldDuringStart := 0, false
	turn := backend.Bind(nil, false)
	turn.AdmitEachDecide(func() (func(), error) {
		mu.Lock()
		acquired++
		mu.Unlock()
		return gate.Acquire()
	})
	done := make(chan error, 1)
	go func() {
		_, err := turn.Decide(t.Context(), sessionDecisionRequest(t))
		done <- err
	}()
	// Mid-hang the gate is free: another call is admitted.
	time.Sleep(startTimeout / 2)
	if gate.Inflight() != 0 {
		heldDuringStart = true
	}
	release, err := gate.Acquire()
	if err != nil {
		t.Fatalf("the hanging start holds the only slot: %v", err)
	}
	release()
	if err := <-done; err != nil {
		t.Fatalf("decide: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if heldDuringStart || acquired != 1 || gate.Inflight() != 0 || gate.HighWater() != 1 {
		t.Fatalf("held during start %t, acquired %d, inflight %d, high water %d",
			heldDuringStart, acquired, gate.Inflight(), gate.HighWater())
	}
}

// A full gate sheds the decide with the gate's error and sends nothing; the
// session the turn started is kept for the caller to end.
func TestPolicySessionAdmissionRefusalSendsNoDecide(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	backend, _ := retryingSessionBackend(t, fake, time.Second)
	gate := NewAdmissionGate(1)
	hold, err := gate.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	defer hold()
	turn := backend.Bind(nil, false)
	turn.AdmitEachDecide(gate.Acquire)
	_, err = turn.Decide(t.Context(), sessionDecisionRequest(t))
	var shed *EncoderFailure
	if !errors.As(err, &shed) || shed.Class != EncoderFailureAdmission {
		t.Fatalf("err = %v, want the gate's refusal", err)
	}
	fake.mu.Lock()
	calls := len(fake.calls)
	fake.mu.Unlock()
	if calls != 0 {
		t.Fatal("a shed decide reached the service")
	}
}
