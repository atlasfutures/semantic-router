package raylinearc

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// fakeSessionlessService is a non-sessioned deployment of the same package:
// it answers every decide that carries the proxy key, and refuses with the
// service's own 503 while down.
type fakeSessionlessService struct {
	server *httptest.Server
	body   []byte

	mu         sync.Mutex
	down       bool
	calls      int
	violations []string
}

func newFakeSessionlessService(t *testing.T, body []byte) *fakeSessionlessService {
	t.Helper()
	fake := &fakeSessionlessService{body: body}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		if r.Header.Get("Modal-Key") != fakeModalKey || r.Header.Get("Modal-Secret") != fakeModalSecret {
			fake.violations = append(fake.violations, "decide without the proxy key")
		}
		if r.Header.Get("Modal-Authorization") != "" {
			fake.violations = append(fake.violations, "decide carried a session token")
		}
		fake.calls++
		if fake.down {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"backend_unavailable"}`))
			return
		}
		_, _ = w.Write(fake.body)
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func (fake *fakeSessionlessService) state() (int, []string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return fake.calls, append([]string(nil), fake.violations...)
}

func testFallbackBackend(
	t *testing.T,
	fake *fakeModalServer,
	sessionless *fakeSessionlessService,
	startTimeout time.Duration,
) *PolicySessionBackend {
	t.Helper()
	client := NewPolicyServiceClient(PolicyServiceConfig{
		BaseURL: fake.server.URL, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: 5 * time.Second,
	})
	config := PolicySessionConfig{IdleTimeout: 900 * time.Second, StartTimeout: startTimeout, ReplacedGrace: time.Millisecond}
	if sessionless != nil {
		config.Sessionless = NewPolicyServiceClient(PolicyServiceConfig{
			BaseURL: sessionless.server.URL, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: 5 * time.Second,
		})
	}
	backend, err := NewPolicySessionBackend(client, config)
	if err != nil {
		t.Fatal(err)
	}
	return backend
}

// A start that lapses or is refused decides the turn on the sessionless
// fallback, labelled with why; the turn records no session, so the episode's
// next turn starts one as usual, and every later call of the turn (busy
// retries, re-asks) stays on the fallback without another start.
func TestPolicySessionStartFailureFallsBackSessionless(t *testing.T) {
	cases := []struct {
		name    string
		arrange func(*fakeModalServer)
		reason  string
	}{
		{"start timeout", func(fake *fakeModalServer) { fake.startSleep = 2 * time.Second }, PolicySessionFallbackStartTimeout},
		{"start refused", func(fake *fakeModalServer) { fake.startCode = http.StatusServiceUnavailable }, PolicySessionFallbackStartRefused},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			fake := newFakeModalServer(t, 1)
			test.arrange(fake)
			sessionless := newFakeSessionlessService(t, fake.body)
			backend := testFallbackBackend(t, fake, sessionless, 100*time.Millisecond)

			turn := backend.Bind(nil, false)
			for call := 0; call < 2; call++ {
				response, err := sessionDecide(t, turn)
				if err != nil || response == nil {
					t.Fatalf("call %d: err = %v", call, err)
				}
			}
			if got := turn.Fallback(); got != test.reason {
				t.Fatalf("fallback = %q, want %q", got, test.reason)
			}
			if session, changed := turn.Session(); session != nil || changed {
				t.Fatal("a sessionless turn recorded a session")
			}
			calls, violations := sessionless.state()
			if calls != 2 || len(violations) != 0 {
				t.Fatalf("fallback calls %d violations %v", calls, violations)
			}
			fake.mu.Lock()
			starts := fake.starts
			decides := len(fake.calls)
			fake.mu.Unlock()
			if starts != 0 || decides != 0 {
				t.Fatalf("starts %d, decides on the sessioned Server %d", starts, decides)
			}
			if _, err := turn.Append(t.Context(), PolicyAppendRequest{}); err == nil {
				t.Fatal("a sessionless turn appended")
			}
		})
	}
}

// Without a fallback the start's failure is still the turn's: it fails closed
// as session_start and nothing is labelled.
func TestPolicySessionStartFailureWithoutFallbackFailsClosed(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startSleep = 2 * time.Second
	turn := testFallbackBackend(t, fake, nil, 100*time.Millisecond).Bind(nil, false)
	_, err := sessionDecide(t, turn)
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass || failure.Cause != "timeout" {
		t.Fatalf("err = %v, want class %s caused by timeout", err, PolicySessionStartClass)
	}
	if turn.Fallback() != "" {
		t.Fatal("a turn without a fallback was labelled")
	}
}

// A fallback that fails too fails the turn closed with its own failure.
func TestPolicySessionSessionlessFailureFailsClosed(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startCode = http.StatusServiceUnavailable
	sessionless := newFakeSessionlessService(t, fake.body)
	sessionless.down = true
	turn := testFallbackBackend(t, fake, sessionless, time.Second).Bind(nil, false)
	response, err := sessionDecide(t, turn)
	var failure *PolicyServiceError
	if response != nil || !errors.As(err, &failure) || failure.Status != http.StatusServiceUnavailable {
		t.Fatalf("response %v err = %v, want the fallback's 503", response, err)
	}
	if session, _ := turn.Session(); session != nil {
		t.Fatal("a failed turn recorded a session")
	}
}

// A persisted session the turn must leave (suspect, or gone) whose
// replacement cannot start falls back too, and the episode keeps the session
// it records: nothing replaced it.
func TestPolicySessionLostThenStartFailureFallsBack(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	sessionless := newFakeSessionlessService(t, fake.body)
	backend := testFallbackBackend(t, fake, sessionless, 100*time.Millisecond)
	first := backend.Bind(nil, false)
	if _, err := sessionDecide(t, first); err != nil {
		t.Fatal(err)
	}
	persisted, _ := first.Session()
	fake.expire(*persisted)
	fake.mu.Lock()
	fake.startCode = http.StatusServiceUnavailable
	fake.mu.Unlock()

	turn := backend.Bind(persisted, false)
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatal(err)
	}
	if turn.Fallback() != PolicySessionFallbackStartRefused {
		t.Fatalf("fallback = %q", turn.Fallback())
	}
	if turn.Replaces() {
		t.Fatal("a sessionless turn replaced the episode's session")
	}
	// Nil and unchanged: the commit keeps the session the episode records.
	if session, changed := turn.Session(); changed || session != nil {
		t.Fatal("a sessionless turn changed the episode's session")
	}
}

// A private (zdr) turn never falls back: its release could not reach the
// fallback container holding its episode, so it fails closed as before.
func TestPolicySessionPrivateTurnNeverFallsBack(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startCode = http.StatusServiceUnavailable
	sessionless := newFakeSessionlessService(t, fake.body)
	turn := testFallbackBackend(t, fake, sessionless, time.Second).BindPrivate()
	_, err := sessionDecide(t, turn)
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass {
		t.Fatalf("err = %v, want class %s", err, PolicySessionStartClass)
	}
	if calls, _ := sessionless.state(); calls != 0 || turn.Fallback() != "" {
		t.Fatalf("a private turn fell back (%d calls)", calls)
	}
}

// A turn that started a session, lost it (its container gone), and could not
// start another falls back without recording the lost session.
func TestPolicySessionFallbackNeverRecordsALostSession(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.afterStart = func(container int) {
		fake.dead[container] = true                    // the session's container goes away
		fake.startCode = http.StatusServiceUnavailable // and no other start is admitted
	}
	sessionless := newFakeSessionlessService(t, fake.body)
	turn := testFallbackBackend(t, fake, sessionless, time.Second).Bind(nil, false)
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatal(err)
	}
	if turn.Fallback() != PolicySessionFallbackStartRefused {
		t.Fatalf("fallback = %q", turn.Fallback())
	}
	if session, changed := turn.Session(); session != nil || changed {
		t.Fatal("a sessionless turn recorded the session it lost")
	}
	if calls, _ := sessionless.state(); calls != 1 {
		t.Fatalf("fallback calls %d", calls)
	}
}
