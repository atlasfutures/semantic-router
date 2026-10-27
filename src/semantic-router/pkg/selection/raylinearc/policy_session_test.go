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

const (
	fakeModalKey    = "wk-test-key"
	fakeModalSecret = "ws-test-secret"
)

// fakeModalServer emulates a Modal Server with sticky sessions as measured on
// a dev Server: a proxy-authenticated session start, token routing to one
// container, 401 without a token, 404 for an expired or ended session, 502
// for a token whose container is gone, and a terminate that needs the proxy
// key and the token.
type fakeModalServer struct {
	server *httptest.Server
	body   []byte
	stale  []byte

	mu       sync.Mutex
	sessions map[string]int // token -> container; absent = expired or ended
	dead     map[int]bool
	// hung containers hold every decide until the caller gives up: a stopped
	// container's sessions before Modal answers 502 for them (#358).
	hung       map[int]bool
	release    chan struct{} // closed at cleanup, before the server, so a hung decide returns
	staleOn    map[int]bool
	calls      map[int]int
	containers int
	starts     int
	terminates int
	nextID     int
	startSleep time.Duration
	startCode  int
	// violations records a decide that carried the proxy key, or a start
	// or terminate without the headers Modal needs.
	violations []string
}

func newFakeModalServer(t *testing.T, containers int) *fakeModalServer {
	t.Helper()
	body := readPolicyFixture(t, "decision_response.v1.json")
	fake := &fakeModalServer{
		body:       body,
		stale:      []byte(strings.Replace(string(body), fixturePackage(t).PackageSHA256, strings.Repeat("f", 64), 1)),
		sessions:   map[string]int{},
		dead:       map[int]bool{},
		hung:       map[int]bool{},
		staleOn:    map[int]bool{},
		calls:      map[int]int{},
		containers: containers,
	}
	fake.release = make(chan struct{})
	fake.server = httptest.NewServer(http.HandlerFunc(fake.serve))
	t.Cleanup(fake.server.Close)
	t.Cleanup(func() { close(fake.release) }) // runs first (cleanups are LIFO)
	return fake
}

func (fake *fakeModalServer) serve(w http.ResponseWriter, r *http.Request) {
	proxied := r.Header.Get("Modal-Key") == fakeModalKey && r.Header.Get("Modal-Secret") == fakeModalSecret
	token := strings.TrimPrefix(r.Header.Get("Modal-Authorization"), "Bearer ")
	switch r.URL.Path {
	case policySessionStartPath:
		fake.mu.Lock()
		sleep, code := fake.startSleep, fake.startCode
		fake.mu.Unlock()
		if sleep > 0 {
			select {
			case <-time.After(sleep):
			case <-r.Context().Done():
				return
			}
		}
		if !proxied {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"proxy auth required"}`))
			return
		}
		if code != 0 {
			w.WriteHeader(code)
			return
		}
		fake.mu.Lock()
		if r.Header.Get("x-modal-server-session-idle-timeout") == "" || r.Header.Get("x-idempotency-key") == "" ||
			r.Header.Get("x-modal-timestamp") == "" {
			fake.violations = append(fake.violations, "start without its headers")
		}
		fake.starts++
		fake.nextID++
		container := -1
		for offset := 0; offset < fake.containers; offset++ {
			candidate := (fake.nextID + offset) % fake.containers
			if !fake.dead[candidate] && !fake.hung[candidate] {
				container = candidate
				break
			}
		}
		id := fmt.Sprintf("sd-%04d", fake.nextID)
		issued := fmt.Sprintf("tok.%04d.secret-material", fake.nextID)
		fake.sessions[issued] = container
		fake.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]string{"session_id": id, "token": issued})
	case policySessionTerminatePath:
		fake.mu.Lock()
		defer fake.mu.Unlock()
		if !proxied || token == "" {
			fake.violations = append(fake.violations, "terminate without the proxy key and the token")
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		fake.terminates++
		delete(fake.sessions, token)
		w.WriteHeader(http.StatusOK)
	default:
		fake.mu.Lock()
		if r.Header.Get("Modal-Key") != "" {
			fake.violations = append(fake.violations, "decide carried the proxy key")
		}
		if token == "" {
			fake.mu.Unlock()
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"session token required: start a server session with POST /_modal/sessions/start"}`))
			return
		}
		container, live := fake.sessions[token]
		if !live {
			fake.mu.Unlock()
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte("Unknown or expired session"))
			return
		}
		if fake.hung[container] {
			fake.mu.Unlock()
			select {
			case <-r.Context().Done():
			case <-fake.release:
			}
			return
		}
		if fake.dead[container] {
			fake.mu.Unlock()
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		fake.calls[container]++
		stale := fake.staleOn[container]
		fake.mu.Unlock()
		if stale {
			_, _ = w.Write(fake.stale)
			return
		}
		_, _ = w.Write(fake.body)
	}
}

func (fake *fakeModalServer) containerOf(session PolicySession) int {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	container, live := fake.sessions[session.Token]
	if !live {
		return -1
	}
	return container
}

func (fake *fakeModalServer) kill(container int) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.dead[container] = true
}

func (fake *fakeModalServer) expire(session PolicySession) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	delete(fake.sessions, session.Token)
}

func (fake *fakeModalServer) counts() (starts, terminates int, violations []string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return fake.starts, fake.terminates, append([]string(nil), fake.violations...)
}

func testSessionBackend(t *testing.T, fake *fakeModalServer, startTimeout time.Duration) *PolicySessionBackend {
	t.Helper()
	client := NewPolicyServiceClient(PolicyServiceConfig{
		BaseURL: fake.server.URL, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: 5 * time.Second,
	})
	backend, err := NewPolicySessionBackend(client, PolicySessionConfig{IdleTimeout: 900 * time.Second, StartTimeout: startTimeout})
	if err != nil {
		t.Fatal(err)
	}
	return backend
}

func sessionDecide(t *testing.T, binding *PolicySessionBinding) (*PolicyDecisionResponse, error) {
	t.Helper()
	return binding.Decide(context.Background(), sessionDecisionRequest(t))
}

func sessionDecisionRequest(t *testing.T) PolicyDecisionRequest {
	t.Helper()
	return PolicyDecisionRequest{
		SchemaVersion: PolicyDecisionRequestSchema, EpisodeIDHash: episodeHash("a"), Package: fixturePackage(t),
	}
}

// A new episode starts one session; every call of the turn (busy retries)
// and every later turn, through the persisted session, reaches its one
// container, with the token and never the proxy key.
func TestPolicySessionStartsOnceAndSticks(t *testing.T) {
	fake := newFakeModalServer(t, 3)
	backend := testSessionBackend(t, fake, time.Second)

	turn := backend.Bind(nil, false)
	for call := 0; call < 3; call++ {
		if _, err := sessionDecide(t, turn); err != nil {
			t.Fatal(err)
		}
	}
	session, changed := turn.Session()
	if session == nil || !changed {
		t.Fatal("a new episode's turn records no session")
	}
	if used, moved := turn.Sessions(); used != 1 || moved {
		t.Fatalf("sessions %d moved %t, want 1 and no move", used, moved)
	}
	container := fake.containerOf(*session)
	for next := 0; next < 4; next++ {
		later := backend.Bind(session, false)
		if _, err := sessionDecide(t, later); err != nil {
			t.Fatal(err)
		}
		if again, changed := later.Session(); changed || again == nil || *again != *session {
			t.Fatalf("turn %d changed the session", next)
		}
	}
	starts, _, violations := fake.counts()
	if starts != 1 || len(violations) != 0 {
		t.Fatalf("starts %d violations %v", starts, violations)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.calls[container] != 7 || len(fake.calls) != 1 {
		t.Fatalf("calls per container %v, want 7 on container %d", fake.calls, container)
	}
}

// A start that does not answer within the start timeout fails the turn
// closed: no decide is sent and no session is recorded.
func TestPolicySessionStartTimeoutFailsClosed(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startSleep = 2 * time.Second
	backend := testSessionBackend(t, fake, 100*time.Millisecond)
	turn := backend.Bind(nil, false)
	began := time.Now()
	_, err := sessionDecide(t, turn)
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass {
		t.Fatalf("err = %v, want class %s", err, PolicySessionStartClass)
	}
	if elapsed := time.Since(began); elapsed > time.Second {
		t.Fatalf("the start waited %s", elapsed)
	}
	if session, _ := turn.Session(); session != nil {
		t.Fatal("a failed start recorded a session")
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.calls) != 0 {
		t.Fatal("a decide was sent without a session")
	}
}

// A refused start (no capacity, bad proxy key) fails the turn as
// session_start too.
func TestPolicySessionRefusedStartFailsClosed(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	fake.startCode = http.StatusGatewayTimeout
	_, err := sessionDecide(t, testSessionBackend(t, fake, time.Second).Bind(nil, false))
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != PolicySessionStartClass || failure.Status != http.StatusGatewayTimeout {
		t.Fatalf("err = %v", err)
	}
}

// A token whose container is gone (502) moves the episode to a new session
// once, and the turn is answered there; a second loss in the same turn is not
// moved again.
func TestPolicySessionLostContainerMovesOnce(t *testing.T) {
	fake := newFakeModalServer(t, 3)
	backend := testSessionBackend(t, fake, time.Second)
	first := backend.Bind(nil, false)
	if _, err := sessionDecide(t, first); err != nil {
		t.Fatal(err)
	}
	persisted, _ := first.Session()
	fake.kill(fake.containerOf(*persisted))

	turn := backend.Bind(persisted, false)
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatalf("the moved turn failed: %v", err)
	}
	moved, changed := turn.Session()
	if !changed || moved == nil || *moved == *persisted {
		t.Fatal("the turn did not record its new session")
	}
	if used, wasMoved := turn.Sessions(); used != 2 || !wasMoved {
		t.Fatalf("sessions %d moved %t", used, wasMoved)
	}

	// The new container dies too, within the same turn (a busy retry): the
	// turn has moved once already, so it fails.
	fake.kill(fake.containerOf(*moved))
	_, err := sessionDecide(t, turn)
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Status != http.StatusBadGateway {
		t.Fatalf("second loss err = %v, want the 502", err)
	}
	if starts, _, _ := fake.counts(); starts != 2 {
		t.Fatalf("starts = %d, want 2 (no second move)", starts)
	}
}

// An idle-expired session (404 "Unknown or expired session") starts a new
// one; the service's own 404 (package_not_loaded, a JSON refusal) does not.
func TestPolicySessionExpiredStartsANewOne(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	backend := testSessionBackend(t, fake, time.Second)
	first := backend.Bind(nil, false)
	if _, err := sessionDecide(t, first); err != nil {
		t.Fatal(err)
	}
	persisted, _ := first.Session()
	fake.expire(*persisted)
	turn := backend.Bind(persisted, false)
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatal(err)
	}
	if _, moved := turn.Sessions(); !moved {
		t.Fatal("an expired session was not replaced")
	}
	if policySessionLost(&PolicyServiceError{Class: "package_not_loaded", Status: http.StatusNotFound}) {
		t.Fatal("the service's package_not_loaded counts as a lost session")
	}
	if policySessionLost(&PolicyServiceError{Class: "timeout"}) || policySessionLost(&PolicyServiceError{Class: "transport"}) {
		t.Fatal("a timeout or transport failure counts as a lost session")
	}
}

// A session that answers for another package (a container still on the
// previous build after a rollover) is ended with the proxy key and the
// token, and the turn is retried once on a new session.
func TestPolicySessionStaleBuildTerminatesAndRestarts(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	backend := testSessionBackend(t, fake, time.Second)
	first := backend.Bind(nil, false)
	if _, err := sessionDecide(t, first); err != nil {
		t.Fatal(err)
	}
	persisted, _ := first.Session()
	old := fake.containerOf(*persisted)
	fake.mu.Lock()
	fake.staleOn[old] = true
	fake.mu.Unlock()

	turn := backend.Bind(persisted, false)
	response, err := sessionDecide(t, turn)
	if err != nil || response.Package != fixturePackage(t) {
		t.Fatalf("the restarted turn answered %+v, %v", response, err)
	}
	if fake.containerOf(*persisted) != -1 {
		t.Fatal("the stale session was not ended")
	}
	moved, _ := turn.Session()
	if fake.containerOf(*moved) == old {
		t.Fatal("the new session landed on the stale container")
	}
	_, terminates, violations := fake.counts()
	if terminates != 1 || len(violations) != 0 {
		t.Fatalf("terminates %d violations %v", terminates, violations)
	}
}

// A side call's binding never records a session. It uses the episode's
// session when there is one; a session it had to start is ended by Release.
func TestPolicySessionSideCallIsEphemeral(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	backend := testSessionBackend(t, fake, time.Second)

	side := backend.Bind(nil, true)
	if _, err := sessionDecide(t, side); err != nil {
		t.Fatal(err)
	}
	if session, changed := side.Session(); session != nil || changed {
		t.Fatal("a side call recorded a session")
	}
	side.Release(context.Background())
	if _, terminates, _ := fake.counts(); terminates != 1 {
		t.Fatalf("terminates = %d, want the side call's own session ended", terminates)
	}

	main := backend.Bind(nil, false)
	if _, err := sessionDecide(t, main); err != nil {
		t.Fatal(err)
	}
	persisted, _ := main.Session()
	onEpisode := backend.Bind(persisted, true)
	if _, err := sessionDecide(t, onEpisode); err != nil {
		t.Fatal(err)
	}
	onEpisode.Release(context.Background())
	if fake.containerOf(*persisted) == -1 {
		t.Fatal("a side call ended the episode's session")
	}
	if starts, terminates, _ := fake.counts(); starts != 2 || terminates != 1 {
		t.Fatalf("starts %d terminates %d", starts, terminates)
	}
}

// Readiness goes through a session of its own, which it ends.
func TestPolicySessionProbeUsesAndEndsASession(t *testing.T) {
	fake := newFakeModalServer(t, 1)
	backend := testSessionBackend(t, fake, time.Second)
	var sawToken bool
	err := backend.Probe(context.Background(), func(_ context.Context, client *PolicyServiceClient) error {
		sawToken = client.config.SessionToken != ""
		return nil
	})
	if err != nil || !sawToken {
		t.Fatalf("probe err %v, token %t", err, sawToken)
	}
	if starts, terminates, _ := fake.counts(); starts != 1 || terminates != 1 {
		t.Fatalf("starts %d terminates %d", starts, terminates)
	}
}

// Sessions need the proxy key; a session's token never prints.
func TestPolicySessionNeedsTheProxyKeyAndRedactsItsToken(t *testing.T) {
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: "https://example.invalid", TotalTimeout: time.Second})
	if _, err := NewPolicySessionBackend(client, PolicySessionConfig{}); err == nil {
		t.Fatal("a backend without the proxy key was accepted")
	}
	session := PolicySession{ID: "sd-1", Token: "tok.secret-material"}
	encoded, _ := json.Marshal(struct{ Session PolicySession }{session})
	for _, rendered := range []string{
		fmt.Sprintf("%v %+v %#v %s", session, session, session, session),
		fmt.Sprintf("%v %+v", &session, struct{ S *PolicySession }{&session}),
		string(encoded),
	} {
		if strings.Contains(rendered, "secret-material") {
			t.Fatalf("token printed: %s", rendered)
		}
	}
	for _, bad := range []PolicySession{{ID: "", Token: "t"}, {ID: "sd", Token: ""}, {ID: "sd", Token: "a b"},
		{ID: "sd", Token: strings.Repeat("t", maxPolicySessionTokenBytes+1)}} {
		if ValidatePolicySession(bad) == nil {
			t.Fatalf("invalid session accepted: id %q, token %d bytes", bad.ID, len(bad.Token))
		}
	}
}

// The episode record keeps its earlier bytes without a session, carries one
// as v10, and refuses a v10 label without one or a session under an older
// label.
func TestEpisodeStateCarriesAPolicySessionAsV10(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	state, _ := NewEpisodeState(2)
	state.Policy = &PolicyEpisodeState{}
	before, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(before, []byte(episodeStateSchemaV10)) || bytes.Contains(before, []byte("policy_session")) {
		t.Fatalf("a record without a session changed: %s", before)
	}
	state.PolicySession = &PolicySession{ID: "sd-01M5", Token: "tok.secret-material"}
	v10, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(v10, []byte(episodeStateSchemaV10)) {
		t.Fatalf("a record with a session is not v10: %s", v10)
	}
	restored, _, err := unmarshalEpisodeState(v10, 2, now)
	if err != nil {
		t.Fatal(err)
	}
	if restored.PolicySession == nil || *restored.PolicySession != *state.PolicySession {
		t.Fatal("the session did not survive the round trip")
	}
	if cloned := cloneEpisodeState(restored); cloned.PolicySession == restored.PolicySession ||
		*cloned.PolicySession != *restored.PolicySession {
		t.Fatal("cloning shares or drops the session")
	}
	if old, _, err := unmarshalEpisodeState(before, 2, now); err != nil || old.PolicySession != nil {
		t.Fatalf("an older record decodes as %+v, %v", old, err)
	}
	for name, payload := range map[string][]byte{
		"v10 labelled v3":  bytes.Replace(v10, []byte(episodeStateSchemaV10), []byte(episodeStateSchema), 1),
		"v3 labelled v10":  bytes.Replace(before, []byte(episodeStateSchema), []byte(episodeStateSchemaV10), 1),
		"empty token":      bytes.Replace(v10, []byte(`"token":"tok.secret-material"`), []byte(`"token":""`), 1),
		"unknown member":   bytes.Replace(v10, []byte(`"id":"sd-01M5"`), []byte(`"id":"sd-01M5","extra":1`), 1),
		"whitespace in id": bytes.Replace(v10, []byte(`"id":"sd-01M5"`), []byte(`"id":"sd 01M5"`), 1),
	} {
		if _, _, err := unmarshalEpisodeState(payload, 2, now); err == nil {
			t.Fatalf("%s was accepted", name)
		}
	}
}

// Only Modal's verified signatures count as a lost session; any other 502 or
// 404 keeps its usual class and fails the turn closed.
func TestPolicySessionLossNeedsModalsSignature(t *testing.T) {
	for _, tc := range []struct {
		status int
		body   string
		want   string
	}{
		{http.StatusBadGateway, "", PolicySessionGoneClass},
		{http.StatusBadGateway, "upstream connect error", ""},
		{http.StatusNotFound, "Unknown or expired session", PolicySessionExpiredClass},
		{http.StatusNotFound, "not found", ""},
		{http.StatusNotFound, `{"error":"package_not_loaded","detail":{"message":"Unknown or expired session"}}`, ""},
		{http.StatusNotFound, "Unknown or expired session\n", PolicySessionExpiredClass},
		{http.StatusUnauthorized, `{"error":"session token required"}`, ""},
		{http.StatusServiceUnavailable, "", ""},
	} {
		if got := policySessionFailureClass(tc.status, []byte(tc.body)); got != tc.want {
			t.Fatalf("%d %q classed %q, want %q", tc.status, tc.body, got, tc.want)
		}
	}
	if !PolicySessionUnanswered(&PolicyServiceError{Class: PolicySessionStartClass}) ||
		!PolicySessionUnanswered(&PolicyServiceError{Class: PolicySessionGoneClass, Status: http.StatusBadGateway}) ||
		PolicySessionUnanswered(&PolicyServiceError{Class: "selection_refused", Status: http.StatusUnprocessableEntity}) {
		t.Fatal("PolicySessionUnanswered misclassifies")
	}
	// Without a session token the client classes Modal-looking answers as
	// before: a base_url cell is unchanged.
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer plain.Close()
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: plain.URL, TotalTimeout: time.Second})
	_, err := client.Decide(context.Background(), PolicyDecisionRequest{SchemaVersion: PolicyDecisionRequestSchema, EpisodeIDHash: episodeHash("b"), Package: fixturePackage(t)})
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != "status" || failure.Status != http.StatusBadGateway {
		t.Fatalf("base_url 502 classed %v", err)
	}
}

// A redirect is never followed with a session token or the proxy key, so
// neither reaches another origin.
func TestPolicySessionNeverFollowsARedirect(t *testing.T) {
	var leaked sync.Map
	elsewhere := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Modal-Authorization") != "" || r.Header.Get("Modal-Key") != "" {
			leaked.Store(r.URL.Path, true)
		}
	}))
	defer elsewhere.Close()
	fake := newFakeModalServer(t, 1)
	redirecting := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == policySessionStartPath {
			fake.serve(w, r)
			return
		}
		http.Redirect(w, r, elsewhere.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirecting.Close()
	client := NewPolicyServiceClient(PolicyServiceConfig{
		BaseURL: redirecting.URL, ModalKey: fakeModalKey, ModalSecret: fakeModalSecret, TotalTimeout: time.Second,
	})
	backend, err := NewPolicySessionBackend(client, PolicySessionConfig{StartTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	turn := backend.Bind(nil, false)
	if _, err := sessionDecide(t, turn); err == nil {
		t.Fatal("a redirected decide succeeded")
	}
	session, _ := turn.Session()
	if session != nil {
		_ = backend.Terminate(context.Background(), *session)
	}
	leaked.Range(func(path, _ any) bool {
		t.Fatalf("a credential followed the redirect to %v", path)
		return true
	})
}

// A container that stops answering (its decides time out, no 502 yet): that
// turn fails closed and asks nothing twice, but the episode's next turn moves
// to a new session at once instead of timing out again (#358).
func TestPolicySessionTimeoutMakesTheNextTurnMove(t *testing.T) {
	fake := newFakeModalServer(t, 3)
	backend := testSessionBackend(t, fake, time.Second)
	first := backend.Bind(nil, false)
	if _, err := sessionDecide(t, first); err != nil {
		t.Fatal(err)
	}
	persisted, _ := first.Session()
	fake.mu.Lock()
	fake.hung[fake.sessions[persisted.Token]] = true
	fake.mu.Unlock()

	timedOut := backend.Bind(persisted, false)
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	_, err := timedOut.Decide(ctx, sessionDecisionRequest(t))
	cancel()
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || (failure.Class != "timeout" && failure.Class != "transport") {
		t.Fatalf("hung decide err = %v, want a timeout", err)
	}
	if starts, _, _ := fake.counts(); starts != 1 {
		t.Fatalf("starts = %d: the timed-out turn must not move (fail closed)", starts)
	}

	next := backend.Bind(persisted, false)
	if _, err := sessionDecide(t, next); err != nil {
		t.Fatalf("the next turn did not move: %v", err)
	}
	moved, changed := next.Session()
	if !changed || moved == nil || *moved == *persisted {
		t.Fatal("the next turn did not record its new session")
	}
	if _, wasMoved := next.Sessions(); !wasMoved {
		t.Fatal("the next turn did not report its move")
	}
	deadline := time.Now().Add(2 * time.Second) // the old session is ended off the turn's path
	for {
		starts, terminates, _ := fake.counts()
		if starts == 2 && terminates == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("starts %d terminates %d, want 2 and 1 (the suspect session ended)", starts, terminates)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A caller that goes away (a client disconnect) says nothing about the
// container: the session stays the episode's.
func TestPolicySessionCallerCancelIsNotSuspect(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	backend := testSessionBackend(t, fake, time.Second)
	first := backend.Bind(nil, false)
	if _, err := sessionDecide(t, first); err != nil {
		t.Fatal(err)
	}
	persisted, _ := first.Session()
	fake.mu.Lock()
	fake.hung[fake.sessions[persisted.Token]] = true
	fake.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(100 * time.Millisecond); cancel() }()
	_, _ = backend.Bind(persisted, false).Decide(ctx, sessionDecisionRequest(t))
	if backend.suspect.has(*persisted) {
		t.Fatal("a caller cancel marked the session suspect")
	}
}

// A suspect mark lasts the session's idle timeout; then Modal has ended the
// session anyway (a turn on it gets the 404 and moves).
func TestPolicySessionSuspectMarkExpires(t *testing.T) {
	clock := time.Date(2026, 10, 27, 7, 0, 0, 0, time.UTC)
	suspect := newSuspectSessions(time.Minute, func() time.Time { return clock })
	session := PolicySession{ID: "sd-1", Token: "tok.1"}
	suspect.mark(session)
	if !suspect.has(session) {
		t.Fatal("not suspect right after the mark")
	}
	clock = clock.Add(time.Minute)
	if suspect.has(session) {
		t.Fatal("still suspect after its idle timeout")
	}
	// Marking prunes the expired entries, so the set stays bounded.
	suspect.mark(PolicySession{ID: "sd-2", Token: "tok.2"})
	clock = clock.Add(2 * time.Minute)
	suspect.mark(PolicySession{ID: "sd-3", Token: "tok.3"})
	if n := len(suspect.until); n != 1 {
		t.Fatalf("%d entries after marking past the others' expiry, want 1", n)
	}
	var none *suspectSessions
	none.mark(session)
	if none.has(session) {
		t.Fatal("a nil set reports a suspect")
	}
}
