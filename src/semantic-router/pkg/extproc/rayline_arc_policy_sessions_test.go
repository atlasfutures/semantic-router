package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fakeSessionedServer fronts policy-service fakes ("containers") the way a
// Modal Server with sticky sessions does: a proxy-authenticated session start
// places a session on a live container, and a request carrying its token is
// proxied to that container only (404 for an unknown session, 502 for a
// container that is gone).
type fakeSessionedServer struct {
	server     *httptest.Server
	containers []*fakePolicyService
	mu         sync.Mutex
	sessions   map[string]int
	dead       map[int]bool
	// reset containers drop a decide's connection without an answer.
	reset  map[int]bool
	starts int
}

func newFakeSessionedServer(t *testing.T, containers []*fakePolicyService) *fakeSessionedServer {
	t.Helper()
	front := &fakeSessionedServer{
		containers: containers, sessions: map[string]int{}, dead: map[int]bool{}, reset: map[int]bool{},
	}
	front.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		front.mu.Lock()
		if r.URL.Path == "/_modal/sessions/start" {
			defer front.mu.Unlock()
			if r.Header.Get("Modal-Key") != "wk-test" || r.Header.Get("Modal-Secret") != "ws-test" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			front.starts++
			container := front.starts % len(front.containers)
			for front.dead[container] {
				container = (container + 1) % len(front.containers)
			}
			token := fmt.Sprintf("tok.%d.secret", front.starts)
			front.sessions[token] = container
			_ = json.NewEncoder(w).Encode(map[string]string{"session_id": fmt.Sprintf("sd-%d", front.starts), "token": token})
			return
		}
		if r.URL.Path == "/_modal/sessions/terminate" {
			// Modal reads the terminate token from x-modal-server-session-token (a bearer alone is refused, 401).
			delete(front.sessions, r.Header.Get("x-modal-server-session-token"))
			front.mu.Unlock()
			return
		}
		container, live := front.sessions[strings.TrimPrefix(r.Header.Get("Modal-Authorization"), "Bearer ")]
		dead, reset := front.dead[container], front.reset[container]
		front.mu.Unlock()
		switch {
		case live && reset:
			if conn, _, err := w.(http.Hijacker).Hijack(); err == nil {
				_ = conn.Close()
			}
		case !live:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte("Unknown or expired session"))
		case dead:
			w.WriteHeader(http.StatusBadGateway)
		default:
			target, _ := url.Parse(front.containers[container].URL())
			r.Header.Del("Modal-Authorization")
			httputil.NewSingleHostReverseProxy(target).ServeHTTP(w, r)
		}
	}))
	t.Cleanup(front.server.Close)
	return front
}

func (front *fakeSessionedServer) containerOf(token string) int {
	front.mu.Lock()
	defer front.mu.Unlock()
	if container, live := front.sessions[token]; live {
		return container
	}
	return -1
}

func newPolicySessionFixture(t *testing.T) (*policySelectorFixture, *fakeSessionedServer) {
	t.Helper()
	decision := policyTestDecision("")
	catalog := policyTestActionIDs(decision.Algorithm.RaylineARC.PolicyService.Bindings)
	containers := []*fakePolicyService{
		newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog),
		newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog),
	}
	for _, container := range containers {
		container.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return catalog[0] })
	}
	front := newFakeSessionedServer(t, containers)
	client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
		BaseURL: front.server.URL, ModalKey: "wk-test", ModalSecret: "ws-test", TotalTimeout: 5 * time.Second,
	})
	sessions, err := raylinearc.NewPolicySessionBackend(client, raylinearc.PolicySessionConfig{
		StartTimeout: 2 * time.Second, ReplacedGrace: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	selector.arm(&raylineARCArmedComponents{
		scorer:         newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission:      raylinearc.NewAdmissionGate(0),
		policySessions: sessions,
	})
	return &policySelectorFixture{selector: selector, decision: decision}, front
}

// An episode starts one Modal session, keeps it through the session its
// state persists, and moves once to a new session, on another container,
// when its container is gone. The trace carries the session the commit
// persists, and only when it changed.
func TestPolicySessionsKeepAnEpisodeOnItsContainer(t *testing.T) {
	fixture, front := newPolicySessionFixture(t)
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})

	first, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatal(err)
	}
	session := first.RaylineARC.PolicySession
	if session == nil || first.RaylineARC.PolicySessionMoved || first.RaylineARC.PolicySessions != 1 {
		t.Fatalf("first turn trace %+v", first.RaylineARC)
	}
	owner := front.containerOf(session.Token)
	other := 1 - owner

	state.PolicySession = session // what the commit persists
	for turn := 0; turn < 3; turn++ {
		next, err := fixture.selectOn(t, state, body)
		if err != nil {
			t.Fatal(err)
		}
		if next.RaylineARC.PolicySession != nil || next.RaylineARC.PolicySessionMoved {
			t.Fatalf("turn %d changed the session: %+v", turn, next.RaylineARC)
		}
	}
	if got := len(front.containers[owner].received()); got != 4 {
		t.Fatalf("owner container decides = %d, want 4", got)
	}
	if len(front.containers[other].received()) != 0 || front.starts != 1 {
		t.Fatalf("other container decides %d, starts %d", len(front.containers[other].received()), front.starts)
	}

	front.mu.Lock()
	front.dead[owner] = true
	front.mu.Unlock()
	moved, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatalf("the moved turn failed: %v", err)
	}
	if moved.RaylineARC.PolicySession == nil || *moved.RaylineARC.PolicySession == *session ||
		!moved.RaylineARC.PolicySessionMoved || moved.RaylineARC.PolicySessions != 2 {
		t.Fatalf("moved trace %+v", moved.RaylineARC)
	}
	if front.containerOf(moved.RaylineARC.PolicySession.Token) != other || len(front.containers[other].received()) != 1 {
		t.Fatal("the moved turn did not land on the live container")
	}
}

// A turn's session reaches the episode's next state through the commit; a
// turn that changed none keeps the stored one.
func TestPolicySessionCommitsWithTheTurn(t *testing.T) {
	stored := &raylinearc.PolicySession{ID: "sd-1", Token: "tok.1"}
	state, _ := raylinearc.NewEpisodeState(2)
	state.PolicySession = stored
	transaction := &raylineARCEpisodeTransaction{state: state}
	kept, err := transaction.nextState()
	if err != nil || kept.PolicySession == nil || *kept.PolicySession != *stored {
		t.Fatalf("kept %+v, %v", kept.PolicySession, err)
	}
	transaction.markPolicySession(nil, false, nil)
	moved := &raylinearc.PolicySession{ID: "sd-2", Token: "tok.2"}
	transaction.markPolicySession(moved, false, nil)
	next, err := transaction.nextState()
	if err != nil || next.PolicySession == nil || *next.PolicySession != *moved || next.PolicySession == moved {
		t.Fatalf("next %+v, %v", next.PolicySession, err)
	}
	// A cell that turned sessions off drops the stored session.
	off := &raylineARCEpisodeTransaction{state: state}
	off.markPolicySession(nil, true, nil)
	cleared, err := off.nextState()
	if err != nil || cleared.PolicySession != nil {
		t.Fatalf("cleared %+v, %v", cleared.PolicySession, err)
	}
}

// A cell without sessions whose episode still holds one (sessions were turned
// off) marks it for the commit to drop; one that holds none marks nothing.
func TestPolicyWithoutSessionsDropsAStoredSession(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "hi"})
	state, _ := raylinearc.NewEpisodeState(2)
	plain, err := fixture.selectOn(t, state, body)
	if err != nil || plain.RaylineARC.PolicySessionCleared || plain.RaylineARC.PolicySession != nil {
		t.Fatalf("plain trace %+v, %v", plain.RaylineARC, err)
	}
	state.PolicySession = &raylinearc.PolicySession{ID: "sd-1", Token: "tok.1"}
	stale, err := fixture.selectOn(t, state, body)
	if err != nil || !stale.RaylineARC.PolicySessionCleared {
		t.Fatalf("stored-session trace %+v, %v", stale.RaylineARC, err)
	}
}

// A session that could not be started never reached the policy, so the
// attested-hold re-ask does not ask again on its account.
func TestPolicySessionFailuresAreNotAnswers(t *testing.T) {
	for _, class := range []string{raylinearc.PolicySessionStartClass, raylinearc.PolicySessionGoneClass, raylinearc.PolicySessionExpiredClass} {
		if policyServiceAnswered(&raylinearc.PolicyServiceError{Class: class}) {
			t.Fatalf("%s counts as an answer", class)
		}
	}
	if !policyServiceAnswered(&raylinearc.PolicyServiceError{Class: "invalid_request", Status: 400}) {
		t.Fatal("a service refusal no longer counts as an answer")
	}
}

// sessionTurn prepares a strict turn on the store's episode, decides it, and
// marks the transaction as the request path does before dispatch.
func sessionTurn(
	t *testing.T,
	fixture *policySelectorFixture,
	store *raylinearc.MemoryEpisodeStore,
	episode string,
	body []byte,
) (*raylineARCEpisodeTransaction, *selection.SelectionResult, error) {
	t.Helper()
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
	result, err := fixture.selectOn(t, state, body)
	if err != nil {
		_ = transaction.abort(context.Background(), "selection")
		return nil, nil, err
	}
	transaction.markSelection(result.RaylineARC.SelectedArm, 10)
	transaction.markPolicySession(
		result.RaylineARC.PolicySession, result.RaylineARC.PolicySessionCleared, result.RaylineARC.PolicySessionReplaced,
	)
	transaction.markPolicyState(result.RaylineARC.PolicyNextState, false)
	return transaction, result, nil
}

func waitForSessionEnded(t *testing.T, front *fakeSessionedServer, token string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for front.containerOf(token) != -1 {
		if time.Now().After(deadline) {
			t.Fatal("the replaced session was not ended after the commit")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A turn whose decide on the episode's session fails in transport moves to a
// new session within the turn and is answered there. The old session is the
// one the stored episode records, so the turn does not end it: a turn that
// moved and then aborted leaves it alone, and the commit that records the new
// one ends it. A turn that fails anyway fails as transport, logs what failed
// without a credential, and leaves the episode's session alone.
func TestPolicySessionTransportFailureMovesTheTurnAndEndsTheOldOnCommit(t *testing.T) {
	fixture, front := newPolicySessionFixture(t)
	store, episode := newLedgerTestStore(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	first, _, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.commit(context.Background(), &RequestContext{}); err != nil {
		t.Fatal(err)
	}
	session := readLedgerTestStateNoWait(store, episode).PolicySession
	if session == nil {
		t.Fatal("the first turn stored no session")
	}
	owner := front.containerOf(session.Token)
	front.mu.Lock()
	front.reset[owner] = true
	front.mu.Unlock()

	// The turn moves, then its upstream fails: nothing commits, and the
	// session the episode still records stays.
	aborted, trace, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatalf("the retried turn failed: %v", err)
	}
	if moved := trace.RaylineARC; moved.PolicySession == nil || *moved.PolicySession == *session ||
		!moved.PolicySessionMoved || moved.PolicySessionReplaced == nil {
		t.Fatalf("moved trace %+v", moved)
	}
	if err := aborted.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond) // well past the fixture's replaced grace
	if stored := readLedgerTestStateNoWait(store, episode).PolicySession; *stored != *session ||
		front.containerOf(session.Token) != owner {
		t.Fatal("an aborted turn ended the session the episode records")
	}

	// The container recovers, but the failed session is suspect on this
	// instance, so the next turn moves off it again, and commits: the stored
	// session is the new one, and only then is the old one ended.
	front.mu.Lock()
	front.reset[owner] = false
	front.mu.Unlock()
	committed, trace, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatal(err)
	}
	if err := committed.commit(context.Background(), &RequestContext{}); err != nil {
		t.Fatal(err)
	}
	moved := trace.RaylineARC.PolicySession
	if stored := readLedgerTestStateNoWait(store, episode).PolicySession; moved == nil || *stored != *moved {
		t.Fatal("the commit did not store the new session")
	}
	waitForSessionEnded(t, front, session.Token)

	// Both containers drop the decide: the one move fails too.
	front.mu.Lock()
	front.reset[0], front.reset[1] = true, true
	front.mu.Unlock()
	logs := captureLogs(t)
	_, _, err = sessionTurn(t, fixture, store, episode, body)
	if class := selectionFailureClassOf(t, err); class != "policy_service_transport" {
		t.Fatalf("class %s, want policy_service_transport", class)
	}
	time.Sleep(50 * time.Millisecond)
	if front.containerOf(moved.Token) == -1 {
		t.Fatal("a failed turn ended the episode's session")
	}
	failed := logs.FilterField(zap.String("event", "rayline_arc_policy_service_failed")).All()
	if len(failed) != 1 {
		t.Fatalf("%d policy_service_failed lines, want 1", len(failed))
	}
	fields := failed[0].ContextMap()
	cause, _ := fields["error"].(string)
	if fields["class"] != "transport" || cause == "" || strings.Contains(cause, "secret") ||
		strings.Contains(cause, front.server.URL) {
		t.Fatalf("failure fields %v", fields)
	}
}
