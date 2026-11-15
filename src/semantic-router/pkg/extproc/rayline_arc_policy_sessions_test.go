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
	reset      map[int]bool
	starts     int
	terminates int
	// idleTimeouts are the idle timeouts the session starts asked for.
	idleTimeouts []string
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
			front.idleTimeouts = append(front.idleTimeouts, r.Header.Get("x-modal-server-session-idle-timeout"))
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
			// Only a live session's end counts.
			token := r.Header.Get("x-modal-server-session-token")
			if _, live := front.sessions[token]; live {
				delete(front.sessions, token)
				front.terminates++
			}
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
	for _, class := range []string{
		raylinearc.PolicySessionStartClass, raylinearc.PolicySessionGoneClass, raylinearc.PolicySessionExpiredClass,
		raylinearc.PolicySessionInvalidClass, raylinearc.PolicySessionUnrecognisedClass,
	} {
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
		result.RaylineARC.PolicySession, result.RaylineARC.PolicySessionCleared, result.RaylineARC.PolicySessionBinding,
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
		!moved.PolicySessionMoved || !moved.PolicySessionBinding.Replaces() {
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

func (front *fakeSessionedServer) counts() (starts, terminates int) {
	front.mu.Lock()
	defer front.mu.Unlock()
	return front.starts, front.terminates
}

// waitForTerminates waits for the session ends that run off the turn's path.
func waitForTerminates(t *testing.T, front *fakeSessionedServer, want int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		_, terminates := front.counts()
		if terminates == want {
			return
		}
		if terminates > want || time.Now().After(deadline) {
			t.Fatalf("terminates %d, want %d", terminates, want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A throwaway /v1/routes lookup commits nothing, so a session it starts is
// recorded nowhere: it is ended once the lookup is decided, or refused,
// rather than holding a container's session slot until Modal's idle timeout.
// A tracked lookup and a main turn keep theirs for the commit to persist. A
// main turn that fails without a decision ends the session it started (no
// commit can record it), never the one its episode records.
func TestPolicySessionUnrecordableTurnsEndTheirSessions(t *testing.T) {
	fixture, front := newPolicySessionFixture(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	lookup := func(throwaway bool) func(*selection.RaylineARCSelectionContext) {
		return func(arcContext *selection.RaylineARCSelectionContext) {
			arcContext.RoutesLookup, arcContext.RoutesThrowaway = true, throwaway
		}
	}
	fresh := func() *raylinearc.EpisodeState {
		state, _ := raylinearc.NewEpisodeState(2)
		return state
	}
	refuse := func(code string) {
		for _, container := range front.containers {
			container.failNext(code)
		}
	}

	decided, err := fixture.selectOnWith(t, fresh(), body, lookup(true))
	if err != nil {
		t.Fatal(err)
	}
	if decided.RaylineARC.PolicySession != nil {
		t.Fatalf("a throwaway lookup traced a session for a commit: %+v", decided.RaylineARC.PolicySession)
	}
	waitForTerminates(t, front, 1)

	refuse("selection_refused")
	if _, err := fixture.selectOnWith(t, fresh(), body, lookup(true)); err == nil {
		t.Fatal("the refused lookup was answered")
	}
	waitForTerminates(t, front, 2)
	refuse("")

	tracked, err := fixture.selectOnWith(t, fresh(), body, lookup(false))
	if err != nil || tracked.RaylineARC.PolicySession == nil {
		t.Fatalf("tracked lookup %+v, %v", tracked, err)
	}
	main, err := fixture.selectOn(t, fresh(), body)
	if err != nil || main.RaylineARC.PolicySession == nil {
		t.Fatalf("main turn %+v, %v", main, err)
	}
	time.Sleep(50 * time.Millisecond)
	if starts, terminates := front.counts(); starts != 4 || terminates != 2 {
		t.Fatalf("starts %d terminates %d, want 4 and 2: a persisting turn ended its session", starts, terminates)
	}
	if front.containerOf(tracked.RaylineARC.PolicySession.Token) == -1 ||
		front.containerOf(main.RaylineARC.PolicySession.Token) == -1 {
		t.Fatal("a session a commit would persist was ended")
	}

	// A main turn refused on a session it started: nothing commits it.
	refuse("selection_refused")
	if _, err := fixture.selectOn(t, fresh(), body); err == nil {
		t.Fatal("the refused main turn was answered")
	}
	waitForTerminates(t, front, 3)
	// One refused on the session its episode records keeps that session.
	recorded := fresh()
	recorded.PolicySession = main.RaylineARC.PolicySession
	if _, err := fixture.selectOn(t, recorded, body); err == nil {
		t.Fatal("the refused main turn was answered")
	}
	time.Sleep(50 * time.Millisecond)
	if starts, terminates := front.counts(); starts != 5 || terminates != 3 ||
		front.containerOf(main.RaylineARC.PolicySession.Token) == -1 {
		t.Fatalf("starts %d terminates %d: a failed turn ended the session its episode records", starts, terminates)
	}
}

// A main turn decided on a session it started that then aborts (its upstream
// failed) commits nothing, so it ends that session; a committed turn keeps
// its session. One that moved off the session its episode records and then
// aborts ends the session it moved to, and the episode's own stays.
func TestPolicySessionAbortedTurnEndsTheSessionItStarted(t *testing.T) {
	fixture, front := newPolicySessionFixture(t)
	store, episode := newLedgerTestStore(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})

	aborted, trace, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatal(err)
	}
	started := trace.RaylineARC.PolicySession
	if err := aborted.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	waitForSessionEnded(t, front, started.Token)
	if readLedgerTestStateNoWait(store, episode).PolicySession != nil {
		t.Fatal("an aborted turn recorded its session")
	}

	committed, _, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatal(err)
	}
	if err := committed.commit(context.Background(), &RequestContext{}); err != nil {
		t.Fatal(err)
	}
	session := readLedgerTestStateNoWait(store, episode).PolicySession
	time.Sleep(50 * time.Millisecond)
	if session == nil || front.containerOf(session.Token) == -1 {
		t.Fatal("a committed turn's session was ended")
	}

	owner := front.containerOf(session.Token)
	front.mu.Lock()
	front.dead[owner] = true
	front.mu.Unlock()
	moved, trace, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatal(err)
	}
	if !trace.RaylineARC.PolicySessionMoved {
		t.Fatalf("the turn did not move: %+v", trace.RaylineARC)
	}
	if err := moved.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	waitForSessionEnded(t, front, trace.RaylineARC.PolicySession.Token)
	time.Sleep(50 * time.Millisecond)
	if stored := readLedgerTestStateNoWait(store, episode).PolicySession; *stored != *session ||
		front.containerOf(session.Token) != owner {
		t.Fatal("an aborted turn ended the session the episode records")
	}
}

// A refusal reselect decides the turn again on a session of its own; the
// sessions of the selection the turn no longer records are ended by its
// commit, or with the rest by its abort. A rollback to the first selection
// keeps that one's session for the commit and ends the reselect's.
func TestPolicySessionReselectEndsTheSelectionItSupersedes(t *testing.T) {
	fixture, front := newPolicySessionFixture(t)
	store, episode := newLedgerTestStore(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	reselect := func(transaction *raylineARCEpisodeTransaction) *selection.RaylineARCTrace {
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOn(t, state, body)
		if err != nil {
			t.Fatal(err)
		}
		trace := result.RaylineARC
		transaction.markPolicySession(trace.PolicySession, trace.PolicySessionCleared, trace.PolicySessionBinding)
		return trace
	}

	aborted, first, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatal(err)
	}
	second := reselect(aborted)
	if err := aborted.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	waitForSessionEnded(t, front, first.RaylineARC.PolicySession.Token)
	waitForSessionEnded(t, front, second.PolicySession.Token)

	committed, kept, err := sessionTurn(t, fixture, store, episode, body)
	if err != nil {
		t.Fatal(err)
	}
	dropped := reselect(committed)
	// The rollback marks the first selection again (reserveSnapshot.restore).
	committed.markPolicySession(kept.RaylineARC.PolicySession, false, kept.RaylineARC.PolicySessionBinding)
	if err := committed.commit(context.Background(), &RequestContext{}); err != nil {
		t.Fatal(err)
	}
	waitForSessionEnded(t, front, dropped.PolicySession.Token)
	time.Sleep(50 * time.Millisecond)
	stored := readLedgerTestStateNoWait(store, episode).PolicySession
	if stored == nil || *stored != *kept.RaylineARC.PolicySession || front.containerOf(stored.Token) == -1 {
		t.Fatal("the commit did not keep the session it records")
	}
}
