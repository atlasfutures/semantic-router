package extproc

import (
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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	starts     int
}

func newFakeSessionedServer(t *testing.T, containers []*fakePolicyService) *fakeSessionedServer {
	t.Helper()
	front := &fakeSessionedServer{containers: containers, sessions: map[string]int{}, dead: map[int]bool{}}
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
			delete(front.sessions, strings.TrimPrefix(r.Header.Get("Modal-Authorization"), "Bearer "))
			front.mu.Unlock()
			return
		}
		container, live := front.sessions[strings.TrimPrefix(r.Header.Get("Modal-Authorization"), "Bearer ")]
		dead := front.dead[container]
		front.mu.Unlock()
		switch {
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
	sessions, err := raylinearc.NewPolicySessionBackend(client, raylinearc.PolicySessionConfig{StartTimeout: 2 * time.Second})
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
	transaction.markPolicySession(nil, false)
	moved := &raylinearc.PolicySession{ID: "sd-2", Token: "tok.2"}
	transaction.markPolicySession(moved, false)
	next, err := transaction.nextState()
	if err != nil || next.PolicySession == nil || *next.PolicySession != *moved || next.PolicySession == moved {
		t.Fatalf("next %+v, %v", next.PolicySession, err)
	}
	// A cell that turned sessions off drops the stored session.
	off := &raylineARCEpisodeTransaction{state: state}
	off.markPolicySession(nil, true)
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
