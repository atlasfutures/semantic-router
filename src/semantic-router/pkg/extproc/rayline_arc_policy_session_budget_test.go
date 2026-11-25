package extproc

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// scriptedSessionBudget answers each count from admit; err fails every count.
type scriptedSessionBudget struct {
	mu    sync.Mutex
	calls int
	keys  []string
	limit int
	err   error
}

func (budget *scriptedSessionBudget) AdmitPolicySessionStart(
	_ context.Context, callerKey string, limit int, window time.Duration,
) (bool, error) {
	budget.mu.Lock()
	defer budget.mu.Unlock()
	budget.calls++
	budget.keys = append(budget.keys, callerKey)
	if window != raylinearc.PolicySessionBudgetWindow {
		return false, errors.New("unexpected window")
	}
	if budget.err != nil {
		return false, budget.err
	}
	budget.limit = limit
	return budget.calls <= limit, nil
}

func (budget *scriptedSessionBudget) counted() int {
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return budget.calls
}

// newPolicySessionBudgetFixture is newPolicySessionFixture with a caller
// budget of limit new sessions per window, counted by counter.
func newPolicySessionBudgetFixture(
	t *testing.T,
	counter raylinearc.PolicySessionBudget,
	limit int,
) (*policySelectorFixture, *fakeSessionedServer) {
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
	policy := *decision.Algorithm.RaylineARC.PolicyService
	policy.Sessions = config.RaylineARCPolicySessionsConfig{Enabled: true, MaxNewSessionsPerCaller: limit}
	budget := newPolicySessionBudget(&policy, sessions, nil)
	if budget != nil && counter != nil {
		budget.counter = counter
	}
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	selector.arm(&raylineARCArmedComponents{
		scorer:         newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission:      raylinearc.NewAdmissionGate(0),
		policySessions: sessions,
		sessionBudget:  budget,
	})
	return &policySelectorFixture{selector: selector, decision: decision}, front
}

func (front *fakeSessionedServer) startCount() int {
	front.mu.Lock()
	defer front.mu.Unlock()
	return front.starts
}

func receivedDecides(front *fakeSessionedServer) []raylinearc.PolicyDecisionRequest {
	var all []raylinearc.PolicyDecisionRequest
	for _, container := range front.containers {
		all = append(all, container.received()...)
	}
	return all
}

func relaxedDecides(front *fakeSessionedServer) int {
	relaxed := 0
	for _, decide := range receivedDecides(front) {
		if decide.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed {
			relaxed++
		}
	}
	return relaxed
}

func callerTurn(caller string) func(*selection.RaylineARCSelectionContext) {
	return func(arcContext *selection.RaylineARCSelectionContext) {
		arcContext.CallerKey = raylinearc.PolicySessionCallerKey(caller)
	}
}

func budgetOutcome(outcome string) float64 {
	return testutil.ToFloat64(metrics.RaylineARCPolicySessionBudget.WithLabelValues(outcome))
}

// Under its budget a caller's new episode opens its own session, as before;
// past it the turn is decided one-shot on the shared overflow session, with
// no error and no session of its own, and further turns past it start no
// session at all. A turn on a session its episode holds is never counted.
func TestPolicySessionBudgetDecidesPastItSessionless(t *testing.T) {
	counter := &scriptedSessionBudget{}
	fixture, front := newPolicySessionBudgetFixture(t, counter, 2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	admittedBefore, overBefore := budgetOutcome("admitted"), budgetOutcome("over_budget_sessionless")

	var held *raylinearc.PolicySession
	for episode := 0; episode < 2; episode++ {
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOnWith(t, state, body, callerTurn("caller-a"))
		if err != nil {
			t.Fatalf("episode %d under the budget: %v", episode, err)
		}
		if result.RaylineARC.PolicySession == nil {
			t.Fatalf("episode %d under the budget opened no session", episode)
		}
		held = result.RaylineARC.PolicySession
	}
	if front.startCount() != 2 || counter.limit != 2 {
		t.Fatalf("starts %d limit %d", front.startCount(), counter.limit)
	}

	for episode := 0; episode < 3; episode++ {
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOnWith(t, state, body, callerTurn("caller-a"))
		if err != nil {
			t.Fatalf("over-budget turn %d failed: %v", episode, err)
		}
		if result.RaylineARC.PolicySession != nil || result.RaylineARC.PolicyStaging != nil {
			t.Fatalf("over-budget turn %d recorded a session or staged: %+v", episode, result.RaylineARC)
		}
		if got := relaxedDecides(front); got != episode+1 {
			t.Fatalf("over-budget turn %d: %d relaxed decides, want %d", episode, got, episode+1)
		}
	}
	// One overflow session for all three, never one each.
	if front.startCount() != 3 {
		t.Fatalf("starts = %d, want 2 episode sessions and 1 overflow session", front.startCount())
	}
	if front.containerOf(held.Token) == -1 {
		t.Fatal("an over-budget turn ended an episode's session")
	}

	// A turn on a session its episode holds is neither counted nor capped.
	counted := counter.counted()
	state, _ := raylinearc.NewEpisodeState(2)
	state.PolicySession = held
	result, err := fixture.selectOnWith(t, state, body, callerTurn("caller-a"))
	if err != nil || result.RaylineARC.PolicySession != nil || counter.counted() != counted {
		t.Fatalf("existing-session turn: err %v trace %+v counted %d -> %d", err, result.RaylineARC, counted, counter.counted())
	}
	if relaxedDecides(front) != 3 || len(receivedDecides(front)) != 6 {
		t.Fatal("an existing session's turn was sent relaxed")
	}
	// The window rolls over. An episode whose earlier turn was decided
	// sessionless recorded no session, so its next turn opens its own, as a
	// new episode's does, and decides on it: nothing was stranded.
	counter.mu.Lock()
	counter.calls = 0
	counter.mu.Unlock()
	overState, _ := raylinearc.NewEpisodeState(2)
	if over, err := fixture.selectOnWith(t, overState, body, callerTurn("caller-a")); err != nil || over.RaylineARC.PolicySession == nil {
		t.Fatalf("the episode's turn after the rollover: err %v", err)
	}
	if front.startCount() != 4 {
		t.Fatalf("starts = %d, want the episode's own session started", front.startCount())
	}
	if got := budgetOutcome("admitted") - admittedBefore; got != 3 {
		t.Fatalf("admitted = %v, want 2", got)
	}
	if got := budgetOutcome("over_budget_sessionless") - overBefore; got != 3 {
		t.Fatalf("over_budget_sessionless = %v, want 3", got)
	}
	for _, key := range counter.keys {
		if key != raylinearc.PolicySessionCallerKey("caller-a") {
			t.Fatalf("counted under %q, want the caller's hash", key)
		}
	}
}

// A turn past its budget on a strict cell whose service cannot serve relaxed
// is asked strict once, as a side call is, and still decided on the overflow
// session without an error.
func TestPolicySessionBudgetRelaxedUnsupportedStillDecides(t *testing.T) {
	counter := &scriptedSessionBudget{}
	fixture, _ := newPolicySessionBudgetFixture(t, counter, 0)
	// A zero limit arms no budget at all.
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "hi"})
	if _, err := fixture.selectOnWith(t, state, body, callerTurn("caller-a")); err != nil {
		t.Fatal(err)
	}
	if counter.counted() != 0 {
		t.Fatal("a cell without a budget counted a start")
	}

	fixture, front := newPolicySessionBudgetFixture(t, &scriptedSessionBudget{}, 1)
	for _, container := range front.containers {
		container.mu.Lock()
		container.relaxedUnsupported = "relaxed_unretained"
		container.mu.Unlock()
	}
	for episode := 0; episode < 2; episode++ {
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOnWith(t, state, body, callerTurn("caller-a"))
		if err != nil {
			t.Fatalf("episode %d: %v", episode, err)
		}
		if episode == 1 && result.RaylineARC.PolicySession != nil {
			t.Fatal("the over-budget turn recorded a session")
		}
	}
	if front.startCount() != 2 {
		t.Fatalf("starts = %d, want 1 episode session and 1 overflow session", front.startCount())
	}
}

// A count the store cannot give fails open: the turn opens its session.
func TestPolicySessionBudgetStoreFailureFailsOpen(t *testing.T) {
	counter := &scriptedSessionBudget{err: errors.New("ARC Redis episode session budget failed")}
	fixture, front := newPolicySessionBudgetFixture(t, counter, 1)
	before := budgetOutcome("redis_error")
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	for episode := 0; episode < 3; episode++ {
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOnWith(t, state, body, callerTurn("caller-a"))
		if err != nil || result.RaylineARC.PolicySession == nil {
			t.Fatalf("episode %d: err %v trace %+v", episode, err, result)
		}
	}
	if front.startCount() != 3 || budgetOutcome("redis_error")-before != 3 {
		t.Fatalf("starts %d redis_error %v", front.startCount(), budgetOutcome("redis_error")-before)
	}
}

// A turn without a caller (no x-rayline-caller) has no budget: never counted,
// never capped.
func TestPolicySessionBudgetNeedsACaller(t *testing.T) {
	counter := &scriptedSessionBudget{}
	fixture, front := newPolicySessionBudgetFixture(t, counter, 1)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	for episode := 0; episode < 3; episode++ {
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOnWith(t, state, body, callerTurn(""))
		if err != nil || result.RaylineARC.PolicySession == nil {
			t.Fatalf("episode %d: err %v", episode, err)
		}
	}
	if counter.counted() != 0 || front.startCount() != 3 {
		t.Fatalf("counted %d starts %d", counter.counted(), front.startCount())
	}
}

// The caller header is a trusted turn signal: read only from a configured
// gateway, and never forwarded upstream.
func TestPolicySessionCallerHeaderIsATrustedSignal(t *testing.T) {
	request := map[string]string{raylineARCCallerHeader: "caller-a"}
	trusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true}, request)
	if trusted[raylineARCCallerHeader] != "caller-a" {
		t.Fatal("a trusted gateway's caller was not read")
	}
	if untrusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{}, request); untrusted[raylineARCCallerHeader] != "" {
		t.Fatal("an untrusted caller header was read")
	}
	removed := false
	for _, header := range raylineARCTurnSignalHeadersForRemoval() {
		removed = removed || header == raylineARCCallerHeader
	}
	if !removed {
		t.Fatal("the caller header would be forwarded upstream")
	}
}

// The budget counts in the episode store when it is shared, and per process
// otherwise; a cell without sessions or a limit has none.
func TestPolicySessionBudgetCounterFollowsTheStore(t *testing.T) {
	sessions := &raylinearc.PolicySessionBackend{}
	on := &config.RaylineARCPolicyServiceConfig{Sessions: config.RaylineARCPolicySessionsConfig{Enabled: true, MaxNewSessionsPerCaller: 20}}
	if newPolicySessionBudget(on, nil, nil) != nil || newPolicySessionBudget(&config.RaylineARCPolicyServiceConfig{}, sessions, nil) != nil {
		t.Fatal("a budget without sessions or a limit")
	}
	budget := newPolicySessionBudget(on, sessions, nil)
	if budget == nil || budget.limit != 20 || budget.window != 10*time.Minute {
		t.Fatalf("budget %+v", budget)
	}
	if _, local := budget.counter.(*raylinearc.MemoryPolicySessionBudget); !local {
		t.Fatalf("counter %T, want process-local without a shared store", budget.counter)
	}
	redisStore, err := raylinearc.NewRedisEpisodeStore(raylinearc.RedisEpisodeStoreConfig{
		Address: "127.0.0.1:1", KeyPrefix: "test:", LeaseTTL: time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = redisStore.Close() })
	if _, shared := policySessionBudgetCounterFor(redisStore).(*raylinearc.RedisEpisodeStore); !shared {
		t.Fatal("a Redis episode store does not count the budget")
	}
}
