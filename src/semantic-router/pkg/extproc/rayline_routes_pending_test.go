package extproc

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// pendingRoutesFixture is a policy-service cell serving /v1/routes through the
// real policy selection, a fake policy service and a memory episode store.
type pendingRoutesFixture struct {
	router  *OpenAIRouter
	policy  *policySelectorFixture
	store   *raylinearc.MemoryEpisodeStore
	actions []string
	// picks are the actions the fake service answers, in order; the last
	// repeats.
	picks []string
}

func newPendingRoutesFixture(t *testing.T, episodeWrites bool, consistency string) *pendingRoutesFixture {
	t.Helper()
	policy := newPolicySelectorFixtureWith(t, consistency)
	decision := *policy.decision
	arc := *decision.Algorithm.RaylineARC
	arc.Episode = config.RaylineARCEpisodeConfig{
		IDHeader: decisionServiceEpisodeHeader, AcquireTimeoutSeconds: 5, LeaseTTLSeconds: 30, IdleTTLSeconds: 60,
		Consistency: consistency,
	}
	arc.RoutesAPI = config.RaylineARCRoutesAPIConfig{
		Enabled: true, EpisodeWrites: episodeWrites,
	}
	algorithm := *decision.Algorithm
	algorithm.RaylineARC = &arc
	decision.Algorithm = &algorithm
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 16, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{decision}
	cfg.AutoModelNames = []string{"rayline/arc-dev"}
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineARC, policy.selector)
	fixture := &pendingRoutesFixture{
		router:  &OpenAIRouter{Config: cfg, ModelSelector: registry, RaylineARCEpisodeStore: store},
		policy:  policy,
		store:   store,
		actions: policyTestActionIDs(arc.PolicyService.Bindings),
	}
	policy.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string {
		fixture.policy.fake.mu.Lock()
		served := len(fixture.policy.fake.bodies)
		fixture.policy.fake.mu.Unlock()
		if len(fixture.picks) == 0 {
			return fixture.actions[0]
		}
		return fixture.picks[min(served-1, len(fixture.picks)-1)]
	})
	return fixture
}

// lookup posts one /v1/routes call on conversation and returns its answer.
func (fixture *pendingRoutesFixture) lookup(t *testing.T, conversation string, body []byte) raylineRoutesResponse {
	t.Helper()
	headers := map[string]string{"x-rayline-format": "anthropic"}
	if conversation != "" {
		headers[raylineRoutesSessionHeader] = conversation
	}
	response := fixture.router.handleRaylineRoutesAPI(body, routesContext(headers))
	fixture.router.background.Wait()
	if code := immediateStatusCode(t, response); code != 200 {
		t.Fatalf("lookup status %d: %s", code, response.GetImmediateResponse().Body)
	}
	var payload raylineRoutesResponse
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &payload); err != nil {
		t.Fatal(err)
	}
	return payload
}

// stored is the episode record the conversation's lookups wrote, nil when
// none was written.
func (fixture *pendingRoutesFixture) stored(t *testing.T, conversation string) *raylinearc.EpisodeState {
	t.Helper()
	state, read, err := fixture.store.Snapshot(context.Background(), raylinearc.HashEpisodeID(conversation), 2)
	if err != nil {
		t.Fatal(err)
	}
	if read.Version() == 0 {
		return nil
	}
	return state
}

func pendingRoutesBody(t *testing.T, messages ...map[string]any) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{"model": "rayline/arc-dev", "max_tokens": 64, "system": "be brief", "messages": messages})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

var (
	pendingTurnTask   = map[string]any{"role": "user", "content": "fix the bug"}
	pendingTurnReply  = map[string]any{"role": "assistant", "content": "fixed"}
	pendingTurnNext   = map[string]any{"role": "user", "content": "now add a test"}
	pendingTurnEdited = map[string]any{"role": "user", "content": "now add two tests"}
)

// The sequence lookup, the same lookup again, the next turn's lookup and an
// edited repeat of it, with the record read after each call: a lookup
// advances nothing; a repeat replaces the decision and the policy sees none
// of it; the next turn records the last decision as inferred and stores its
// own; an edited repeat breaks nothing. Strict and relaxed episodes alike.
func TestRoutesLookupsKeepTheirDecisionPending(t *testing.T) {
	for _, consistency := range []string{config.RaylineARCConsistencyStrict, config.RaylineARCConsistencyRelaxed} {
		t.Run(consistency, func(t *testing.T) { testRoutesLookupsKeepTheirDecisionPending(t, consistency) })
	}
}

func testRoutesLookupsKeepTheirDecisionPending(t *testing.T, consistency string) {
	fixture := newPendingRoutesFixture(t, true, consistency)
	fixture.picks = []string{fixture.actions[0], fixture.actions[2], fixture.actions[1], fixture.actions[0]}
	const conversation = "conversation-pending"

	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	state := fixture.stored(t, conversation)
	if state == nil || state.Pending == nil || state.TurnIndex != 0 || state.PreviousArm != nil || state.Policy != nil {
		t.Fatalf("after the first lookup the record is %+v", state)
	}
	if state.Pending.RouteID != first.RouteID || state.Pending.PrefixLen != 1 || state.Pending.Policy.Entry.ActionID != fixture.actions[0] {
		t.Fatalf("pending %+v for route %s", state.Pending, first.RouteID)
	}
	if first.Episode == nil || first.Episode.TurnIndex != 1 {
		t.Fatalf("first episode %+v, want turn 1", first.Episode)
	}

	repeat := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	state = fixture.stored(t, conversation)
	if state.TurnIndex != 0 || state.PreviousArm != nil || state.Policy != nil || state.PolicyBoundary != nil ||
		state.Pending.RouteID != repeat.RouteID || state.Pending.Policy.Entry.ActionID != fixture.actions[2] {
		t.Fatalf("after the repeat the record is %+v, pending %+v", state, state.Pending)
	}
	if repeat.Episode == nil || repeat.Episode.TurnIndex != 1 || repeat.Episode.Stayed {
		t.Fatalf("repeat episode %+v, want turn 1 and no stay", repeat.Episode)
	}

	next := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	state = fixture.stored(t, conversation)
	if state.TurnIndex != 1 || state.PreviousArm == nil || len(state.Policy.Ledger) != 1 {
		t.Fatalf("after the next turn the record is %+v", state)
	}
	if entry := state.Policy.Ledger[0]; entry.Message != 1 || entry.ActionID != fixture.actions[2] || entry.Basis != raylinearc.PolicyLedgerBasisInferred {
		t.Fatalf("inferred entry %+v, want the repeat's decision", entry)
	}
	if state.Pending.RouteID != next.RouteID || state.Pending.PrefixLen != 3 || state.Pending.TurnIndex != 1 {
		t.Fatalf("next pending %+v", state.Pending)
	}
	if next.Episode == nil || next.Episode.TurnIndex != 2 {
		t.Fatalf("next episode %+v, want turn 2", next.Episode)
	}

	edited := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnEdited))
	state = fixture.stored(t, conversation)
	if state.TurnIndex != 1 || len(state.Policy.Ledger) != 1 || state.Pending.RouteID != edited.RouteID {
		t.Fatalf("after the edited repeat the record is %+v", state)
	}

	sent := fixture.policy.fake.received()
	if len(sent) != 4 {
		t.Fatalf("decides %d, want 4", len(sent))
	}
	// The repeat is decided on the executed episode: no attribution for the
	// first lookup's pick, the same context.
	if len(sent[1].Attribution) != 0 || sent[1].ContextEpoch != sent[0].ContextEpoch {
		t.Fatalf("repeat decide attribution %+v epoch %q", sent[1].Attribution, sent[1].ContextEpoch)
	}
	// The next turn attributes the reply to the decision that ran.
	if len(sent[2].Attribution) != 1 || sent[2].Attribution[0].Message != 1 || sent[2].Attribution[0].ActionID != fixture.actions[2] {
		t.Fatalf("next decide attribution %+v", sent[2].Attribution)
	}
	// The edited repeat extends the recorded turn: same context, same
	// attribution, no prefix break.
	if sent[3].ContextEpoch != sent[2].ContextEpoch || len(sent[3].Attribution) != 1 {
		t.Fatalf("edited decide epoch %q attribution %+v", sent[3].ContextEpoch, sent[3].Attribution)
	}
}

// A lookup carrying routing.execution decides on the caller's history: on a
// policy-service cell with episode writes on it is decided ephemerally and
// writes no episode, pending decision included. A control lookup without it
// on the same cell stores one.
func TestRoutesExecutionLookupWritesNoPendingDecision(t *testing.T) {
	vectors := readExecutionVectors(t)
	for _, execution := range []bool{true, false} {
		router, selectors := oneCellRouter(t, "main")
		router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}
		body := oneCellBody("arc-1.0", false)
		if execution {
			body = executionBody(string(vectors.Vectors[0].Execution))
		}
		headers := map[string]string{raylineRoutesSessionHeader: "conversation-execution"}
		response := router.handleRaylineRoutesAPI(body, routesContext(headers))
		router.background.Wait()
		if code := immediateStatusCode(t, response); code != 200 {
			t.Fatalf("execution=%t: status %d: %s", execution, code, response.GetImmediateResponse().Body)
		}
		contexts := selectors[config.DefaultRecipeName].contexts
		arcContext := contexts[len(contexts)-1].RaylineARC
		if arcContext.RoutesThrowaway != execution || (arcContext.State != nil) == execution {
			t.Fatalf("execution=%t: throwaway %t, episode read %t", execution, arcContext.RoutesThrowaway, arcContext.State != nil)
		}
		store := router.RaylineARCEpisodeStore.(*raylinearc.MemoryEpisodeStore)
		_, read, err := store.Snapshot(context.Background(), raylinearc.HashEpisodeID("conversation-execution"), 3)
		if err != nil || (read.Version() != 0) == execution {
			t.Fatalf("execution=%t: record version %d (%v)", execution, read.Version(), err)
		}
	}
}

// With episode writes off every lookup is ephemeral, as before: nothing is
// stored and no episode is reported.
func TestRoutesLookupWithEpisodeWritesOffStoresNothing(t *testing.T) {
	fixture := newPendingRoutesFixture(t, false, "")
	advertiseLookupSessions(fixture.policy)
	const conversation = "conversation-off"
	for range 2 {
		answer := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
		if answer.Episode != nil {
			t.Fatalf("episode %+v with writes off", answer.Episode)
		}
	}
	if state := fixture.stored(t, conversation); state != nil {
		t.Fatalf("writes off stored %+v", state)
	}
}

// A generation turn of the conversation resolves the pending decision a
// lookup left before anything reads the episode, as a lookup does: the next
// turn's request records it as inferred, and the turn's own commit persists
// that with the turn it served.
func TestGenerationTurnResolvesALookupsPendingDecision(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-generation"
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))

	decision := fixture.router.Config.Decisions[0]
	reqCtx := &RequestContext{
		Headers:           map[string]string{decisionServiceEpisodeHeader: conversation},
		RaylineARCRawBody: pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext),
		SourceFormat:      llmprotocol.AnthropicMessagesV1,
		RequestID:         "generation-1",
	}
	arcContext := fixture.router.buildRaylineARCSelectionContext(decision.Algorithm, reqCtx, decision.ModelRefs, raylineARCEpisodeRequired)
	if arcContext.PreparationFailure != "" {
		t.Fatalf("preparation failed: %s", arcContext.PreparationFailure)
	}
	transaction := reqCtx.RaylineARCTransaction
	if transaction.decidesPending || arcContext.State != transaction.state || arcContext.State.Pending != nil ||
		arcContext.State.TurnIndex != 1 || arcContext.State.PreviousArm == nil {
		t.Fatalf("a generation turn saw %+v", arcContext.State)
	}
	transaction.markSelection(*arcContext.State.PreviousArm, 7)
	if err := transaction.commit(context.Background(), reqCtx); err != nil {
		t.Fatal(err)
	}
	state := fixture.stored(t, conversation)
	if state.Pending != nil || state.TurnIndex != 2 || len(state.Policy.Ledger) != 1 ||
		state.Policy.Ledger[0].Basis != raylinearc.PolicyLedgerBasisInferred {
		t.Fatalf("after the generation turn the record is %+v", state)
	}
}
