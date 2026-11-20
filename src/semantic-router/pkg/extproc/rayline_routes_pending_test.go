package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// pendingRoutesFixture is a policy-service cell serving /v1/routes through the
// real policy selection, a fake policy service and a memory episode store.
type pendingRoutesFixture struct {
	router *OpenAIRouter
	policy *policySelectorFixture
	// store is the memory store; records reads whichever store the cell
	// uses (a shared Redis one in the two-instance tests).
	store   *raylinearc.MemoryEpisodeStore
	records pendingRecords
	actions []string
	// picks are the actions the fake service answers, in order; the last
	// repeats.
	picks []string
	// now is the episode store's clock; expire moves it past the idle TTL.
	now atomic.Int64
}

type pendingRecords interface {
	raylinearc.EpisodeSnapshotStore
	raylinearc.ServedModelStore
}

// expire moves the store's clock past the episode idle TTL, as an hour
// without a request does on a cell.
func (fixture *pendingRoutesFixture) expire() {
	fixture.now.Add(int64(2 * time.Minute))
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
	fixture := &pendingRoutesFixture{policy: policy, actions: policyTestActionIDs(arc.PolicyService.Bindings)}
	fixture.now.Store(time.Now().UnixNano())
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 16, IdleTTL: time.Minute, Now: func() time.Time { return time.Unix(0, fixture.now.Load()) },
	})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{decision}
	cfg.AutoModelNames = []string{"rayline/arc-dev"}
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineARC, policy.selector)
	fixture.router = &OpenAIRouter{Config: cfg, ModelSelector: registry, RaylineARCEpisodeStore: store}
	fixture.store, fixture.records = store, store
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
	state, read, err := fixture.records.Snapshot(context.Background(), raylinearc.HashEpisodeID(conversation), 2)
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
		stored, read, err := store.Snapshot(context.Background(), raylinearc.HashEpisodeID("conversation-execution"), 3)
		if err != nil || (read.Version() != 0) == execution {
			t.Fatalf("execution=%t: record version %d (%v)", execution, read.Version(), err)
		}
		// The control lookup stored a pending decision, not a turn.
		if !execution && (stored.Pending == nil || stored.TurnIndex != 0 || stored.PreviousArm != nil) {
			t.Fatalf("the control lookup stored %+v", stored)
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

// A turn that resolved a pending decision cleans up the service's staged
// turn from the turn count it read: a successor that committed the turn its
// resolution moved to keeps its record.
func TestResolvedTurnDropsStagedTurnsFromTheTurnItRead(t *testing.T) {
	shared := newFakeStagedTurnStore()
	episode := strings.Repeat("e", 64)
	if err := shared.StoreStagedTurn(context.Background(), episode, 4, []byte("successor")); err != nil {
		t.Fatal(err)
	}
	state, _ := raylinearc.NewEpisodeState(2)
	state.TurnIndex = 4
	transaction := &raylineARCEpisodeTransaction{
		state: state, resolvedPending: true, readTurnIndex: 3,
		policyStaging: &policyStaging{shared: shared, episodeIDHash: episode, turns: newPolicyStagedTurns()},
	}
	transaction.dropPolicyStagedTurn(context.Background())
	if record, _ := shared.LoadStagedTurn(context.Background(), episode); string(record) != "successor" {
		t.Fatalf("the successor's staged turn was deleted: %q", record)
	}
}

// A turn that recorded a pending decision as a turn that ran makes that turn
// durable when it stages its boundary decision, before it commits: the
// stage records the inferred turn's served model with it, as a commit does.
func TestResolvedTurnStagesTheInferredServedModel(t *testing.T) {
	_, store, episode := boundaryFixture(t)
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	resolved := cloneARCState(state)
	resolved.ServedModel = "vendor/inferred"
	transaction := newRaylineARCEpisodeTransaction(store, lease, resolved, episode, 0, nil)
	messages := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	transaction.retainPolicyBoundary(context.Background(), raylinearc.NewPolicyBoundaryDecision(1, 0, &raylinearc.PolicyEpisodeState{}, messages))
	if transaction.state.PolicyBoundary == nil {
		t.Fatal("the boundary decision was not staged")
	}
	if err := store.Abort(context.Background(), lease); err != nil {
		t.Fatal(err)
	}
	if served, err := store.LastServedModel(context.Background(), episode); err != nil || served != "vendor/inferred" {
		t.Fatalf("served model %q (%v), want the inferred turn's", served, err)
	}
}

// A resend that joined an identical tracked lookup in flight shares the
// leader's decision and episode: it reports the same pending turn and
// writes nothing.
func TestCoalescedLookupReportsThePendingTurn(t *testing.T) {
	state, _ := raylinearc.NewEpisodeState(2)
	state.TurnIndex = 3
	reqCtx := &RequestContext{RaylineRouteSurface: string(routerruntime.RouteDecisionSurfaceRoutes)}
	reqCtx.RaylineARCTransaction = newBorrowedRaylineARCEpisodeTransaction(nil, state, strings.Repeat("e", 64), nil)
	resolveRaylineARCPendingDecision(reqCtx, &selection.RaylineARCSelectionContext{
		State: state, Coalesced: &selection.SelectionResult{},
	})
	if got := committedEpisodeTurnIndex(reqCtx); got != 4 {
		t.Fatalf("a coalesced lookup reported turn %d, want the leader's pending turn 4", got)
	}
	if err := reqCtx.RaylineARCTransaction.commit(context.Background(), reqCtx); err != nil {
		t.Fatalf("a coalesced lookup's commit: %v", err)
	}
}

// twoStage makes the fixture's package a two-stage one whose stage one can
// be told the held model (attested_held_action_id), the package kind that
// reads the served-model record on a turn without episode memory.
func (fixture *pendingRoutesFixture) twoStage() {
	scorer := fixture.policy.selector.armedComponents().scorer.(*policyServiceScorer)
	scorer.twoStage, scorer.attestable = true, true
}

func (fixture *pendingRoutesFixture) servedRecord(t *testing.T, conversation string) string {
	t.Helper()
	served, err := fixture.records.LastServedModel(context.Background(), raylinearc.HashEpisodeID(conversation))
	if err != nil {
		t.Fatal(err)
	}
	return served
}

// attestedModel is the model a decide request told the policy was held, ""
// when it told none.
func (fixture *pendingRoutesFixture) attestedModel(request raylinearc.PolicyDecisionRequest) string {
	if request.Selection.AttestedHeldActionID == nil {
		return ""
	}
	for _, binding := range fixture.policy.decision.Algorithm.RaylineARC.PolicyService.Bindings {
		if binding.ActionID == *request.Selection.AttestedHeldActionID {
			return binding.Model
		}
	}
	return "unknown action"
}

// Rule: a conversation that comes back after its episode expired, with
// nothing reported, is held to the last decision, assumed to have run. A
// turn served on A, the episode expires, a lookup in a fresh episode decides
// B, the episode expires again: the next request is offered B as the held
// model.
func TestLookupAfterExpiryHoldsTheLastDecision(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	fixture.twoStage()
	const conversation = "conversation-cold"
	// Turn 1 was served on vendor/off; its episode has since expired.
	if err := fixture.store.SeedServedModel(raylinearc.HashEpisodeID(conversation), "vendor/off"); err != nil {
		t.Fatal(err)
	}
	fixture.picks = []string{fixture.actions[0]} // vendor/think
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	if served := fixture.servedRecord(t, conversation); served != "vendor/think" {
		t.Fatalf("after the lookup the served record is %q, want the decided vendor/think", served)
	}
	fixture.expire()
	next := map[string]any{"role": "user", "content": "and the docs"}
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext,
		map[string]any{"role": "assistant", "content": "tests added"}, next))
	sent := fixture.policy.fake.received()
	if got := fixture.attestedModel(sent[0]); got != "vendor/off" {
		t.Fatalf("the lookup in a fresh episode attested %q, want the served vendor/off", got)
	}
	if got := fixture.attestedModel(sent[len(sent)-1]); got != "vendor/think" {
		t.Fatalf("after the episode expired again the held model offered is %q, want the last decision vendor/think", got)
	}
}

// Rule: within the episode's life the next turn settles the pending decision
// as a turn that ran, and the episode then knows its model: the 7-day record
// is not read, so an older record cannot override the decision.
func TestNextTurnWithinTheEpisodeDoesNotReadTheServedRecord(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	fixture.twoStage()
	const conversation = "conversation-warm"
	hash := raylinearc.HashEpisodeID(conversation)
	if err := fixture.store.SeedServedModel(hash, "vendor/off"); err != nil {
		t.Fatal(err)
	}
	fixture.picks = []string{fixture.actions[0]} // vendor/think
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	// The record lags the decision (as it did before lookups wrote it):
	// only the episode can tell the next turn its model.
	if err := fixture.store.SeedServedModel(hash, "vendor/off"); err != nil {
		t.Fatal(err)
	}
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	state := fixture.stored(t, conversation)
	if state.TurnIndex != 1 || state.PreviousArm == nil || *state.PreviousArm != 0 ||
		state.Policy.Ledger[0].Basis != raylinearc.PolicyLedgerBasisInferred {
		t.Fatalf("the next turn did not settle the decision as inferred: %+v", state)
	}
	sent := fixture.policy.fake.received()
	if got := fixture.attestedModel(sent[1]); got != "" {
		t.Fatalf("the next turn attested %q: the served record was read although the episode knows its model", got)
	}
}

// Rule: asking again for the same turn replaces the decision, and the served
// record names the newest one.
func TestRepeatAskRecordsTheLatestDecisionAsServed(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	fixture.picks = []string{fixture.actions[0], fixture.actions[2]}
	const conversation = "conversation-repeat-served"
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	if served := fixture.servedRecord(t, conversation); served != "vendor/think" {
		t.Fatalf("first lookup recorded %q", served)
	}
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	if served := fixture.servedRecord(t, conversation); served != "vendor/off" {
		t.Fatalf("after the repeat ask the served record is %q, want the newest decision vendor/off", served)
	}
}

// The served-model read on a generation turn is gated as before the pending
// decisions: a policy-service main turn whose episode has no previous arm
// reads the record; one whose episode has a previous arm does not.
func TestGenerationTurnReadsTheServedRecordOnlyWithoutAPreviousArm(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-generation-gate"
	if err := fixture.store.SeedServedModel(raylinearc.HashEpisodeID(conversation), "vendor/off"); err != nil {
		t.Fatal(err)
	}
	decision := fixture.router.Config.Decisions[0]
	build := func(body []byte) (*selection.RaylineARCSelectionContext, *RequestContext) {
		reqCtx := &RequestContext{
			Headers: map[string]string{decisionServiceEpisodeHeader: conversation}, RaylineARCRawBody: body,
			SourceFormat: llmprotocol.AnthropicMessagesV1, RequestID: "generation-gate",
		}
		arcContext := fixture.router.buildRaylineARCSelectionContext(decision.Algorithm, reqCtx, decision.ModelRefs, raylineARCEpisodeRequired)
		if arcContext.PreparationFailure != "" {
			t.Fatalf("preparation failed: %s", arcContext.PreparationFailure)
		}
		return arcContext, reqCtx
	}
	cold, reqCtx := build(pendingRoutesBody(t, pendingTurnTask))
	if len(cold.ServedModels) != 1 || cold.ServedModels[0].Model != "vendor/off" {
		t.Fatalf("a turn without a previous arm read %+v, want the served record", cold.ServedModels)
	}
	reqCtx.RaylineARCTransaction.markSelection(1, 7)
	if err := reqCtx.RaylineARCTransaction.commit(context.Background(), reqCtx); err != nil {
		t.Fatal(err)
	}
	warm, reqCtx := build(pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	if warm.State.PreviousArm == nil || warm.ServedModels != nil {
		t.Fatalf("a turn with previous arm %v read %+v", warm.State.PreviousArm, warm.ServedModels)
	}
	if err := reqCtx.RaylineARCTransaction.abort(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
}

// Rule: a repeat ask decides on the turns that ran, and the policy never
// sees its own unexecuted decision as the held model. In a fresh episode the
// served-model record names the first ask's decision once it is written, so
// the repeat (and an edited one) is held to the record as the first ask read
// it, through any number of repeats.
func TestRepeatAskIsNotHeldToItsOwnDecision(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	fixture.twoStage()
	const conversation = "conversation-repeat-hold"
	if err := fixture.store.SeedServedModel(raylinearc.HashEpisodeID(conversation), "vendor/off"); err != nil {
		t.Fatal(err)
	}
	fixture.picks = []string{fixture.actions[0]} // vendor/think every time
	body := pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext)
	fixture.lookup(t, conversation, body)
	fixture.lookup(t, conversation, body)
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnEdited))
	for index, sent := range fixture.policy.fake.received() {
		if got := fixture.attestedModel(sent); got != "vendor/off" {
			t.Fatalf("ask %d was held to %q, want the record before the asks, vendor/off", index+1, got)
		}
	}
	if served := fixture.servedRecord(t, conversation); served != "vendor/think" {
		t.Fatalf("the served record is %q, want the last decision vendor/think", served)
	}
	if state := fixture.stored(t, conversation); state.Pending.PriorServedModel != "vendor/off" {
		t.Fatalf("the pending decision kept %q as the record before it", state.Pending.PriorServedModel)
	}
}

// A generation turn that drops a lookup's decision and stages before it runs
// (a boundary or a refusal) keeps the dropped decision on the staged
// record: if the turn then fails, its retry is still held to the record from
// before the lookup, not to the lookup's unexecuted pick.
func TestFailedTurnAfterADroppedDecisionKeepsItForTheRetry(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-dropped-stage"
	if err := fixture.store.SeedServedModel(raylinearc.HashEpisodeID(conversation), "vendor/off"); err != nil {
		t.Fatal(err)
	}
	fixture.picks = []string{fixture.actions[0]} // vendor/think
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	decision := fixture.router.Config.Decisions[0]
	reqCtx := &RequestContext{
		Headers:           map[string]string{decisionServiceEpisodeHeader: conversation},
		RaylineARCRawBody: pendingRoutesBody(t, pendingTurnTask), SourceFormat: llmprotocol.AnthropicMessagesV1,
		RequestID: "generation-same-turn",
	}
	arcContext := fixture.router.buildRaylineARCSelectionContext(decision.Algorithm, reqCtx, decision.ModelRefs, raylineARCEpisodeRequired)
	if arcContext.PreparationFailure != "" || len(arcContext.ServedModels) != 1 || arcContext.ServedModels[0].Model != "vendor/off" {
		t.Fatalf("the same-turn generation was held to %+v (%s)", arcContext.ServedModels, arcContext.PreparationFailure)
	}
	transaction := reqCtx.RaylineARCTransaction
	messages := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	transaction.retainPolicyBoundary(context.Background(),
		raylinearc.NewPolicyBoundaryDecision(1, 0, &raylinearc.PolicyEpisodeState{}, messages))
	if err := transaction.abort(context.Background(), "upstream_error"); err != nil {
		t.Fatal(err)
	}
	if served := fixture.servedRecord(t, conversation); served != "vendor/off" {
		t.Fatalf("after the failed turn the served record is %q, want the one before the lookup", served)
	}
	state := fixture.stored(t, conversation)
	if state.PolicyBoundary == nil || state.Pending == nil || state.Pending.PriorServedModel != "vendor/off" {
		t.Fatalf("the staged record lost the dropped decision: boundary %v pending %+v", state.PolicyBoundary, state.Pending)
	}
	retry := fixture.router.buildRaylineARCSelectionContext(decision.Algorithm, &RequestContext{
		Headers:           map[string]string{decisionServiceEpisodeHeader: conversation},
		RaylineARCRawBody: pendingRoutesBody(t, pendingTurnEdited), SourceFormat: llmprotocol.AnthropicMessagesV1,
		RequestID: "generation-retry",
	}, decision.ModelRefs, raylineARCEpisodeRequired)
	if len(retry.ServedModels) != 1 || retry.ServedModels[0].Model != "vendor/off" {
		t.Fatalf("the retry was held to %+v, want the record before the lookup", retry.ServedModels)
	}
}
