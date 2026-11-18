package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// generationTurn builds, selects and stages a generation turn of conversation
// on the fixture's cell, as the routed path does before dispatch.
func (fixture *pendingRoutesFixture) generationTurn(t *testing.T, conversation string, body []byte) *RequestContext {
	t.Helper()
	decision := fixture.router.Config.Decisions[0]
	reqCtx := &RequestContext{
		Headers: map[string]string{decisionServiceEpisodeHeader: conversation}, RaylineARCRawBody: body,
		SourceFormat: llmprotocol.AnthropicMessagesV1, RequestID: "generation-" + conversation,
		TraceContext: context.Background(),
	}
	arcContext := fixture.router.buildRaylineARCSelectionContext(decision.Algorithm, reqCtx, decision.ModelRefs, raylineARCEpisodeRequired)
	if arcContext.PreparationFailure != "" {
		t.Fatalf("preparation failed: %s", arcContext.PreparationFailure)
	}
	result, err := fixture.policy.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: decision.Name, CandidateModels: decision.ModelRefs, RaylineARC: arcContext,
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	reqCtx.VSRRaylineARC, reqCtx.VSRSelectedDecision, reqCtx.VSRSelectedModel = result.RaylineARC, &decision, result.SelectedModel
	reqCtx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, result.RaylineARC.SerializedTokens)
	stageRaylineARCPolicySelection(reqCtx, result.RaylineARC, result.SelectedModel)
	return reqCtx
}

func withFallback(fixture *pendingRoutesFixture) {
	fixture.router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService.Fallback = &config.RaylineARCPolicyFallbackConfig{Enabled: true}
}

// A generation turn that records a lookup's decision as a turn that ran and
// is then refused by the provider after its lease was given back (a late
// refusal) still excludes the refusing model: the store holds the episode
// the turn was decided on, so the refusal matches it.
func TestLateRefusalAfterAnInferredTurnExcludesTheModel(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	withFallback(fixture)
	const conversation = "conversation-late-refusal"
	fixture.picks = []string{fixture.actions[0]}
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	reqCtx := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	if reqCtx.RaylineARCTransaction.state.TurnIndex != 1 {
		t.Fatalf("the turn did not infer the lookup's decision: turn %d", reqCtx.RaylineARCTransaction.state.TurnIndex)
	}
	refused := refusedModel(reqCtx)
	if err := reqCtx.RaylineARCTransaction.abort(context.Background(), "upstream_error"); err != nil {
		t.Fatal(err)
	}
	declineRefusedTurn(reqCtx)
	stored := fixture.stored(t, conversation)
	if stored.Policy == nil || !stored.Policy.Excludes(refused) {
		t.Fatalf("after a late refusal the store does not exclude %q: %+v", refused, stored.Policy)
	}
}

// A lookup decided arm 0; a generation call for the same messages dropped
// that decision, decided itself, served its reply and
// failed to commit. The next turn must not record the lookup's decision as
// the one that wrote the reply, whether the generation call staged no
// boundary decision (an unscheduled cell), or one for another arm or the
// same arm (a boundary names only the arm, and several actions, one per
// thinking level, share one): the generation call's decision came after the
// lookup's and superseded it.
func TestNextTurnDoesNotInferADecisionAGenerationCallSuperseded(t *testing.T) {
	for _, boundaryArm := range []int{-1, 1, 0} {
		fixture := newPendingRoutesFixture(t, true, "")
		conversation := fmt.Sprintf("conversation-superseded-%d", boundaryArm)
		fixture.picks = []string{fixture.actions[0]} // arm 0 (think)
		fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
		decision := fixture.router.Config.Decisions[0]
		reqCtx := &RequestContext{
			Headers:           map[string]string{decisionServiceEpisodeHeader: conversation},
			RaylineARCRawBody: pendingRoutesBody(t, pendingTurnTask), SourceFormat: llmprotocol.AnthropicMessagesV1,
			RequestID: "generation-same-turn",
		}
		arcContext := fixture.router.buildRaylineARCSelectionContext(decision.Algorithm, reqCtx, decision.ModelRefs, raylineARCEpisodeRequired)
		if arcContext.PreparationFailure != "" {
			t.Fatal(arcContext.PreparationFailure)
		}
		raw, _ := json.Marshal(pendingTurnTask)
		digests := raylinearc.NewPrefixDigests([]json.RawMessage{raw})
		if boundaryArm >= 0 {
			reqCtx.RaylineARCTransaction.retainPolicyBoundary(context.Background(),
				raylinearc.NewPolicyBoundaryDecisionFrom(boundaryArm, 0, &raylinearc.PolicyEpisodeState{}, digests))
		}
		if err := reqCtx.RaylineARCTransaction.abort(context.Background(), "commit_failure"); err != nil {
			t.Fatal(err)
		}
		if stored := fixture.stored(t, conversation); stored.Pending == nil || !stored.Pending.Superseded {
			t.Fatalf("boundary arm %d: the dropped decision is not stored as superseded: %+v", boundaryArm, stored.Pending)
		}
		next := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
		state := next.RaylineARCTransaction.state
		if state.TurnIndex != 0 || (state.Policy != nil && len(state.Policy.Ledger) != 0) {
			t.Fatalf("boundary arm %d: the next turn recorded the superseded lookup decision: %+v", boundaryArm, state.Policy)
		}
		_ = next.RaylineARCTransaction.abort(context.Background(), "test")
	}
}

// The reverse order: a failed generation turn stored its boundary decision,
// then a lookup for the same messages answered that arm and kept it. The
// lookup's decision is the newer one, and the next turn records it.
func TestNextTurnInfersALookupThatKeptAnEarlierBoundary(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-kept-then-next"
	fixture.picks = []string{fixture.actions[0]} // arm 0
	turn := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	raw, _ := json.Marshal(pendingTurnTask)
	digests := raylinearc.NewPrefixDigests([]json.RawMessage{raw})
	turn.RaylineARCTransaction.retainPolicyBoundary(context.Background(),
		raylinearc.NewPolicyBoundaryDecisionFrom(0, 0, &raylinearc.PolicyEpisodeState{}, digests))
	if err := turn.RaylineARCTransaction.abort(context.Background(), "upstream_error"); err != nil {
		t.Fatal(err)
	}
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	next := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	state := next.RaylineARCTransaction.state
	if state.TurnIndex != 1 || len(state.Policy.Ledger) != 1 || state.Policy.Ledger[0].Basis != raylinearc.PolicyLedgerBasisInferred {
		t.Fatalf("the lookup's decision was not inferred: turn %d %+v", state.TurnIndex, state.Policy)
	}
	_ = next.RaylineARCTransaction.abort(context.Background(), "test")
}

// A lookup for the same messages as a failed generation turn's stored
// boundary decision keeps that decision when it answered the same arm (it
// replayed it): the generation retry still reuses that arm. A lookup that
// answered another arm is the newer decision and clears it.
func TestLookupKeepsAStoredBoundaryDecisionItReplayed(t *testing.T) {
	for _, tc := range []struct {
		boundaryArm int
		kept        bool
	}{{0, true}, {1, false}} {
		fixture := newPendingRoutesFixture(t, true, "")
		fixture.picks = []string{fixture.actions[0]} // the lookup answers arm 0
		conversation := fmt.Sprintf("conversation-boundary-%d", tc.boundaryArm)
		reqCtx := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask))
		raw, _ := json.Marshal(pendingTurnTask)
		digests := raylinearc.NewPrefixDigests([]json.RawMessage{raw})
		boundary := raylinearc.NewPolicyBoundaryDecisionFrom(tc.boundaryArm, 0, &raylinearc.PolicyEpisodeState{}, digests)
		reqCtx.RaylineARCTransaction.retainPolicyBoundary(context.Background(), boundary)
		if err := reqCtx.RaylineARCTransaction.abort(context.Background(), "upstream_error"); err != nil {
			t.Fatal(err)
		}
		fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
		state := fixture.stored(t, conversation)
		if kept := state.PolicyBoundary != nil && state.PolicyBoundary.Same(boundary); kept != tc.kept || state.Pending == nil {
			t.Fatalf("boundary arm %d: kept %t, want %t (pending %+v)", tc.boundaryArm, kept, tc.kept, state.Pending)
		}
	}
}

// lookupAs posts a /v1/routes lookup in format with body.
func (fixture *pendingRoutesFixture) lookupAs(t *testing.T, conversation, format string, body map[string]any) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	response := fixture.router.handleRaylineRoutesAPI(raw, routesContext(map[string]string{
		"x-rayline-format": format, raylineRoutesSessionHeader: conversation,
	}))
	fixture.router.background.Wait()
	if code := immediateStatusCode(t, response); code != 200 {
		t.Fatalf("%s lookup status %d: %s", format, code, response.GetImmediateResponse().Body)
	}
}

// The next-turn rule per wire format, through the endpoint: a lookup, then
// the next turn's lookup carrying a reply. The decision is recorded as ran
// (turn 1, one inferred entry) whatever item the reply opens with; a request
// that only adds a prefill to the same turn drops it.
func TestNextTurnRulePerWireFormat(t *testing.T) {
	user := func(text string) map[string]any { return map[string]any{"role": "user", "content": text} }
	responsesUser := func(text string) map[string]any {
		return map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": text}}}
	}
	cases := []struct {
		name, format string
		first, next  map[string]any
		inferred     bool
	}{
		{
			name: "chat with the system message in messages", format: "chat",
			first: map[string]any{"model": "rayline/arc-dev", "messages": []any{
				map[string]any{"role": "system", "content": "be brief"}, user("fix it"),
			}},
			next: map[string]any{"model": "rayline/arc-dev", "messages": []any{
				map[string]any{"role": "system", "content": "be brief"},
				user("fix it"),
				map[string]any{"role": "assistant", "content": "fixed"},
				user("thanks"),
			}},
			inferred: true,
		},
		{
			name: "anthropic prefill completed", format: "anthropic",
			first: map[string]any{"model": "rayline/arc-dev", "max_tokens": 64, "messages": []any{
				user("fix it"), map[string]any{"role": "assistant", "content": "The fix is"},
			}},
			next: map[string]any{"model": "rayline/arc-dev", "max_tokens": 64, "messages": []any{
				user("fix it"), map[string]any{"role": "assistant", "content": "The fix is a nil check."}, user("thanks"),
			}},
			inferred: true,
		},
		{
			name: "anthropic prefill added to the same turn", format: "anthropic",
			first: map[string]any{"model": "rayline/arc-dev", "max_tokens": 64, "messages": []any{user("fix it")}},
			next: map[string]any{"model": "rayline/arc-dev", "max_tokens": 64, "messages": []any{
				user("fix it"), map[string]any{"role": "assistant", "content": "The fix is"},
			}},
			inferred: false,
		},
	}
	for _, opener := range []map[string]any{
		{"type": "reasoning", "id": "rs_1", "summary": []any{}},
		{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{}"},
		{"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": "*** Begin Patch"},
		{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "done"}}},
	} {
		cases = append(cases, struct {
			name, format string
			first, next  map[string]any
			inferred     bool
		}{
			name: "responses reply opening with " + opener["type"].(string), format: "responses",
			first: map[string]any{"model": "rayline/arc-dev", "input": []any{responsesUser("fix it")}},
			next: map[string]any{"model": "rayline/arc-dev", "input": []any{
				responsesUser("fix it"), opener, responsesUser("thanks"),
			}},
			inferred: true,
		})
	}
	for index, tc := range cases {
		fixture := newPendingRoutesFixture(t, true, "")
		conversation := fmt.Sprintf("conversation-format-%d", index)
		fixture.lookupAs(t, conversation, tc.format, tc.first)
		fixture.lookupAs(t, conversation, tc.format, tc.next)
		state := fixture.stored(t, conversation)
		inferred := state.TurnIndex == 1 && state.Policy != nil && len(state.Policy.Ledger) == 1 &&
			state.Policy.Ledger[0].Basis == raylinearc.PolicyLedgerBasisInferred
		if inferred != tc.inferred || state.Pending == nil {
			t.Fatalf("%s: inferred %t, want %t (turn %d, policy %+v)", tc.name, inferred, tc.inferred, state.TurnIndex, state.Policy)
		}
	}
}

// A lookup the selection read as a side call wrote nothing: it reports the
// recorded turn count, as a lookup classified as a side call before the
// episode was read does, not the pending turn.
func TestSideCallLookupReportsTheRecordedTurn(t *testing.T) {
	state, _ := raylinearc.NewEpisodeState(2)
	state.TurnIndex = 3
	reqCtx := &RequestContext{RaylineARCTransaction: &raylineARCEpisodeTransaction{state: state, decidesPending: true}}
	if got := committedEpisodeTurnIndex(reqCtx); got != 4 {
		t.Fatalf("a pending lookup reported %d, want 4", got)
	}
	reqCtx.RaylineARCTransaction.sideCall = true
	if got := committedEpisodeTurnIndex(reqCtx); got != 3 {
		t.Fatalf("a lookup read as a side call reported %d, want the recorded 3", got)
	}
}

// Session telemetry is not written for a lookup at all (a consult names no
// requested model, which the record requires), so no turn number of a
// pending decision reaches it.
func TestLookupWritesNoSessionTelemetry(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-telemetry"
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	hash := raylinearc.HashEpisodeID(conversation)
	if _, ok := sessiontelemetry.GetRouterSessionSnapshot(hash, time.Now()); ok {
		t.Fatal("a lookup wrote a session telemetry record")
	}
	if _, ok := sessiontelemetry.GetLastModel(hash); ok {
		t.Fatal("a lookup wrote a session last-model record")
	}
}
