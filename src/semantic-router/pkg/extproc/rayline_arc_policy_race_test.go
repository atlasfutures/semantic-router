/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A racing cell (ADR 0155): turn 1 stages and, at its commit, appends its
// reply; turn 2 extends that prefix with a tool result, sends a delta and
// races it against the append's lease.

type raceFixture struct {
	*policySelectorFixture
	store   *raylinearc.MemoryEpisodeStore
	episode string
	// decided is the action the service's decide picks, leased the one its
	// append leases.
	decided, leased string
}

func newRaceFixture(t *testing.T, schedule string, edit func(*config.RaylineARCPolicyServiceConfig)) *raceFixture {
	t.Helper()
	decision := policyTestDecision(schedule)
	policy := decision.Algorithm.RaylineARC.PolicyService
	policy.DecideMode, policy.RaceMS, policy.TotalTimeoutSeconds = config.RaylineARCDecideModeRace, 30, 5
	if edit != nil {
		edit(policy)
	}
	bindings := policy.Bindings
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, policyTestActionIDs(bindings))
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fake.URL(), TotalTimeout: 5 * time.Second}),
	})
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	fixture := &raceFixture{
		policySelectorFixture: &policySelectorFixture{fake: fake, selector: selector, decision: decision},
		store:                 store, episode: strings.Repeat("e", 64),
		decided: bindings[0].ActionID, leased: bindings[2].ActionID,
	}
	fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		for _, action := range request.Selection.AvailableActionIDs {
			if action == fixture.decided {
				return action
			}
		}
		return request.Selection.AvailableActionIDs[0]
	})
	leased := fixture.leased
	fake.leaseAction = func(raylinearc.PolicyAppendRequest) *string { return &leased }
	return fixture
}

func (fixture *raceFixture) scorer() *policyServiceScorer {
	return fixture.selector.armedComponents().scorer.(*policyServiceScorer)
}

// turn prepares the strict episode, selects and stages the selection as the
// request path does.
func (fixture *raceFixture) turn(t *testing.T, body []byte) (*selection.SelectionResult, *RequestContext) {
	t.Helper()
	lease, state, err := fixture.store.Prepare(context.Background(), fixture.episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	result, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		TraceContext:          context.Background(),
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(fixture.store, lease, state, fixture.episode, time.Minute, nil),
	}
	ctx.RaylineARCTransaction.commitOnCompletion = true
	ctx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, 10)
	stageRaylineARCPolicySelection(ctx, result.RaylineARC, "")
	// A turn a test leaves unfinished is aborted, which stops its renewal;
	// after a commit the abort is a no-op.
	t.Cleanup(func() { _ = ctx.RaylineARCTransaction.abort(context.Background(), "test_cleanup") })
	return result, ctx
}

// complete is the client receiving the whole reply: the turn commits.
func complete(t *testing.T, ctx *RequestContext, reply []byte) {
	t.Helper()
	stageRaylineARCAppendSource(ctx, reply)
	ctx.RaylineARCTransaction.stageCallUsage(responseUsageMetrics{})
	if err := ctx.RaylineARCTransaction.commit(context.Background(), ctx); err != nil {
		t.Fatal(err)
	}
}

var (
	raceUser = map[string]any{"role": "user", "content": "list the files"}
	// raceReplyContent is turn 1's reply as the upstream states it, and
	// raceAssistant the same message as Claude Code echoes it in the next
	// request (testdata/thread_continuation): citations [] on text, a direct
	// caller on tool_use.
	raceReplyContent = []any{
		map[string]any{"type": "text", "text": "Listing them."},
		map[string]any{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": map[string]any{"command": "ls"}},
	}
	raceAssistant = map[string]any{"role": "assistant", "content": []any{
		map[string]any{"type": "text", "text": "Listing them.", "citations": []any{}},
		map[string]any{"type": "tool_use", "id": "toolu_1", "caller": map[string]any{"type": "direct"}, "name": "Bash", "input": map[string]any{"command": "ls"}},
	}}
	raceToolResult = map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"},
	}}
)

// raceReply is turn 1's reply as the client receives it.
func raceReply(t *testing.T) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "think",
		"content": raceReplyContent, "stop_reason": "tool_use", "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 10, "output_tokens": 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// stagedFirstTurn runs and commits turn 1, which stages and appends.
func (fixture *raceFixture) stagedFirstTurn(t *testing.T) {
	t.Helper()
	first, ctx := fixture.turn(t, policyTestRequest(t, raceUser))
	if first.RaylineARC.PolicyDecisionMode != policyDecisionModeSync {
		t.Fatalf("turn 1 decision mode %q", first.RaylineARC.PolicyDecisionMode)
	}
	complete(t, ctx, raceReply(t))
	sent := fixture.fake.received()
	if len(sent) != 1 || sent[0].Stage == nil || !*sent[0].Stage || sent[0].TurnDelta != nil {
		t.Fatalf("turn 1 decide %+v", sent)
	}
}

func slowDeltas(fixture *raceFixture, delay time.Duration) {
	fixture.fake.mu.Lock()
	defer fixture.fake.mu.Unlock()
	fixture.fake.decideDelay = func(request raylinearc.PolicyDecisionRequest) time.Duration {
		if request.TurnDelta != nil {
			return delay
		}
		return 0
	}
}

// A decide that lands within the race serves its own action; it went out as
// a delta on the appended prefix, with the shell only and staged.
func TestRaceWonServesTheDecide(t *testing.T) {
	fixture := newRaceFixture(t, "", func(policy *config.RaylineARCPolicyServiceConfig) { policy.RaceMS = 2000 })
	fixture.stagedFirstTurn(t)
	appends := fixture.fake.receivedAppends()
	if len(appends) != 1 || len(appends[0].Messages) != 1 || appends[0].ContextEpoch != "0" ||
		len(appends[0].Attribution) != 1 || appends[0].Attribution[0].Message != 1 ||
		appends[0].Attribution[0].ActionID != fixture.decided {
		t.Fatalf("turn 1 append %+v", appends)
	}
	second, _ := fixture.turn(t, policyTestRequest(t, raceUser, raceAssistant, raceToolResult))
	if second.RaylineARC.PolicyActionID != fixture.decided || second.RaylineARC.PolicyDecisionMode != policyDecisionModeRaceWon {
		t.Fatalf("turn 2 served %s (%s)", second.RaylineARC.PolicyActionID, second.RaylineARC.PolicyDecisionMode)
	}
	sent := fixture.fake.received()
	delta := sent[len(sent)-1]
	if delta.TurnDelta == nil || delta.TurnDelta.BasePrefixDigest != fakePrefixDigest(fixture.episode, 2) ||
		len(delta.TurnDelta.Messages) != 1 || delta.Request.Shell != raylinearc.PolicyShellMessages ||
		delta.Stage == nil || !*delta.Stage || len(delta.Attribution) != 1 {
		t.Fatalf("turn 2 decide %+v", delta)
	}
	if body := fixture.fake.receivedBodies()[len(sent)-1]; bytes.Contains(body, []byte(`"messages":[{"content":"list`)) {
		t.Fatalf("the delta re-sent the staged conversation: %s", body)
	}
}

// A decide that misses the race serves the lease's action. The late decide
// still finishes inside the episode lease, before the append; the ledger and
// the append's attribution name the action that served.
func TestRaceLostServesTheLeaseAndRecordsIt(t *testing.T) {
	fixture := newRaceFixture(t, "", nil)
	fixture.stagedFirstTurn(t)
	slowDeltas(fixture, 300*time.Millisecond)
	baseline := runtime.NumGoroutine()
	second, ctx := fixture.turn(t, policyTestRequest(t, raceUser, raceAssistant, raceToolResult))
	if second.RaylineARC.PolicyActionID != fixture.leased || second.SelectedModel != "off" ||
		second.RaylineARC.PolicyDecisionMode != policyDecisionModeRaceLost || second.RaylineARC.OfferKind != "lease" {
		t.Fatalf("turn 2 served %s on %s (%s)", second.RaylineARC.PolicyActionID, second.SelectedModel, second.RaylineARC.PolicyDecisionMode)
	}
	if wait := second.RaylineARC.PolicyRaceWait; wait < 30*time.Millisecond || wait > 250*time.Millisecond {
		t.Fatalf("race waited %v", wait)
	}
	// The caller's finalize budget is shorter than the late decide: the
	// store's commit still gets one of its own.
	stageRaylineARCAppendSource(ctx, raceReply(t))
	ctx.RaylineARCTransaction.stageCallUsage(responseUsageMetrics{})
	short, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := ctx.RaylineARCTransaction.commit(short, ctx); err != nil {
		t.Fatal(err)
	}
	pending := ctx.RaylineARCTransaction.policyStaging.pending
	select {
	case <-pending.done:
	default:
		t.Fatal("the commit returned before the late decide finished")
	}
	fixture.fake.mu.Lock()
	startedAfter := append([]int(nil), fixture.fake.appendStartedAfter...)
	fixture.fake.mu.Unlock()
	if len(startedAfter) != 2 || startedAfter[1] != 2 {
		t.Fatalf("appends arrived after %v finished decides; want the second after both", startedAfter)
	}
	state := readLedgerTestStateNoWait(fixture.store, fixture.episode)
	if state == nil || state.TurnIndex != 2 || state.Policy.Ledger[len(state.Policy.Ledger)-1].ActionID != fixture.leased {
		t.Fatalf("committed %+v", state)
	}
	appended := fixture.fake.receivedAppends()[1]
	last := appended.Attribution[len(appended.Attribution)-1]
	if last.Message != 3 || last.ActionID != fixture.leased {
		t.Fatalf("append attribution %+v", appended.Attribution)
	}
	// The late decide's goroutine is gone with it.
	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > baseline && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if now := runtime.NumGoroutine(); now > baseline {
		t.Fatalf("goroutines %d after the turn, %d before", now, baseline)
	}
}

// race_max_new_tokens skips a race the decide cannot win: the lease serves
// at once.
func TestRaceSkippedPastMaxNewTokensServesTheLease(t *testing.T) {
	fixture := newRaceFixture(t, "", func(policy *config.RaylineARCPolicyServiceConfig) {
		policy.RaceMS, policy.RaceMaxNewTokens = 2000, 1
	})
	fixture.stagedFirstTurn(t)
	second, ctx := fixture.turn(t, policyTestRequest(t, raceUser, raceAssistant, raceToolResult))
	if second.RaylineARC.PolicyActionID != fixture.leased || second.RaylineARC.PolicyDecisionMode != policyDecisionModeRaceLost {
		t.Fatalf("turn 2 served %s (%s)", second.RaylineARC.PolicyActionID, second.RaylineARC.PolicyDecisionMode)
	}
	complete(t, ctx, raceReply(t))
}

// A lease that may not serve leaves the turn to its decide, however late.
func TestRaceLeaseRejected(t *testing.T) {
	typed := map[string]any{"role": "user", "content": "now count them"}
	rewritten := map[string]any{"role": "assistant", "content": "something else"}
	for _, tc := range []struct {
		name     string
		schedule string
		body     []map[string]any
		before   func(*raceFixture)
		delta    bool
	}{
		// A typed user message may be a stage-one boundary.
		{name: "boundary", body: []map[string]any{raceUser, raceAssistant, typed}, delta: true},
		// Between schedule boundaries only the held model is offered.
		{name: "unoffered_arm", schedule: config.RaylineARCModelScheduleTaskTurnCompaction,
			body: []map[string]any{raceUser, raceAssistant, raceToolResult}, delta: true},
		{name: "expired_ttl", body: []map[string]any{raceUser, raceAssistant, raceToolResult}, delta: true,
			before: func(fixture *raceFixture) {
				staged, _ := fixture.scorer().staged.get(fixture.episode)
				staged.holdExpires = time.Now().Add(-time.Second)
				fixture.scorer().staged.put(fixture.episode, staged)
			}},
		// The client's history is not the appended one: no delta, no lease.
		{name: "prefix_mismatch", body: []map[string]any{raceUser, rewritten, raceToolResult}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newRaceFixture(t, tc.schedule, nil)
			fixture.stagedFirstTurn(t)
			if tc.before != nil {
				tc.before(fixture)
			}
			slowDeltas(fixture, 150*time.Millisecond)
			second, ctx := fixture.turn(t, policyTestRequest(t, tc.body...))
			if second.RaylineARC.PolicyActionID != fixture.decided || second.RaylineARC.PolicyDecisionMode != policyDecisionModeSync {
				t.Fatalf("served %s (%s), want the decide's", second.RaylineARC.PolicyActionID, second.RaylineARC.PolicyDecisionMode)
			}
			sent := fixture.fake.received()
			if got := sent[len(sent)-1].TurnDelta != nil; got != tc.delta {
				t.Fatalf("delta sent %v, want %v", got, tc.delta)
			}
			complete(t, ctx, raceReply(t))
		})
	}
}

// A failed append fails nothing: the turn commits, and the next turn is
// decided in full.
func TestRaceAppendFailureCommitsAndDecidesTheNextTurnInFull(t *testing.T) {
	fixture := newRaceFixture(t, "", nil)
	fixture.fake.appendStatus = 503
	fixture.stagedFirstTurn(t)
	if state := readLedgerTestStateNoWait(fixture.store, fixture.episode); state == nil || state.TurnIndex != 1 {
		t.Fatalf("turn 1 did not commit: %+v", state)
	}
	if _, ok := fixture.scorer().staged.get(fixture.episode); ok {
		t.Fatal("a failed append left a staged turn")
	}
	second, ctx := fixture.turn(t, policyTestRequest(t, raceUser, raceAssistant, raceToolResult))
	sent := fixture.fake.received()
	if second.RaylineARC.PolicyActionID != fixture.decided || sent[len(sent)-1].TurnDelta != nil {
		t.Fatalf("turn 2 served %s, delta %v", second.RaylineARC.PolicyActionID, sent[len(sent)-1].TurnDelta)
	}
	complete(t, ctx, raceReply(t))
}

// A delta the service holds no staged turn for is sent again in full.
func TestRaceSessionMismatchResendsInFull(t *testing.T) {
	fixture := newRaceFixture(t, "", func(policy *config.RaylineARCPolicyServiceConfig) { policy.RaceMS = 2000 })
	fixture.stagedFirstTurn(t)
	fixture.fake.mu.Lock()
	fixture.fake.refuseDelta = true
	fixture.fake.mu.Unlock()
	second, ctx := fixture.turn(t, policyTestRequest(t, raceUser, raceAssistant, raceToolResult))
	sent := fixture.fake.received()[1:]
	if len(sent) != 2 || sent[0].TurnDelta == nil || sent[1].TurnDelta != nil || sent[1].Request.Shell != "" {
		t.Fatalf("turn 2 decides %+v", sent)
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(sent[1].Request.Messages, &messages); err != nil || len(messages) != 3 {
		t.Fatalf("full resend carried %d messages", len(messages))
	}
	if second.RaylineARC.PolicyActionID != fixture.decided {
		t.Fatalf("turn 2 served %s", second.RaylineARC.PolicyActionID)
	}
	complete(t, ctx, raceReply(t))
}

// The episode lease is held until the append finishes: the next turn cannot
// start before it.
func TestRaceLeaseHeldUntilTheAppendFinishes(t *testing.T) {
	fixture := newRaceFixture(t, "", nil)
	gate := make(chan struct{})
	fixture.fake.appendGate = gate
	_, ctx := fixture.turn(t, policyTestRequest(t, raceUser))
	stageRaylineARCAppendSource(ctx, raceReply(t))
	committed := make(chan error, 1)
	go func() { committed <- ctx.RaylineARCTransaction.commit(context.Background(), ctx) }()
	deadline := time.Now().Add(2 * time.Second)
	for len(fixture.fake.receivedAppends()) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if state := readLedgerTestStateNoWait(fixture.store, fixture.episode); state != nil {
		t.Fatal("the episode was free while its append ran")
	}
	close(gate)
	if err := <-committed; err != nil {
		t.Fatal(err)
	}
	if state := readLedgerTestStateNoWait(fixture.store, fixture.episode); state == nil || state.TurnIndex != 1 {
		t.Fatalf("after the append: %+v", state)
	}
}

// A sync cell sends neither member, posts no append and records no mode.
func TestSyncCellStagesNothing(t *testing.T) {
	fixture := newRaceFixture(t, "", func(policy *config.RaylineARCPolicyServiceConfig) { policy.DecideMode = "" })
	first, ctx := fixture.turn(t, policyTestRequest(t, raceUser))
	complete(t, ctx, raceReply(t))
	body := fixture.fake.receivedBodies()[0]
	if bytes.Contains(body, []byte(`"stage"`)) || bytes.Contains(body, []byte(`"turn_delta"`)) ||
		first.RaylineARC.PolicyDecisionMode != "" || first.RaylineARC.PolicyStaging != nil ||
		len(fixture.fake.receivedAppends()) != 0 {
		t.Fatalf("sync cell staged: %s", body)
	}
	fields := map[string]interface{}{}
	addPolicyDecisionModeFields(fields, first.RaylineARC)
	record := llmUsageRecord{}
	addUsageRecordARCAttribution(&record, &RequestContext{VSRRaylineARC: first.RaylineARC})
	if _, present := record.fields()["decision_mode"]; present || len(fields) != 0 {
		t.Fatalf("sync records carry a decision mode: %v", fields)
	}
}

// A racing cell's dispatch-time records carry the mode and the wait.
func TestRaceDecisionModeReachesTheUsageRecord(t *testing.T) {
	trace := &selection.RaylineARCTrace{PolicyDecisionMode: policyDecisionModeRaceLost, PolicyRaceWait: 100 * time.Millisecond}
	record := llmUsageRecord{}
	addUsageRecordARCAttribution(&record, &RequestContext{VSRRaylineARC: trace})
	fields := record.fields()
	if fields["decision_mode"] != policyDecisionModeRaceLost || fields["race_ms_observed"] != 100.0 {
		t.Fatalf("usage fields %v %v", fields["decision_mode"], fields["race_ms_observed"])
	}
}

// A scheduled package's schedule boundary may change the model, so no lease
// holds through it, even on a tool-result continuation.
func TestRaceLeaseNeverServesAtAScheduleBoundary(t *testing.T) {
	fixture := newRaceFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction, nil)
	messages := []map[string]any{raceUser}
	var served *selection.SelectionResult
	for turn := 0; turn <= 5; turn++ {
		if turn == 5 {
			slowDeltas(fixture, 150*time.Millisecond)
		}
		result, ctx := fixture.turn(t, policyTestRequest(t, messages...))
		served = result
		id := "toolu_" + string(rune('a'+turn))
		content := []any{map[string]any{"type": "tool_use", "id": id, "name": "Bash", "input": map[string]any{"command": "ls"}}}
		reply, _ := json.Marshal(map[string]any{
			"id": "msg", "type": "message", "role": "assistant", "model": "think", "content": content,
			"stop_reason": "tool_use", "stop_sequence": nil, "usage": map[string]any{"input_tokens": 1, "output_tokens": 1},
		})
		complete(t, ctx, reply)
		echoed := []any{map[string]any{"type": "tool_use", "id": id, "caller": map[string]any{"type": "direct"}, "name": "Bash", "input": map[string]any{"command": "ls"}}}
		messages = append(messages, map[string]any{"role": "assistant", "content": echoed},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": id, "content": "ok"}}})
	}
	if served.RaylineARC.PolicyActionID != fixture.decided || served.RaylineARC.PolicyDecisionMode != policyDecisionModeSync {
		t.Fatalf("turn 5 served %s (%s), want the decide's", served.RaylineARC.PolicyActionID, served.RaylineARC.PolicyDecisionMode)
	}
	sent := fixture.fake.received()
	if sent[len(sent)-1].TurnDelta == nil || len(sent[len(sent)-1].Selection.AvailableActionIDs) != 3 {
		t.Fatalf("turn 5 was not a delta at the boundary: %+v", sent[len(sent)-1])
	}
}

// A Responses reply's output items are appended as the next turn
// materializes them: without the router's retention fields.
func TestPolicyAssistantOutputNormalizesResponsesItems(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","status":"completed","output":[` +
		`{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"a <b>"}]},` +
		`{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"ls","arguments":"{}"}]}`)
	items, ok := policyAssistantOutput(policyFormatResponses, body)
	if !ok || len(items) != 2 {
		t.Fatalf("items %s", items)
	}
	for _, item := range items {
		if bytes.Contains(item, []byte(`"status"`)) || bytes.Contains(item, []byte(`"id":"`)) {
			t.Fatalf("retention fields kept: %s", item)
		}
	}
	if !bytes.Contains(items[0], []byte(`"a <b>"`)) || !bytes.Contains(items[1], []byte(`"call_id":"call_1"`)) {
		t.Fatalf("items %s", items)
	}
	roles, err := policyResponsesRoles(items)
	if err != nil || roles[0] != "assistant" || roles[1] != "assistant" {
		t.Fatalf("roles %v %v", roles, err)
	}
}

// The staging inside the lease stays well within the next turn's acquire
// timeout, so it never makes that turn contended.
func TestPolicyStagingBudgetsStayInsideTheAcquireTimeout(t *testing.T) {
	appendTimeout, lateDecide := policyStagingBudgets(60*time.Second, 30*time.Second)
	if appendTimeout != 7500*time.Millisecond || lateDecide != 15*time.Second || appendTimeout+lateDecide >= 30*time.Second {
		t.Fatalf("budgets %v %v", appendTimeout, lateDecide)
	}
	if appendTimeout, lateDecide = policyStagingBudgets(2*time.Second, 30*time.Second); appendTimeout != 2*time.Second || lateDecide != 4*time.Second {
		t.Fatalf("short decide budgets %v %v", appendTimeout, lateDecide)
	}
}

// A raced decide the lease cannot replace keeps a sync decide's budget: the
// staging bound applies only once a lease has served.
func TestRaceDecideTheLeaseCannotReplaceKeepsTheSyncBudget(t *testing.T) {
	fixture := newRaceFixture(t, "", nil)
	fixture.stagedFirstTurn(t)
	if fixture.scorer().lateDecideTimeout >= 3*time.Second {
		t.Fatalf("late decide bound %v; the test needs it under the decide's 3 s", fixture.scorer().lateDecideTimeout)
	}
	slowDeltas(fixture, 3*time.Second)
	// The lease expires during the race, so the turn waits for its decide.
	staged, _ := fixture.scorer().staged.get(fixture.episode)
	staged.holdExpires = time.Now().Add(20 * time.Millisecond)
	fixture.scorer().staged.put(fixture.episode, staged)
	second, ctx := fixture.turn(t, policyTestRequest(t, raceUser, raceAssistant, raceToolResult))
	if second.RaylineARC.PolicyActionID != fixture.decided {
		t.Fatalf("served %s", second.RaylineARC.PolicyActionID)
	}
	complete(t, ctx, raceReply(t))
}

// An append never takes a slot a turn's decide needs: with one slot, an
// append in flight leaves another episode's decide served.
func TestRaceAppendTakesNoDecideSlot(t *testing.T) {
	fixture := newRaceFixture(t, "", func(policy *config.RaylineARCPolicyServiceConfig) { policy.MaxInflightCalls = 1 })
	fixture.selector.armedComponents().admission = raylinearc.NewAdmissionGate(1)
	gate := make(chan struct{})
	fixture.fake.appendGate = gate
	_, ctx := fixture.turn(t, policyTestRequest(t, raceUser))
	stageRaylineARCAppendSource(ctx, raceReply(t))
	ctx.RaylineARCTransaction.stageCallUsage(responseUsageMetrics{})
	committed := make(chan error, 1)
	go func() { committed <- ctx.RaylineARCTransaction.commit(context.Background(), ctx) }()
	deadline := time.Now().Add(2 * time.Second)
	for len(fixture.fake.receivedAppends()) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	other, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("f", 64), RequestID: "req-other", State: other,
			RawRequest: policyTestRequest(t, raceUser), RequestFormat: policyFormatAnthropic,
		},
	}); err != nil {
		t.Fatalf("another episode's decide was shed while an append ran: %v", err)
	}
	close(gate)
	if err := <-committed; err != nil {
		t.Fatal(err)
	}
}

// A decision-only consult (no reply delivered) keeps its caller's deadline:
// one that expired while the late decide ran is not recorded.
func TestRaceExpiredConsultIsNotRecorded(t *testing.T) {
	fixture := newRaceFixture(t, "", nil)
	fixture.stagedFirstTurn(t)
	slowDeltas(fixture, 200*time.Millisecond)
	_, ctx := fixture.turn(t, policyTestRequest(t, raceUser, raceAssistant, raceToolResult))
	expired, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := ctx.RaylineARCTransaction.commit(expired, ctx); err == nil {
		t.Fatal("an expired consult committed")
	}
	if state := readLedgerTestStateNoWait(fixture.store, fixture.episode); state == nil || state.TurnIndex != 1 {
		t.Fatalf("the expired consult advanced the episode: %+v", state)
	}
}

// Claude Code's continuation (testdata/thread_continuation, 2.1.309) follows
// its tool result with a mid-conversation system message: the lease may
// serve it. Its appended reply must reproduce the echo the client sends; one
// echoed otherwise is told apart as the appended reply's mismatch.
func TestRaceClaudeCodeContinuationShapes(t *testing.T) {
	resend := readThreadFixture(t, "claude-code-2.1.309-thread-create-resend.request.json")
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(resend, &body); err != nil {
		t.Fatal(err)
	}
	// The captured tool turn: assistant tool_use, user tool_result, system.
	echoed, toolResult, system := body.Messages[8], body.Messages[9], body.Messages[10]
	if !policyLeaseNewMessagesAllowed(policyFormatAnthropic, []json.RawMessage{toolResult, system}) {
		t.Fatal("Claude Code's tool-result continuation is not lease-eligible")
	}
	if policyLeaseNewMessagesAllowed(policyFormatAnthropic, []json.RawMessage{system}) ||
		policyLeaseNewMessagesAllowed(policyFormatAnthropic, []json.RawMessage{body.Messages[1]}) {
		t.Fatal("a system message without a tool result, or with a tool_addition, is lease-eligible")
	}
	// The reply the captured tool_use came from, as an upstream states it.
	var message struct {
		Content []map[string]json.RawMessage `json:"content"`
	}
	_ = json.Unmarshal(echoed, &message)
	delete(message.Content[0], "caller")
	reply, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": message.Content})
	appended, ok := policyAssistantOutput(policyFormatAnthropic, reply)
	if !ok || raylinearc.MessagesDigest(appended, 1) != raylinearc.MessagesDigest([]json.RawMessage{echoed}, 1) {
		t.Fatalf("appended %s does not reproduce the echo %s", appended, echoed)
	}
	fixture := newRaceFixture(t, "", nil)
	fixture.stagedFirstTurn(t)
	bare := map[string]any{"role": "assistant", "content": raceReplyContent}
	var messages []json.RawMessage
	_ = json.Unmarshal(gjsonMessages(t, policyTestRequest(t, raceUser, bare, raceToolResult)), &messages)
	_, reason := policyDeltaFor(fixture.scorer(), fixture.episode, &raylinearc.EpisodeState{TurnIndex: 1},
		&raylinearc.PolicyEpisodeState{}, raylinearc.PolicyTransitionExtends, policyFormatAnthropic, messages)
	if reason != "appended_mismatch" {
		t.Fatalf("delta reason %q, want appended_mismatch", reason)
	}
}

func gjsonMessages(t *testing.T, body []byte) []byte {
	t.Helper()
	var request struct {
		Messages json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	return request.Messages
}

// A reply with thinking is not appended: no capture shows its echo.
func TestRaceThinkingReplyIsNotAppended(t *testing.T) {
	fixture := newRaceFixture(t, "", nil)
	_, ctx := fixture.turn(t, policyTestRequest(t, raceUser))
	reply, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": []any{
		map[string]any{"type": "thinking", "thinking": "", "signature": "sig"},
		map[string]any{"type": "text", "text": "done"},
	}})
	complete(t, ctx, reply)
	if len(fixture.fake.receivedAppends()) != 0 {
		t.Fatal("a thinking reply was appended")
	}
	if _, ok := fixture.scorer().staged.get(fixture.episode); ok {
		t.Fatal("a thinking reply left a staged turn")
	}
}
