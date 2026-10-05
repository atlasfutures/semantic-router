//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fallbackFixture is boundaryFixture with ADR 0120's fallback switched to
// enabled, and the "off" worker's action, which its service prefers.
func fallbackFixture(t *testing.T, enabled bool) (*policySelectorFixture, *raylinearc.MemoryEpisodeStore, string, string) {
	t.Helper()
	fixture, store, episode := boundaryFixture(t)
	fixture.decision.Algorithm.RaylineARC.PolicyService.Fallback = &config.RaylineARCPolicyFallbackConfig{Enabled: enabled}
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second}),
	})
	offAction := ""
	for _, binding := range fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings {
		if binding.Worker == "off" {
			offAction = binding.ActionID
		}
	}
	return fixture, store, episode, offAction
}

// With the fallback on, a refused turn excludes its model for the rest of
// the context: the retry is offered only the other model, and the
// exclusion is committed with the turn that served, so later turns keep it.
// Control: with the fallback off, the retry is offered the refusing model
// again, and the service picks it again.
func TestRaylineARCFallbackExcludesTheRefusingModel(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		fixture, store, episode, offAction := fallbackFixture(t, enabled)
		body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
		first, decided := boundaryAttempt(t, fixture, store, episode, body)
		if decided.RaylineARC.PolicyActionID != offAction {
			t.Fatalf("enabled=%v: session start chose %s", enabled, decided.RaylineARC.PolicyActionID)
		}
		first.VSRRaylineARC, first.VSRSelectedDecision = decided.RaylineARC, fixture.decision
		declineRefusedTurn(first)
		if err := first.RaylineARCTransaction.abort(context.Background(), "test"); err != nil {
			t.Fatal(err)
		}
		if excluded := readLedgerTestState(t, store, episode).Policy.Excludes("vendor/off"); excluded != enabled {
			t.Fatalf("enabled=%v: stored exclusion=%v", enabled, excluded)
		}

		retry, retried := boundaryAttempt(t, fixture, store, episode, body)
		offered := offeredActions(fixture, 1)
		if slices.Contains(offered, offAction) == enabled || (retried.RaylineARC.SelectedArm == 1) == enabled {
			t.Fatalf("enabled=%v: retry offered %v, chose arm %d", enabled, offered, retried.RaylineARC.SelectedArm)
		}
		if !enabled {
			continue
		}
		retry.RaylineARCTransaction.markPolicyState(retried.RaylineARC.PolicyNextState, false)
		if err := retry.RaylineARCTransaction.commit(context.Background(), retry); err != nil {
			t.Fatal(err)
		}
		committed := readLedgerTestStateNoWait(store, episode)
		if committed == nil || committed.TurnIndex != 1 || !committed.Policy.Excludes("vendor/off") {
			t.Fatalf("after the served retry: %+v", committed)
		}
	}
}

// A held model that the context excluded is not held: its turn decides
// again among the rest, stores that as a boundary decision, and logs the
// re-decision with what Phase 2 learns from. Control: with the fallback
// off the held model is offered alone, excluded or not.
func TestRaylineARCFallbackRedecidesOffAnExcludedHeldModel(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		fixture, _, _, offAction := fallbackFixture(t, enabled)
		fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			return request.Selection.AvailableActionIDs[0]
		})
		state := heldEpisode(t)
		off := 1
		state.PreviousArm = &off
		state.Policy = state.Policy.WithExclusion("vendor/off", "refusal")
		logs := captureLogs(t)
		next := policyTestRequest(t,
			map[string]any{"role": "user", "content": "fix the bug"},
			map[string]any{"role": "assistant", "content": "done"},
			map[string]any{"role": "user", "content": "now the tests"})
		result, err := fixture.selectOn(t, state, next)
		if err != nil {
			t.Fatal(err)
		}
		offered := offeredActions(fixture, 0)
		if enabled {
			if slices.Contains(offered, offAction) || len(offered) != 2 || result.RaylineARC.SelectedArm != 0 ||
				result.RaylineARC.PolicyBoundary == nil || fixture.fake.received()[0].Selection.HeldActionID != nil {
				t.Fatalf("offered %v, arm %d, boundary %+v", offered, result.RaylineARC.SelectedArm, result.RaylineARC.PolicyBoundary)
			}
			event := findLogEvent(t, logs, "rayline_arc_fallback_decision")
			raw, _ := json.Marshal(event)
			for _, want := range []string{`"reason":"fallback_redecide"`, `"model":"vendor/off"`, `"class":"refusal"`, `"turn_index":2`, `"scores":[`, `"request_id":"req-policy-test"`, `"selected_action_id":"` + offered[0] + `"`} {
				if !strings.Contains(string(raw), want) {
					t.Fatalf("the re-decision log lacks %s: %s", want, raw)
				}
			}
			continue
		}
		if len(offered) != 1 || offered[0] != offAction || result.RaylineARC.PolicyBoundary != nil {
			t.Fatalf("fallback off: offered %v, boundary %+v", offered, result.RaylineARC.PolicyBoundary)
		}
	}
}

// A refusal on the turn that compacted excludes the model in the compacted
// context the retry continues, not in the one the compaction left. When the
// store has moved on since the turn read it, the exclusion follows it only
// to this very turn committed by a coalesced copy; any other state, even a
// sibling context with the same epoch number, starts with the full offer.
func TestRaylineARCRefusalExcludesInTheTurnsContext(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	sibling := []json.RawMessage{json.RawMessage(`{"role":"user","content":"something else"}`)}
	longer := append(append([]json.RawMessage(nil), task...), json.RawMessage(`{"role":"assistant","content":"done"}`))
	read := (&raylinearc.PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
	compacted := &raylinearc.PolicyEpisodeState{Epoch: 1, EpochStartTurn: 3, CompactionCount: 1}
	thisTurn := compacted.Next(task, strings.Repeat("b", 64), "arm-b")
	for name, tc := range map[string]struct {
		stored   *raylinearc.PolicyEpisodeState
		excludes bool
		epoch    int
	}{
		"store unchanged":                  {read, true, 1},
		"this turn, committed by its copy": {thisTurn, true, 1},
		"a sibling context, same epoch":    {compacted.Next(sibling, strings.Repeat("c", 64), "arm-c"), false, 1},
		"another context":                  {read.Next(longer, strings.Repeat("d", 64), "arm-d"), false, 0},
	} {
		state, _ := raylinearc.NewEpisodeState(2)
		state.Policy = tc.stored
		refusal := refusedTurn{arm: -1, exclude: "vendor/off", context: compacted, decidedFrom: read, committed: thisTurn}
		next, changed := refusal.apply(state)
		if changed != tc.excludes || next.Policy.Excludes("vendor/off") != tc.excludes || next.Policy.Epoch != tc.epoch {
			t.Fatalf("%s: changed=%v stored policy %+v", name, changed, next.Policy)
		}
	}
}

// A late refusal keeps what the store excluded since its turn read it: it
// lands in the turn's context without lifting a newer exclusion.
func TestRaylineARCLateRefusalKeepsNewerExclusions(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	read := (&raylinearc.PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
	state, _ := raylinearc.NewEpisodeState(2)
	state.Policy = read.WithExclusion("vendor/think", turnFailureRefusal)
	next, changed := refusedTurn{arm: -1, exclude: "vendor/off", context: read.Clone(), decidedFrom: read}.apply(state)
	if !changed || !next.Policy.Excludes("vendor/off") || !next.Policy.Excludes("vendor/think") {
		t.Fatalf("exclusions after the late refusal: %+v", next.Policy.Exclusions)
	}
}

// A coalesced resend refused while the leader holds the lease hands its
// refusal over; when the leader's own turn succeeds, the exclusion is
// committed with it. Control: with no refusal handed over, the commit
// excludes nothing.
func TestRaylineARCHandedOverRefusalCommitsWithTheLeader(t *testing.T) {
	for _, refusal := range []bool{true, false} {
		fixture, store, episode, _ := fallbackFixture(t, true)
		leader, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		entry := &raylineARCInflightEntry{done: make(chan struct{}), finished: make(chan struct{})}
		leader.RaylineARCTransaction.inflight = entry
		if refusal {
			declineRefusedTurn(&RequestContext{
				RaylineARCTransaction: newBorrowedRaylineARCEpisodeTransaction(store, leader.RaylineARCTransaction.state, episode, entry),
				VSRRaylineARC:         decided.RaylineARC,
				VSRSelectedDecision:   fixture.decision,
			})
		}
		leader.RaylineARCTransaction.markPolicyState(decided.RaylineARC.PolicyNextState, false)
		if err := leader.RaylineARCTransaction.commit(context.Background(), leader); err != nil {
			t.Fatal(err)
		}
		committed := readLedgerTestStateNoWait(store, episode)
		if committed == nil || committed.TurnIndex != 1 || committed.Policy.Excludes("vendor/off") != refusal {
			t.Fatalf("refusal=%v: committed %+v", refusal, committed)
		}
	}
}

// The loader's model bound is the episode's exclusion bound: every model a
// fallback-enabled package serves can be excluded.
func TestRaylineARCFallbackModelBoundMatchesTheEpisode(t *testing.T) {
	if config.RaylineARCMaxFallbackModels != raylinearc.MaxPolicyExclusions ||
		config.RaylineARCMaxFallbackModelBytes != raylinearc.MaxPolicyExclusionModelBytes {
		t.Fatalf("loader bounds %d models of %d bytes, episode bounds %d of %d",
			config.RaylineARCMaxFallbackModels, config.RaylineARCMaxFallbackModelBytes,
			raylinearc.MaxPolicyExclusions, raylinearc.MaxPolicyExclusionModelBytes)
	}
}

// A leader that never takes a hand-over (a relaxed or side-call turn) refuses
// it, so the refused resend stages for itself once the leader finishes.
// Control: a strict leader takes it.
func TestRaylineARCRefusalIsNotHandedToALeaderThatCannotTakeIt(t *testing.T) {
	_, store, episode := boundaryFixture(t)
	state, read, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	relaxed := newRelaxedRaylineARCEpisodeTransaction(store, state, read, episode, false)
	if relaxed.takesHandovers() || !(&raylineARCEpisodeTransaction{}).takesHandovers() {
		t.Fatal("a relaxed leader takes hand-overs, or a strict one does not")
	}
	for _, takes := range []bool{true, false} {
		entry := &raylineARCInflightEntry{done: make(chan struct{}), finished: make(chan struct{})}
		leader := &raylineARCEpisodeTransaction{inflight: entry}
		if !takes {
			leader.markPolicyState(nil, true)
		}
		if handed := entry.noteRefusal(refusedTurn{arm: 1, exclude: "vendor/off"}); handed != takes {
			t.Fatalf("leader takes=%v: handed over=%v", takes, handed)
		}
	}
}

// A sibling context a concurrent request opened from the same state, with the
// same messages but another compaction, is not the refused turn.
func TestRaylineARCLateRefusalTellsCompactionSiblingsApart(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	read := (&raylinearc.PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
	thisTurn := (&raylinearc.PolicyEpisodeState{Epoch: 1, EpochStartTurn: 3, CompactionCount: 1, CompactionSummary: strings.Repeat("1", 64)}).
		Next(task, strings.Repeat("b", 64), "arm-b")
	for name, sibling := range map[string]*raylinearc.PolicyEpisodeState{
		"compaction count": {Epoch: 1, EpochStartTurn: 3, CompactionCount: 2, CompactionSummary: strings.Repeat("1", 64)},
		"summary":          {Epoch: 1, EpochStartTurn: 3, CompactionCount: 1, CompactionSummary: strings.Repeat("2", 64)},
		"start turn":       {Epoch: 1, EpochStartTurn: 4, CompactionCount: 1, CompactionSummary: strings.Repeat("1", 64)},
	} {
		state, _ := raylinearc.NewEpisodeState(2)
		state.Policy = sibling.Next(task, strings.Repeat("c", 64), "arm-c")
		refusal := refusedTurn{arm: -1, exclude: "vendor/off", context: thisTurn, decidedFrom: read, committed: thisTurn}
		if next, changed := refusal.apply(state); changed || next.Policy.Excludes("vendor/off") {
			t.Fatalf("%s sibling took the late refusal", name)
		}
	}
}

// A provider that reports its refusal as an error (Alibaba's
// data_inspection_failed) is a refusal like any other: with the fallback on,
// the model is excluded, even though the turn was aborted at the error's
// headers. Control: an ordinary provider error excludes no model.
func TestRaylineARCProviderErrorRefusalExcludesTheModel(t *testing.T) {
	for _, body := range []string{
		`{"error":{"code":"data_inspection_failed","message":"Input data may contain inappropriate content."}}`,
		`{"error":{"code":"invalid_request","message":"bad request"}}`,
	} {
		refusal := strings.Contains(body, "data_inspection_failed")
		fixture, store, episode, _ := fallbackFixture(t, true)
		attempt, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		attempt.RequestID, attempt.RequestModel, attempt.UpstreamStatusCode = "req-provider-refusal", "off", 400
		attempt.SourceFormat, attempt.TargetFormat = llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1
		attempt.StartTime, attempt.VSRRaylineARC, attempt.VSRSelectedDecision = time.Now(), decided.RaylineARC, fixture.decision
		// The non-2xx headers abort the turn before its body is read.
		if err := attempt.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
			t.Fatal(err)
		}
		(&OpenAIRouter{}).handleUpstreamTransportError([]byte(body), attempt)
		if excluded := readLedgerTestState(t, store, episode).Policy.Excludes("vendor/off"); excluded != refusal {
			t.Fatalf("refusal=%v: excluded=%v", refusal, excluded)
		}
	}
}

// A refused turn that opened a new context excludes only its own model there:
// what the old context excluded stays behind. Control: a refused turn that
// stayed in the context keeps the earlier exclusion beside its own.
func TestRaylineARCRefusalInANewContextLeavesOldExclusions(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	read := (&raylinearc.PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a").WithExclusion("vendor/think", turnFailureRefusal)
	for name, tc := range map[string]struct {
		context    *raylinearc.PolicyEpisodeState
		keepsThink bool
	}{
		"opened a new context": {&raylinearc.PolicyEpisodeState{Epoch: 1, EpochStartTurn: 3, CompactionCount: 1}, false},
		"stayed in it":         {read.Clone(), true},
	} {
		state, _ := raylinearc.NewEpisodeState(2)
		state.Policy = read
		next, _ := refusedTurn{arm: -1, exclude: "vendor/off", context: tc.context, decidedFrom: read}.apply(state)
		if !next.Policy.Excludes("vendor/off") || next.Policy.Excludes("vendor/think") != tc.keepsThink {
			t.Fatalf("%s: exclusions %+v", name, next.Policy.Exclusions)
		}
	}
}

// Every path that classes a turn's failure as a refusal declines it, once:
// a provider's in-stream refusal error reaches recordTurnFailure only, and
// still excludes the model. Control: a stream cut excludes nothing.
func TestRaylineARCStreamedRefusalErrorExcludesTheModel(t *testing.T) {
	for _, class := range []string{turnFailureRefusal, turnFailureStreamCut} {
		fixture, store, episode, _ := fallbackFixture(t, true)
		attempt, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		attempt.VSRRaylineARC, attempt.VSRSelectedDecision = decided.RaylineARC, fixture.decision
		recordTurnFailure(attempt, class, true)
		if err := attempt.RaylineARCTransaction.abort(context.Background(), "stream_error"); err != nil {
			t.Fatal(err)
		}
		if excluded := readLedgerTestState(t, store, episode).Policy.Excludes("vendor/off"); excluded != (class == turnFailureRefusal) {
			t.Fatalf("%s: excluded=%v", class, excluded)
		}
	}
}

// A finished coalescing entry takes no hand-over: its leader is gone, so the
// refused request stages for itself.
func TestRaylineARCFinishedEntryRefusesHandovers(t *testing.T) {
	registry := &raylineARCInflightRegistry{entries: map[string]*raylineARCInflightEntry{}}
	entry, leader := registry.join("key")
	if !leader {
		t.Fatal("the first request did not lead")
	}
	if !entry.noteRefusal(refusedTurn{arm: 1}) {
		t.Fatal("a running strict leader refused the hand-over")
	}
	registry.finish(entry)
	if entry.noteRefusal(refusedTurn{arm: 1}) {
		t.Fatal("a finished entry took a hand-over")
	}
}

// A turn whose request did not grow the recorded prefix still commits a
// ledger entry; folding a duplicate's refusal into that commit keeps it.
func TestRaylineARCRefusalKeepsTheCommittedTurnsLedger(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	read := (&raylinearc.PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
	committed := read.Next(task, strings.Repeat("b", 64), "arm-b")
	state, _ := raylinearc.NewEpisodeState(2)
	state.Policy = committed
	next, changed := refusedTurn{arm: -1, exclude: "vendor/off", context: read.Clone(), decidedFrom: read, committed: committed}.apply(state)
	if !changed || !next.Policy.Excludes("vendor/off") || len(next.Policy.Ledger) != len(committed.Ledger) {
		t.Fatalf("ledger %d entries after the fold, want %d", len(next.Policy.Ledger), len(committed.Ledger))
	}
}

// A refusal staged after its leader finished waits out a newer lease holder
// within its whole bound, not a short one.
func TestRaylineARCRefusalOutwaitsANewerLeaseHolder(t *testing.T) {
	fixture, store, episode, _ := fallbackFixture(t, true)
	attempt, decided := boundaryAttempt(t, fixture, store, episode,
		policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
	attempt.VSRRaylineARC, attempt.VSRSelectedDecision = decided.RaylineARC, fixture.decision
	entry := &raylineARCInflightEntry{done: make(chan struct{}), finished: make(chan struct{})}
	attempt.RaylineARCTransaction.inflight = entry
	if err := attempt.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	entry.refuseHandovers()
	entry.finishOnce.Do(func() { close(entry.finished) })
	// A newer turn holds the lease for well over the short wait.
	lease, _, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		time.Sleep(6 * relaxedBoundaryStageTimeout)
		_ = store.Abort(context.Background(), lease)
	}()
	declineRefusedTurn(attempt)
	if !readLedgerTestState(t, store, episode).Policy.Excludes("vendor/off") {
		t.Fatal("the refusal was dropped while a newer turn held the lease")
	}
}

// resetCellExclusions gives a test an empty process table and restores it.
func resetCellExclusions(t *testing.T) {
	t.Helper()
	saved := raylineARCWorkerExclusions
	raylineARCWorkerExclusions = &raylineARCCellExclusions{worker: map[string]cellExclusion{}}
	t.Cleanup(func() { raylineARCWorkerExclusions = saved })
}

// A turn that failed for capacity or availability excludes its route, when
// its decision serves with the fallback on. Controls: the fallback off, and
// a failure that is about the conversation rather than the route.
func TestRaylineARCRouteFailureExcludesTheRoute(t *testing.T) {
	for _, tc := range []struct {
		class    string
		enabled  bool
		excludes bool
	}{
		{turnFailureRateLimited, true, true},
		{turnFailureUpstream5xx, true, true},
		{turnFailureTimeout, true, true},
		{turnFailureNoEndpoint, true, false},
		{turnFailureRateLimited, false, false},
		{turnFailureRefusal, true, false},
		{turnFailureContextOverflow, true, false},
	} {
		resetCellExclusions(t)
		fixture, _, _, _ := fallbackFixture(t, tc.enabled)
		fixture.decision.Algorithm.RaylineARC.PolicyService.Fallback.CellExclusionSeconds = 60
		ctx := &RequestContext{
			RequestID: "req-route", VSRSelectedDecision: fixture.decision,
			VSRRaylineARC: &selection.RaylineARCTrace{SelectedArm: 1, WorkerProviderModel: "off", WorkerRoute: fallbackTestRoute(fixture, 1)},
		}
		recordTurnFailure(ctx, tc.class, false)
		_, excluded := raylineARCWorkerExclusions.active(fallbackTestRoute(fixture, 1), time.Now())
		_, expired := raylineARCWorkerExclusions.active(fallbackTestRoute(fixture, 1), time.Now().Add(61*time.Second))
		if excluded != tc.excludes || expired {
			t.Fatalf("%s enabled=%v: excluded=%v, after the TTL=%v", tc.class, tc.enabled, excluded, expired)
		}
	}
}

// A route the cell excluded leaves the offer, and a turn held on it decides
// again; once the exclusion expires it is offered again. Control: the
// fallback off holds the route regardless.
func TestRaylineARCRouteExclusionNarrowsTheOfferUntilItExpires(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		resetCellExclusions(t)
		fixture, _, _, offAction := fallbackFixture(t, enabled)
		fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			return request.Selection.AvailableActionIDs[0]
		})
		now := time.Date(2026, 10, 3, 12, 0, 0, 0, time.UTC)
		fixture.selector.now = func() time.Time { return now }
		raylineARCWorkerExclusions.exclude(fallbackTestRoute(fixture, 1), turnFailureRateLimited, now.Add(30*time.Second))
		logs := captureLogs(t)
		state := heldEpisode(t)
		off := 1
		state.PreviousArm = &off
		next := policyTestRequest(t,
			map[string]any{"role": "user", "content": "fix the bug"},
			map[string]any{"role": "assistant", "content": "done"},
			map[string]any{"role": "user", "content": "now the tests"})
		result, err := fixture.selectOn(t, state, next)
		if err != nil {
			t.Fatal(err)
		}
		offered := offeredActions(fixture, 0)
		if !enabled {
			if len(offered) != 1 || offered[0] != offAction {
				t.Fatalf("fallback off: offered %v", offered)
			}
			continue
		}
		if slices.Contains(offered, offAction) || result.RaylineARC.SelectedArm != 0 || result.RaylineARC.PolicyBoundary == nil {
			t.Fatalf("offered %v, arm %d, boundary %+v", offered, result.RaylineARC.SelectedArm, result.RaylineARC.PolicyBoundary)
		}
		raw, _ := json.Marshal(findLogEvent(t, logs, "rayline_arc_fallback_decision"))
		if !strings.Contains(string(raw), `"excluded_routes":[{"class":"rate_limited","worker":"off"}]`) ||
			!strings.Contains(string(raw), `"reason":"fallback_redecide"`) {
			t.Fatalf("the re-decision log lacks the route: %s", raw)
		}
		now = now.Add(31 * time.Second)
		if _, err := fixture.selectOn(t, heldEpisodeOn(t, 1), next); err != nil {
			t.Fatal(err)
		}
		if offered := offeredActions(fixture, 1); len(offered) != 1 || offered[0] != offAction {
			t.Fatalf("after expiry: offered %v", offered)
		}
	}
}

// Route exclusions are advice: when every route is excluded, the turn is
// offered as though none were, rather than failing until they expire. A
// model the episode excluded stays excluded either way.
func TestRaylineARCRouteExclusionsNeverEmptyTheOffer(t *testing.T) {
	resetCellExclusions(t)
	fixture, _, _, offAction := fallbackFixture(t, true)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	until := time.Now().Add(time.Minute)
	raylineARCWorkerExclusions.exclude(fallbackTestRoute(fixture, 1), turnFailureUpstream5xx, until)
	raylineARCWorkerExclusions.exclude(fallbackTestRoute(fixture, 0), turnFailureTimeout, until)
	logs := captureLogs(t)
	task := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	fresh, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selectOn(t, fresh, task); err != nil {
		t.Fatal(err)
	}
	if offered := offeredActions(fixture, 0); len(offered) != 3 {
		t.Fatalf("every route excluded: offered %v", offered)
	}
	// The routes set aside are still the reason this decision was logged.
	raw, _ := json.Marshal(findLogEvent(t, logs, "rayline_arc_fallback_decision"))
	if !strings.Contains(string(raw), `{"class":"upstream_5xx","worker":"off"}`) || !strings.Contains(string(raw), `{"class":"timeout","worker":"think"}`) {
		t.Fatalf("the bypassed routes are not logged: %s", raw)
	}
	excluded, _ := raylinearc.NewEpisodeState(2)
	excluded.Policy = (*raylinearc.PolicyEpisodeState)(nil).WithExclusion("vendor/think", turnFailureRefusal)
	if _, err := fixture.selectOn(t, excluded, task); err != nil {
		t.Fatal(err)
	}
	if offered := offeredActions(fixture, 1); len(offered) != 1 || offered[0] != offAction {
		t.Fatalf("with think refused: offered %v", offered)
	}
}

// heldEpisodeOn is heldEpisode held on arm.
func heldEpisodeOn(t *testing.T, arm int) *raylinearc.EpisodeState {
	t.Helper()
	state := heldEpisode(t)
	state.PreviousArm = &arm
	return state
}

// A route's exclusion is the route's: the same worker moved to another
// endpoint (a config reload) is offered at once.
func TestRaylineARCRouteExclusionFollowsTheProviderModel(t *testing.T) {
	resetCellExclusions(t)
	fixture, _, _, offAction := fallbackFixture(t, true)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	// The same worker and provider model, on the endpoint it had before.
	before := strings.Replace(fallbackTestRoute(fixture, 1), "off\x00", "off\x00old-endpoint", 1)
	raylineARCWorkerExclusions.exclude(before, turnFailureRateLimited, time.Now().Add(time.Minute))
	fresh, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selectOn(t, fresh, policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})); err != nil {
		t.Fatal(err)
	}
	if offered := offeredActions(fixture, 0); !slices.Contains(offered, offAction) {
		t.Fatalf("a rebound worker stayed excluded: offered %v", offered)
	}
}

// End to end through the response path: a provider's 429 on the turn's
// route, classed where every failed turn is, takes that route out of the
// next turn's offer. Control: a provider error that is not about the route
// (a 400) leaves the offer whole.
func TestRaylineARCProviderRateLimitChangesTheNextOffer(t *testing.T) {
	for _, status := range []int{429, 400} {
		resetCellExclusions(t)
		fixture, _, _, offAction := fallbackFixture(t, true)
		task := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
		fresh, _ := raylinearc.NewEpisodeState(2)
		decided, err := fixture.selectOn(t, fresh, task)
		if err != nil || decided.RaylineARC.PolicyActionID != offAction {
			t.Fatalf("status %d: first turn chose %v, %v", status, decided, err)
		}
		ctx := &RequestContext{
			RequestID: "req-provider-error", RequestModel: "off", UpstreamStatusCode: status,
			SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
			StartTime: time.Now(), TraceContext: context.Background(),
			VSRRaylineARC: decided.RaylineARC, VSRSelectedDecision: fixture.decision,
		}
		(&OpenAIRouter{}).handleUpstreamTransportError([]byte(fmt.Sprintf(`{"error":{"message":"Provider returned error","code":%d}}`, status)), ctx)
		again, _ := raylinearc.NewEpisodeState(2)
		if _, err := fixture.selectOn(t, again, task); err != nil {
			t.Fatal(err)
		}
		if offersOff := slices.Contains(offeredActions(fixture, 1), offAction); offersOff == (status == 429) {
			t.Fatalf("status %d: the next offer %v", status, offeredActions(fixture, 1))
		}
	}
}

// fallbackTestRoute is the route the fixture's armed scorer names for arm.
func fallbackTestRoute(fixture *policySelectorFixture, arm int) string {
	return fixture.selector.armedComponents().scorer.(*policyServiceScorer).routes[arm]
}

// A reload builds new scorers, which start with no route excluded, even when
// nothing the route's name sees has changed (a credential, a base path).
// Control: the scorer that recorded the exclusion still honours it.
func TestRaylineARCRouteExclusionsEndWithTheirScorer(t *testing.T) {
	resetCellExclusions(t)
	fixture, _, _, offAction := fallbackFixture(t, true)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	raylineARCWorkerExclusions.exclude(fallbackTestRoute(fixture, 1), turnFailureRateLimited, time.Now().Add(time.Minute))
	task := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	for call, reloaded := range []bool{false, true} {
		if reloaded {
			fixture.selector.arm(&raylineARCArmedComponents{
				scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
				admission: raylinearc.NewAdmissionGate(0),
				policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second}),
			})
		}
		fresh, _ := raylinearc.NewEpisodeState(2)
		if _, err := fixture.selectOn(t, fresh, task); err != nil {
			t.Fatal(err)
		}
		if offered := slices.Contains(offeredActions(fixture, call), offAction); offered != reloaded {
			t.Fatalf("reloaded=%v: offered %v", reloaded, offeredActions(fixture, call))
		}
	}
}

// A worker's route changes with its endpoint and with its provider pin.
func TestPolicyWorkerRouteNamesTheEndpoint(t *testing.T) {
	ref := config.ModelRef{Model: "kimi"}
	route := func(endpoint string, order []string) string {
		params := config.ModelParams{PreferredEndpoints: []string{endpoint}}
		if order != nil {
			params.ProviderPreferences = &config.OpenRouterProviderPreferences{Order: order}
		}
		return policyWorkerRoute(&config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"kimi": params},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "modal-a", Address: "10.0.0.1", Port: 8000, Weight: 1},
				{Name: "modal-b", Address: "10.0.0.2", Port: 8000, Weight: 1},
			},
		}}, ref)
	}
	base := route("modal-a", nil)
	if base == route("modal-b", nil) {
		t.Fatal("one route for two endpoints")
	}
	if base == route("modal-a", []string{"moonshotai"}) || route("modal-a", []string{"moonshotai"}) == route("modal-a", []string{"groq"}) {
		t.Fatal("one route for two provider pins")
	}
	if base != route("modal-a", nil) {
		t.Fatal("the same route named twice differently")
	}
}

// A route no scorer looks up again (a reload removed it) leaves the table
// once it has expired, at the next insertion.
func TestRaylineARCCellExclusionsSweepExpiredRoutes(t *testing.T) {
	resetCellExclusions(t)
	raylineARCWorkerExclusions.exclude("removed-route", turnFailureRateLimited, time.Now().Add(-time.Second))
	raylineARCWorkerExclusions.exclude("live-route", turnFailureRateLimited, time.Now().Add(time.Minute))
	if _, kept := raylineARCWorkerExclusions.worker["removed-route"]; kept || len(raylineARCWorkerExclusions.worker) != 1 {
		t.Fatalf("table after the insertion: %v", raylineARCWorkerExclusions.worker)
	}
}

// A worker Envoy balances across several endpoints is never route-excluded:
// its failure cannot be pinned on one endpoint. Control: the same worker on
// one endpoint has a route.
func TestPolicyWorkerRouteSkipsBalancedWorkers(t *testing.T) {
	endpoints := []config.VLLMEndpoint{
		{Name: "modal-a", Address: "10.0.0.1", Port: 8000, Weight: 1},
		{Name: "modal-b", Address: "10.0.0.2", Port: 8000, Weight: 1},
	}
	for preferred, routed := range map[string]bool{"one": true, "both": false} {
		params := config.ModelParams{PreferredEndpoints: []string{"modal-a"}}
		if preferred == "both" {
			params.PreferredEndpoints = []string{"modal-a", "modal-b"}
		}
		cfg := &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"kimi": params}, VLLMEndpoints: endpoints,
		}}
		if route := policyWorkerRoute(cfg, config.ModelRef{Model: "kimi"}); (route != "") != routed {
			t.Fatalf("%s endpoint(s): route %q", preferred, route)
		}
	}
}

// The tool-loop family hold yields to a model exclusion. Mid-loop on the
// "off" arm, whose family has no other arm, the context excluded "off": the
// hold would leave nothing to offer, so it is lifted and the turn decides
// among the other family rather than failing.
func TestRaylineARCFallbackLiftsTheToolLoopHoldOffAnExcludedFamily(t *testing.T) {
	fixture, _, _, offAction := fallbackFixture(t, true)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	state := heldEpisodeOn(t, 1)
	state.Policy = state.Policy.WithExclusion("vendor/off", "refusal")
	next := policyTestRequest(t,
		map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "done"},
		map[string]any{"role": "user", "content": "now the tests"})
	logs := captureLogs(t)
	result, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), RequestID: "req-policy-test", State: state, RawRequest: next,
			RequestFormat: policyFormatAnthropic, ToolLoopForeignArms: []bool{true, false},
		},
	})
	if err != nil {
		t.Fatalf("select: %v, want the hold lifted", err)
	}
	offered := offeredActions(fixture, 0)
	if slices.Contains(offered, offAction) || len(offered) != 2 || result.RaylineARC.SelectedArm != 0 {
		t.Fatalf("offered %v, arm %d; want the think worker's two levels", offered, result.RaylineARC.SelectedArm)
	}
	// One outcome per turn: the lift replaces the preliminary hold.
	if outcomes := toolLoopHoldOutcomes(logs); !slices.Equal(outcomes, []string{"lifted_by_exclusion"}) {
		t.Fatalf("hold outcomes = %v, want only lifted_by_exclusion", outcomes)
	}
}

// The hold yields to a route exclusion before the exclusion is relaxed.
// Mid-loop on the "off" arm, whose family has no other arm, the cell
// excluded "off"'s route while the other family's route is healthy: the turn
// leaves the family rather than going back to the route that just failed.
func TestRaylineARCFallbackLiftsTheToolLoopHoldOffAnExcludedRoute(t *testing.T) {
	resetCellExclusions(t)
	fixture, _, _, offAction := fallbackFixture(t, true)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	raylineARCWorkerExclusions.exclude(fallbackTestRoute(fixture, 1), turnFailureRateLimited, time.Now().Add(time.Minute))
	next := policyTestRequest(t,
		map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "done"},
		map[string]any{"role": "user", "content": "now the tests"})
	result, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), RequestID: "req-policy-test", State: heldEpisodeOn(t, 1), RawRequest: next,
			RequestFormat: policyFormatAnthropic, ToolLoopForeignArms: []bool{true, false},
		},
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	offered := offeredActions(fixture, 0)
	if slices.Contains(offered, offAction) || len(offered) != 2 || result.RaylineARC.SelectedArm != 0 {
		t.Fatalf("offered %v, arm %d; want the think worker's two levels", offered, result.RaylineARC.SelectedArm)
	}
}
