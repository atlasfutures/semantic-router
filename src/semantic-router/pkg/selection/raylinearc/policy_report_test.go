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

package raylinearc

import (
	"bytes"
	"reflect"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

// reportFinished is when the reported calls finished: after the pending
// lookup was decided (pendingNow) and after every recorded turn.
var reportFinished = pendingNow.Add(30 * time.Second)

func intPointer(value int) *int { return &value }

// ranOn is a report run on arm with action, its usage known.
func ranOn(arm int, action, model string) ReportedRun {
	return ReportedRun{
		Arm: arm, ActionID: action, Model: model,
		Call: cachecontext.CallCache{PromptTokens: intPointer(1200), CacheReadTokens: intPointer(1000), TTLSeconds: 300, TTLSource: ReportedTTLSource},
	}
}

// reportOf is a report of run; the decided action it is asked with is
// recorded in asked.
func reportOf(source, routeID string, run ReportedRun, asked *string) Report {
	return Report{
		RouteID: routeID, Source: source, FinishedAt: reportFinished,
		Run: func(decided string) ReportedRun {
			if asked != nil {
				*asked = decided
			}
			return run
		},
	}
}

// pendingEpisode is an executed episode with a lookup's pending decision for
// its second turn: arm 1, action B, route rte_1, reply at message 3.
func pendingEpisode(t *testing.T) *EpisodeState {
	t.Helper()
	return withPendingLookup(t, executedEpisode(t), pendingTask, pendingReply1, pendingAsk2)
}

func mustApplyReport(t *testing.T, state *EpisodeState, report Report) (*EpisodeState, ReportOutcome) {
	t.Helper()
	next, outcome, err := ApplyReport(state, report)
	if err != nil {
		t.Fatal(err)
	}
	return next, outcome
}

// A report naming the pending decision records it as a turn that ran,
// marked reported, with the reported usage as the arm's last call. The model
// mapping is asked with the decided action.
func TestApplyReportResolvesThePendingDecisionAsReported(t *testing.T) {
	stored := pendingEpisode(t)
	var asked string
	next, outcome := mustApplyReport(t, stored, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), &asked))
	if !outcome.Recorded || outcome.Reason != ReportPendingResolved || outcome.RouteID != "rte_1" || outcome.TurnIndex != 2 || !outcome.Attributed {
		t.Fatalf("outcome %+v", outcome)
	}
	if asked != pendingActionB {
		t.Fatalf("the mapping was asked with %q, want the decided action", asked)
	}
	if next.Pending != nil || next.TurnIndex != 2 || next.PreviousArm == nil || *next.PreviousArm != 1 || next.ServedModel != "vendor/b" {
		t.Fatalf("record %+v", next)
	}
	entry := next.Policy.Ledger[len(next.Policy.Ledger)-1]
	if entry.Message != 3 || entry.ActionID != pendingActionB || entry.ArmID != "arm-b" || entry.Basis != PolicyLedgerBasisReported {
		t.Fatalf("entry %+v", entry)
	}
	if next.Policy.PrefixLen != 3 {
		t.Fatalf("prefix %d, want the decision's messages", next.Policy.PrefixLen)
	}
	warmth := next.Warmth[1]
	if !warmth.LastUsed.Equal(reportFinished) || warmth.LastInputTokens != 42 || warmth.Call == nil ||
		warmth.Call.ActionID != pendingActionB || !warmth.Call.CompletedAt.Equal(reportFinished) ||
		*warmth.Call.PromptTokens != 1200 || *warmth.Call.CacheReadTokens != 1000 || warmth.Call.CacheWriteTokens != nil ||
		warmth.Call.TTLSeconds != 300 {
		t.Fatalf("warmth %+v call %+v", warmth, warmth.Call)
	}
	last := next.LastReport
	if last == nil || last.RouteID != "rte_1" || last.Source != ReportSourceClient || last.TurnIndex != 2 ||
		last.Message != 3 || last.Arm != 1 || last.Replaced == nil {
		t.Fatalf("last report %+v", last)
	}
	if stored.Pending == nil || stored.LastReport != nil {
		t.Fatal("the stored state was changed")
	}
}

// A report of another model than the decided one records the turn for the
// model that ran, and leaves the decided arm's warmth as it was. The ledger
// entry names the reported action with no arm id: the service reads that as
// the action's model with no trained-arm identity.
func TestApplyReportOfAnotherModelLeavesTheDecidedArm(t *testing.T) {
	stored := pendingEpisode(t)
	next, outcome := mustApplyReport(t, stored, reportOf(ReportSourceClient, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil))
	if !outcome.Recorded || !outcome.Attributed {
		t.Fatalf("outcome %+v", outcome)
	}
	entry := next.Policy.Ledger[len(next.Policy.Ledger)-1]
	if entry.ActionID != pendingActionA || entry.ArmID != "" || entry.Prices != nil || entry.Basis != PolicyLedgerBasisReported {
		t.Fatalf("entry %+v", entry)
	}
	if *next.PreviousArm != 0 || next.ServedModel != "vendor/a" || !next.Warmth[0].LastUsed.Equal(reportFinished) {
		t.Fatalf("record %+v", next)
	}
	if !reflect.DeepEqual(next.Warmth[1], stored.Warmth[1]) {
		t.Fatalf("the decided arm's warmth changed: %+v", next.Warmth[1])
	}
	raw, roles := rawMessages(t, pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`)
	attribution := next.Policy.Attribution(raw, roles)
	if got := attribution[len(attribution)-1]; got.ActionID != pendingActionA || got.ArmID != nil {
		t.Fatalf("attribution %+v, want the reported action with a null arm id", got)
	}
}

// A run outside the package records the turn with no arm: it is counted,
// the held arm is cleared, no ledger entry and no usage entry are written,
// and the decision's context still moves the prefix.
func TestApplyReportOfAnUnattributedRunCountsTheTurnOnly(t *testing.T) {
	stored := pendingEpisode(t)
	next, outcome := mustApplyReport(t, stored, reportOf(ReportSourceClient, "rte_1", ReportedRun{Arm: -1, Model: "local/llama"}, nil))
	if !outcome.Recorded || outcome.Attributed || outcome.TurnIndex != 2 {
		t.Fatalf("outcome %+v", outcome)
	}
	if next.TurnIndex != 2 || next.PreviousArm != nil || next.ServedModel != "local/llama" || next.Pending != nil {
		t.Fatalf("record %+v", next)
	}
	if len(next.Policy.Ledger) != len(stored.Policy.Ledger) || next.Policy.PrefixLen != 3 {
		t.Fatalf("ledger %+v prefix %d", next.Policy.Ledger, next.Policy.PrefixLen)
	}
	if !reflect.DeepEqual(next.Warmth, stored.Warmth) {
		t.Fatal("an unattributed run changed the warmth")
	}
	if next.LastReport.Arm != -1 || next.LastReport.Replaced != nil {
		t.Fatalf("last report %+v", next.LastReport)
	}
}

// A run whose arm is known but whose action is not (several levels, none
// named) holds the arm and writes no ledger entry.
func TestApplyReportOfAnArmWithoutActionHoldsTheArm(t *testing.T) {
	stored := pendingEpisode(t)
	next, _ := mustApplyReport(t, stored, reportOf(ReportSourceClient, "rte_1", ranOn(1, "", "vendor/b"), nil))
	if len(next.Policy.Ledger) != len(stored.Policy.Ledger) || *next.PreviousArm != 1 || next.Warmth[1].Call.ActionID != "" {
		t.Fatalf("record %+v", next)
	}
}

// Rule (owner's choice): a client report whose route id is no longer the
// pending decision is dropped, never recorded as a new turn. The cell knows
// the decision existed and is gone: it was recorded (inferred), replaced by a
// repeat ask, or dropped, so recording it would count a turn twice.
func TestReportRuleStaleRouteIDIsDropped(t *testing.T) {
	replaced := pendingEpisode(t)
	replaced.Pending.RouteID = "rte_2"
	raw, roles := rawMessages(t, pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`, `{"role":"user","content":"three"}`)
	inferred, resolution := ResolvePending(pendingEpisode(t), NewPrefixDigests(raw), roles)
	if resolution != PendingInferred {
		t.Fatalf("resolution %q", resolution)
	}
	for name, state := range map[string]*EpisodeState{
		"replaced by a repeat ask":     replaced,
		"inferred by the next turn":    inferred,
		"never pending on this record": executedEpisode(t),
	} {
		next, outcome := mustApplyReport(t, state, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
		if outcome.Recorded || outcome.Reason != ReportNotPending || next != state {
			t.Fatalf("%s: outcome %+v", name, outcome)
		}
	}
}

// A report with no route id and no pending decision records a new reported
// turn on a conversation with history, with no ledger entry: it carries no
// messages to say which one is the reply.
func TestApplyReportWithoutPendingRecordsATurn(t *testing.T) {
	stored := executedEpisode(t)
	next, outcome := mustApplyReport(t, stored, reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil))
	if !outcome.Recorded || outcome.Reason != ReportTurnRecorded || outcome.TurnIndex != 2 || outcome.RouteID != "" {
		t.Fatalf("outcome %+v", outcome)
	}
	if next.TurnIndex != 2 || *next.PreviousArm != 1 || !reflect.DeepEqual(next.Policy, stored.Policy) {
		t.Fatalf("record %+v", next)
	}
	if next.Warmth[1].LastInputTokens != 1200 {
		t.Fatalf("input tokens %d, want the reported prompt", next.Warmth[1].LastInputTokens)
	}
	if next.LastReport.Message != -1 || next.LastReport.Source != ReportSourceGateway || next.LastReport.Replaced != nil {
		t.Fatalf("last report %+v", next.LastReport)
	}
}

// Ignored reports change nothing: a conversation with no history, a
// duplicate by route id or by report id, and a call that finished before the
// pending decision was made or before the newest recorded turn.
func TestApplyReportIgnores(t *testing.T) {
	fresh, _ := NewEpisodeState(2)
	resolved, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	byID := reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	byID.ReportID = "rt_edge_1"
	recorded, _ := mustApplyReport(t, executedEpisode(t), byID)
	early := reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	early.FinishedAt = pendingNow.Add(-time.Second)
	beforeTurn := reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	beforeTurn.FinishedAt = pendingNow.Add(-90 * time.Second)
	cases := map[string]struct {
		state  *EpisodeState
		report Report
		want   string
	}{
		"no history":                          {fresh, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportNoEpisode},
		"duplicate route id":                  {resolved, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportDuplicate},
		"duplicate report id":                 {recorded, byID, ReportDuplicate},
		"finished before the pending":         {pendingEpisode(t), early, ReportLate},
		"finished before the newest turn":     {executedEpisode(t), beforeTurn, ReportLate},
		"client after the edge, by route id":  {mustEdgeResolved(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportSuperseded},
		"client after the edge, no route id":  {mustEdgeResolved(t), reportOf(ReportSourceClient, "", ranOn(1, pendingActionB, "vendor/b"), nil), ReportSuperseded},
		"edge again after the edge, route id": {mustEdgeResolved(t), reportOf(ReportSourceGateway, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportDuplicate},
	}
	for name, tc := range cases {
		next, outcome := mustApplyReport(t, tc.state, tc.report)
		if outcome.Recorded || outcome.Reason != tc.want || next != tc.state {
			t.Fatalf("%s: outcome %+v, want ignored %q", name, outcome, tc.want)
		}
	}
}

// mustEdgeResolved is the pending episode after the edge reported its turn.
func mustEdgeResolved(t *testing.T) *EpisodeState {
	t.Helper()
	next, outcome := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil))
	if outcome.Reason != ReportPendingResolved || next.LastReport.RouteID != "rte_1" {
		t.Fatalf("edge resolution %+v", outcome)
	}
	return next
}

// The edge's report wins over a client's for the same turn (here named by
// its route id): arriving after it, it rewrites the turn with what the edge saw and restores the client's
// arm warmth when the arms differ; the turn count does not move. It still
// does after the client's next lookup stored a pending decision, which it
// leaves pending. Once a later turn is recorded, it is late.
func TestEdgeReportReplacesTheClientsForTheSameTurn(t *testing.T) {
	stored := pendingEpisode(t)
	client, _ := mustApplyReport(t, stored, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	var asked string
	edge, outcome := mustApplyReport(t, client, reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), &asked))
	if !outcome.Recorded || outcome.Reason != ReportClientReplaced || outcome.TurnIndex != 2 || outcome.RouteID != "rte_1" {
		t.Fatalf("outcome %+v", outcome)
	}
	if asked != pendingActionB {
		t.Fatalf("the mapping was asked with %q, want the client's action", asked)
	}
	if edge.TurnIndex != 2 || *edge.PreviousArm != 0 || edge.ServedModel != "vendor/a" {
		t.Fatalf("record %+v", edge)
	}
	if len(edge.Policy.Ledger) != len(client.Policy.Ledger) {
		t.Fatalf("ledger %+v", edge.Policy.Ledger)
	}
	if entry := edge.Policy.Ledger[len(edge.Policy.Ledger)-1]; entry.Message != 3 || entry.ActionID != pendingActionA ||
		entry.Basis != PolicyLedgerBasisReported {
		t.Fatalf("entry %+v", entry)
	}
	if !reflect.DeepEqual(edge.Warmth[1], stored.Warmth[1]) {
		t.Fatalf("the client's arm warmth was not restored: %+v", edge.Warmth[1])
	}
	if edge.LastReport.Source != ReportSourceGateway || edge.LastReport.Replaced != nil || edge.LastReport.Arm != 0 {
		t.Fatalf("last report %+v", edge.LastReport)
	}

	// The client's next lookup stored a pending decision; the edge's late
	// report for the turn before still replaces the client's, and the next
	// decision stays pending.
	nextLookup := withPendingLookup(t, client, pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`, `{"role":"user","content":"three"}`)
	nextLookup.LastReport = client.LastReport
	nextLookup.Pending.RouteID = "rte_2"
	nextLookup.Pending.DecidedUnixMS = reportFinished.Add(time.Minute).UnixMilli()
	replaced, outcome := mustApplyReport(t, nextLookup, reportOf(ReportSourceGateway, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	if outcome.Reason != ReportClientReplaced || replaced.Pending == nil || replaced.Pending.RouteID != nextLookup.Pending.RouteID {
		t.Fatalf("outcome %+v pending %+v", outcome, replaced.Pending)
	}

	// A later turn recorded: the client's version stands.
	later := cloneEpisodeState(client)
	if err := later.Commit(0, 5, reportFinished.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	_, outcome = mustApplyReport(t, later, reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil))
	if outcome.Recorded || outcome.Reason != ReportLate {
		t.Fatalf("after a later turn: %+v", outcome)
	}
}

// v12 is written exactly when a record carries a reported ledger entry or a
// last report, round trips, and leaves every other record's bytes as they
// were.
func TestEpisodeSchemaV12CarriesReports(t *testing.T) {
	inferredOnly := pendingEpisode(t)
	payload, err := marshalEpisodeState(inferredOnly, 3, pendingNow)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(payload, []byte(episodeStateSchemaV11)) || bytes.Contains(payload, []byte("last_report")) {
		t.Fatalf("a record without a report is not v11: %s", payload)
	}
	reported, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	payload, err = marshalEpisodeState(reported, 4, pendingNow)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(payload, []byte(episodeStateSchemaV12)) {
		t.Fatalf("a reported record is not v12: %s", payload)
	}
	read, version, err := unmarshalEpisodeState(payload, 2, pendingNow)
	if err != nil || version != 4 {
		t.Fatalf("read back: %v (version %d)", err, version)
	}
	reported.ServedModel = ""
	if !reflect.DeepEqual(read, reported) {
		t.Fatalf("round trip:\n got %+v\nwant %+v\nlast %+v vs %+v", read, reported, read.LastReport, reported.LastReport)
	}
	// A v11 schema on a reported record is refused, as is v12 on one without.
	if _, _, err := unmarshalEpisodeState(bytes.Replace(payload, []byte(episodeStateSchemaV12), []byte(episodeStateSchemaV11), 1), 2, pendingNow); err == nil {
		t.Fatal("a reported record read as v11")
	}
}

// Validation refuses a last report the cell could not have written.
func TestLastReportValidation(t *testing.T) {
	reported, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	mutations := map[string]func(*LastReport){
		"unknown source":       func(last *LastReport) { last.Source = "admin" },
		"arm out of range":     func(last *LastReport) { last.Arm = 2 },
		"turn after the state": func(last *LastReport) { last.TurnIndex = 3 },
		"message below -1":     func(last *LastReport) { last.Message = -2 },
		"long route id":        func(last *LastReport) { last.RouteID = string(make([]byte, 129)) },
		"replaced on the edge": func(last *LastReport) { last.Source = ReportSourceGateway },
	}
	for name, mutate := range mutations {
		state := cloneEpisodeState(reported)
		mutate(state.LastReport)
		if _, err := marshalEpisodeState(state, 4, pendingNow); err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}

// A report is the same turn as the last report only when it names its
// route id, its report id, or finished at the same moment; otherwise it is
// an ordinary report. A gateway report of a later pinned call after a
// client's report records a new turn rather than replacing the client's,
// and an older one is late.
func TestReportPrecedenceNeedsTheSameTurn(t *testing.T) {
	client, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	later := reportOf(ReportSourceGateway, "", ranOn(0, pendingActionA, "vendor/a"), nil)
	later.ReportID, later.FinishedAt = "rt_pinned_9", reportFinished.Add(time.Minute)
	next, outcome := mustApplyReport(t, client, later)
	if outcome.Reason != ReportTurnRecorded || next.TurnIndex != 3 {
		t.Fatalf("a later pinned call: %+v, turn %d", outcome, next.TurnIndex)
	}
	older := reportOf(ReportSourceGateway, "", ranOn(0, pendingActionA, "vendor/a"), nil)
	older.FinishedAt = reportFinished.Add(-time.Second)
	if _, outcome := mustApplyReport(t, client, older); outcome.Recorded || outcome.Reason != ReportLate {
		t.Fatalf("an older gateway report: %+v", outcome)
	}
}

// A gateway report naming the client report's report id is the same turn:
// it replaces the client's, rather than being dropped as a duplicate.
func TestGatewayReportWithTheClientsReportIDReplacesIt(t *testing.T) {
	byClient := reportOf(ReportSourceClient, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	byClient.ReportID = "call-2"
	client, _ := mustApplyReport(t, pendingEpisode(t), byClient)
	byGateway := reportOf(ReportSourceGateway, "", ranOn(0, pendingActionA, "vendor/a"), nil)
	byGateway.ReportID, byGateway.FinishedAt = "call-2", reportFinished.Add(-time.Second)
	next, outcome := mustApplyReport(t, client, byGateway)
	if outcome.Reason != ReportClientReplaced || *next.PreviousArm != 0 || next.TurnIndex != client.TurnIndex {
		t.Fatalf("outcome %+v record %+v", outcome, next)
	}
}

// A retried report with neither id is recognised by its completion time:
// the identical body is a duplicate, not a second turn.
func TestRetriedReportWithoutIDsIsADuplicate(t *testing.T) {
	retry := reportOf(ReportSourceClient, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	first, outcome := mustApplyReport(t, pendingEpisode(t), retry)
	if outcome.Reason != ReportPendingResolved {
		t.Fatalf("first: %+v", outcome)
	}
	if next, outcome := mustApplyReport(t, first, retry); outcome.Recorded || outcome.Reason != ReportDuplicate || next != first {
		t.Fatalf("retry: %+v", outcome)
	}
}

// A turn recorded with no arm keeps its completion time, so a report of a
// call that finished before it is late, though it finished after every
// arm's last use.
func TestUnattributedTurnKeepsItsTimeForTheLateCheck(t *testing.T) {
	local := reportOf(ReportSourceGateway, "", ReportedRun{Arm: -1, Model: "local/llama"}, nil)
	local.ReportID = "rt_local"
	recorded, outcome := mustApplyReport(t, executedEpisode(t), local)
	if outcome.Reason != ReportTurnRecorded || recorded.LastReport.FinishedUnixMS != reportFinished.UnixMilli() {
		t.Fatalf("outcome %+v last %+v", outcome, recorded.LastReport)
	}
	before := reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	before.ReportID, before.FinishedAt = "rt_before", reportFinished.Add(-time.Second)
	if _, outcome := mustApplyReport(t, recorded, before); outcome.Recorded || outcome.Reason != ReportLate {
		t.Fatalf("a report from before the unattributed turn: %+v", outcome)
	}
}

// A superseded pending decision -- a generation call through the cell
// decided its turn and records it -- is not reportable.
func TestSupersededPendingDecisionIsNotReportable(t *testing.T) {
	stored := pendingEpisode(t)
	stored.Pending.Superseded = true
	for _, routeID := range []string{"rte_1", ""} {
		next, outcome := mustApplyReport(t, stored, reportOf(ReportSourceClient, routeID, ranOn(1, pendingActionB, "vendor/b"), nil))
		if outcome.Recorded || outcome.Reason != ReportNotPending || next != stored {
			t.Fatalf("route %q: %+v", routeID, outcome)
		}
	}
}

// A decision whose request ended with a prefill is reported at the
// prefill's place, with the prefix the decision saw before it.
func TestReportOfAPrefillDecisionTakesThePrefillsPlace(t *testing.T) {
	stored := withPendingLookup(t, executedEpisode(t), pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"Sure:"}`)
	raw, _ := rawMessages(t, pendingTask, pendingReply1, pendingAsk2)
	stored.Pending.PrefillBaseDigest = NewPrefixDigests(raw).Digest(3)
	next, outcome := mustApplyReport(t, stored, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	if outcome.Reason != ReportPendingResolved {
		t.Fatalf("outcome %+v", outcome)
	}
	entry := next.Policy.Ledger[len(next.Policy.Ledger)-1]
	if entry.Message != 3 || next.Policy.PrefixLen != 3 || next.Policy.PrefixDigest != stored.Pending.PrefillBaseDigest ||
		next.LastReport.Message != 3 {
		t.Fatalf("entry %+v prefix %d last %+v", entry, next.Policy.PrefixLen, next.LastReport)
	}
}

// Two reports with different report ids are different calls, even when they
// finished in the same millisecond.
func TestReportsWithDifferentReportIDsAreDifferentCalls(t *testing.T) {
	first := reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	first.ReportID = "rt_pinned_1"
	recorded, _ := mustApplyReport(t, executedEpisode(t), first)
	second := first
	second.ReportID = "rt_pinned_2"
	if next, outcome := mustApplyReport(t, recorded, second); outcome.Reason != ReportTurnRecorded || next.TurnIndex != recorded.TurnIndex+1 {
		t.Fatalf("outcome %+v", outcome)
	}
}

// clientThenNextLookup is a client's report of the first turn (report id
// call-1, no route id), then the next turn's lookup pending as rte_2,
// decided after the report.
func clientThenNextLookup(t *testing.T) *EpisodeState {
	t.Helper()
	byClient := reportOf(ReportSourceClient, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	byClient.ReportID = "call-1"
	client, _ := mustApplyReport(t, pendingEpisode(t), byClient)
	next := withPendingLookup(t, client, pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`, `{"role":"user","content":"three"}`)
	next.LastReport, next.Pending.RouteID = client.LastReport, "rte_2"
	next.Pending.DecidedUnixMS = reportFinished.Add(time.Minute).UnixMilli()
	next.Pending.PriorServedModel = client.ServedModel
	return next
}

// A gateway report naming the last report's id corrects that turn, even
// when it finished after the next lookup was decided: the identity is
// matched before the report is applied to the pending decision, which stays
// pending, and whose prior served model is the corrected one.
func TestReportOfTheLastReportedCallIsMatchedBeforeThePendingDecision(t *testing.T) {
	stored := clientThenNextLookup(t)
	byGateway := reportOf(ReportSourceGateway, "", ranOn(0, pendingActionA, "vendor/a"), nil)
	byGateway.ReportID, byGateway.FinishedAt = "call-1", reportFinished.Add(2*time.Minute)
	next, outcome := mustApplyReport(t, stored, byGateway)
	if outcome.Reason != ReportClientReplaced || next.TurnIndex != stored.TurnIndex || next.Pending == nil || next.Pending.RouteID != "rte_2" {
		t.Fatalf("outcome %+v record %+v", outcome, next)
	}
	if next.Pending.PriorServedModel != "vendor/a" {
		t.Fatalf("the pending decision's prior served model is %q, want the corrected model", next.Pending.PriorServedModel)
	}
}

// A replacement keeps the client report's identity when the gateway brings
// none: the client's retry is then superseded, not a new turn.
func TestReplacementKeepsTheClientReportIdentity(t *testing.T) {
	byClient := reportOf(ReportSourceClient, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	byClient.ReportID = "call-1"
	client, _ := mustApplyReport(t, pendingEpisode(t), byClient)
	byGateway := reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil)
	byGateway.FinishedAt = reportFinished.Add(-time.Second)
	replaced, outcome := mustApplyReport(t, client, byGateway)
	if outcome.Reason != ReportClientReplaced {
		t.Fatalf("replacement %+v", outcome)
	}
	if next, outcome := mustApplyReport(t, replaced, byClient); outcome.Recorded || outcome.Reason != ReportSuperseded || next != replaced {
		t.Fatalf("the client's retry: %+v", outcome)
	}
}

// A superseded pending decision blocks only reports of that decision: the
// gateway's correction of the turn before it still applies.
func TestSupersededPendingDoesNotBlockACorrectionOfTheTurnBefore(t *testing.T) {
	stored := clientThenNextLookup(t)
	stored.Pending.Superseded = true
	byGateway := reportOf(ReportSourceGateway, "", ranOn(0, pendingActionA, "vendor/a"), nil)
	byGateway.ReportID = "call-1"
	if _, outcome := mustApplyReport(t, stored, byGateway); outcome.Reason != ReportClientReplaced {
		t.Fatalf("outcome %+v", outcome)
	}
	if _, outcome := mustApplyReport(t, stored, reportOf(ReportSourceClient, "rte_2", ranOn(0, pendingActionA, "vendor/a"), nil)); outcome.Reason != ReportNotPending {
		t.Fatalf("a report of the superseded decision: %+v", outcome)
	}
}
