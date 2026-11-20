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

// Rule (owner's choice): a report naming no route id is used only to settle
// the conversation's pending decision. With nothing pending -- a pinned call
// with no lookup before it, a second report of a call already reported, a
// retry, a helper call -- it changes nothing and is answered not_pending:
// without a route id the cell cannot tell a new turn from one it has.
func TestReportRuleWithoutRouteIDOnlySettlesThePendingDecision(t *testing.T) {
	resolved, outcome := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil))
	if outcome.Reason != ReportPendingResolved || outcome.RouteID != "rte_1" {
		t.Fatalf("with a pending decision: %+v", outcome)
	}
	clientResolved, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	early := reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	early.FinishedAt = pendingNow.Add(-time.Second)
	cases := map[string]struct {
		state  *EpisodeState
		report Report
	}{
		"nothing pending, a later pinned call":             {executedEpisode(t), reportOf(ReportSourceGateway, "", ranOn(0, pendingActionA, "vendor/a"), nil)},
		"the gateway's copy of a call the client reported": {clientResolved, reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)},
		"the client's copy of a call the gateway reported": {resolved, reportOf(ReportSourceClient, "", ranOn(1, pendingActionB, "vendor/b"), nil)},
		"a retry with neither id":                          {resolved, reportOf(ReportSourceGateway, "", ranOn(1, pendingActionB, "vendor/b"), nil)},
		"a call that finished before the pending decision": {pendingEpisode(t), early},
	}
	for name, tc := range cases {
		next, outcome := mustApplyReport(t, tc.state, tc.report)
		if outcome.Recorded || outcome.Reason != ReportNotPending || next != tc.state {
			t.Fatalf("%s: %+v", name, outcome)
		}
	}
}

// Rule (owner's choice): the gateway's report replaces a client's only when
// both name the same route id. A matching report id does not: nothing hands
// the gateway a client's report id, and a report id only recognises a
// retry of the same report.
func TestReportRuleGatewayReplacesOnlyByRouteID(t *testing.T) {
	byClient := reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil)
	byClient.ReportID = "call-1"
	client, _ := mustApplyReport(t, pendingEpisode(t), byClient)
	sameReportID := reportOf(ReportSourceGateway, "", ranOn(0, pendingActionA, "vendor/a"), nil)
	sameReportID.ReportID = "call-1"
	if next, outcome := mustApplyReport(t, client, sameReportID); outcome.Recorded || outcome.Reason != ReportNotPending || next != client {
		t.Fatalf("by report id: %+v", outcome)
	}
	if _, outcome := mustApplyReport(t, client, reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil)); outcome.Reason != ReportClientReplaced {
		t.Fatalf("by route id: %+v", outcome)
	}
}

// Rule (owner's choice): a report id recognises a retry of the same report
// from the same reporter, answered duplicate, and is matched before the
// pending decision: a retry is never applied to a decision made after it.
func TestReportRuleReportIDRecognisesARetry(t *testing.T) {
	stored := clientThenNextLookup(t)
	retry := reportOf(ReportSourceClient, "", ranOn(1, pendingActionB, "vendor/b"), nil)
	retry.ReportID, retry.FinishedAt = "call-1", reportFinished.Add(2*time.Minute)
	next, outcome := mustApplyReport(t, stored, retry)
	if outcome.Recorded || outcome.Reason != ReportDuplicate || next != stored {
		t.Fatalf("outcome %+v", outcome)
	}
}

// A correction of the turn before a later lookup's pending decision leaves
// that decision pending and updates the served model it restores when it
// is dropped; a superseded later decision does not block the correction.
func TestGatewayCorrectionBeforeALaterPendingDecision(t *testing.T) {
	for _, superseded := range []bool{false, true} {
		stored := clientThenNextLookup(t)
		stored.Pending.Superseded = superseded
		next, outcome := mustApplyReport(t, stored, reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil))
		if outcome.Reason != ReportClientReplaced || next.TurnIndex != stored.TurnIndex || next.Pending == nil ||
			next.Pending.RouteID != "rte_2" || next.Pending.PriorServedModel != "vendor/a" {
			t.Fatalf("superseded=%t: outcome %+v pending %+v", superseded, outcome, next.Pending)
		}
	}
}

// Ignored reports change nothing: no history, a duplicate by route id, a
// client's report after the gateway's, and the gateway's again.
func TestApplyReportIgnores(t *testing.T) {
	fresh, _ := NewEpisodeState(2)
	resolved, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	cases := map[string]struct {
		state  *EpisodeState
		report Report
		want   string
	}{
		"no history":                          {fresh, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportNoEpisode},
		"duplicate route id":                  {resolved, reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportDuplicate},
		"client after the edge, by route id":  {mustEdgeResolved(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportSuperseded},
		"edge again after the edge, route id": {mustEdgeResolved(t), reportOf(ReportSourceGateway, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil), ReportDuplicate},
	}
	for name, tc := range cases {
		next, outcome := mustApplyReport(t, tc.state, tc.report)
		if outcome.Recorded || outcome.Reason != tc.want || next != tc.state {
			t.Fatalf("%s: outcome %+v, want ignored %q", name, outcome, tc.want)
		}
	}
}

// A reported call is stamped with the context epoch the turn is recorded
// in, as a served turn's call is, and a gateway correction keeps it.
func TestReportedCallCarriesTheContextEpoch(t *testing.T) {
	client, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	if call := client.Warmth[1].Call; call.ContextEpoch != client.Policy.ContextEpoch() {
		t.Fatalf("call epoch %q, want %q", call.ContextEpoch, client.Policy.ContextEpoch())
	}
	edge, _ := mustApplyReport(t, client, reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil))
	if call := edge.Warmth[0].Call; call.ContextEpoch != client.Policy.ContextEpoch() {
		t.Fatalf("corrected call epoch %q, want %q", call.ContextEpoch, client.Policy.ContextEpoch())
	}
}

// A gateway correction keeps the turn's input-token count, also when the
// client's report recorded no arm to read it from.
func TestGatewayCorrectionKeepsTheTurnsInputTokens(t *testing.T) {
	client, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ReportedRun{Arm: -1, Model: "local/llama"}, nil))
	edge, outcome := mustApplyReport(t, client, reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil))
	if outcome.Reason != ReportClientReplaced || edge.Warmth[0].LastInputTokens != 42 {
		t.Fatalf("outcome %+v tokens %d, want the decision's 42", outcome, edge.Warmth[0].LastInputTokens)
	}
}

// A gateway correction after the conversation moved to another policy
// context (a refused request with a rewritten history staged one) corrects
// the arm and model but writes no ledger entry into the new context, whose
// messages the old position does not describe; the record stays valid.
func TestGatewayCorrectionAfterAContextChange(t *testing.T) {
	client, _ := mustApplyReport(t, pendingEpisode(t), reportOf(ReportSourceClient, "rte_1", ranOn(1, pendingActionB, "vendor/b"), nil))
	moved := cloneEpisodeState(client)
	moved.Policy = &PolicyEpisodeState{Epoch: client.Policy.Epoch + 1}
	edge, outcome := mustApplyReport(t, moved, reportOf(ReportSourceGateway, "rte_1", ranOn(0, pendingActionA, "vendor/a"), nil))
	if outcome.Reason != ReportClientReplaced || *edge.PreviousArm != 0 || edge.ServedModel != "vendor/a" || len(edge.Policy.Ledger) != 0 {
		t.Fatalf("outcome %+v record %+v ledger %+v", outcome, edge, edge.Policy.Ledger)
	}
	if _, err := marshalEpisodeState(edge, 5, pendingNow); err != nil {
		t.Fatalf("the corrected record does not validate: %v", err)
	}
}
