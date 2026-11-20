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
	"errors"
	"math"
	"strconv"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

// PolicyLedgerBasisReported marks a ledger entry recorded from a report of
// what ran: a caller that ran the model itself said which one, after a route
// lookup.
const PolicyLedgerBasisReported = "reported"

// Report sources. The gateway in front of the cell reports the calls it sent
// to a provider itself; a client reports the turns it ran. When both report
// one turn, the gateway's report wins.
const (
	ReportSourceGateway = "gateway"
	ReportSourceClient  = "client"
)

// ReportedTTLSource is the TTL source of a call's cache lifetime taken from a
// report rather than from the request the cell sent.
const ReportedTTLSource = "reported"

// MaxReportIDBytes bounds the ids a report may carry and a last report
// keeps: a route id, as a pending decision keeps it, and a caller's report
// id.
const MaxReportIDBytes = maxPendingRouteIDBytes

// Report outcomes, as ApplyReport reports them.
const (
	// ReportPendingResolved: the pending decision was recorded as a turn
	// that ran, for the reported model.
	ReportPendingResolved = "pending_resolved"
	// ReportClientReplaced: the gateway's report replaced a client's for the
	// same turn.
	ReportClientReplaced = "client_report_replaced"
	// ReportNoEpisode: the conversation has no history; nothing was written.
	ReportNoEpisode = "no_episode"
	// ReportNotPending: the decision the report names is no longer pending,
	// or, naming none, there is no pending decision it could settle.
	ReportNotPending = "not_pending"
	// ReportDuplicate: the same report was already recorded.
	ReportDuplicate = "duplicate"
	// ReportSuperseded: the gateway already reported this turn.
	ReportSuperseded = "superseded"
)

// ReportedRun is what a report says ran, mapped onto the package.
type ReportedRun struct {
	// Arm is the policy arm that served the model, -1 when no arm of the
	// package does: the turn is then recorded with no arm.
	Arm int
	// ActionID is the package action, empty when the model is the arm's but
	// the action is not known.
	ActionID string
	// Model is the model the turn records as served.
	Model string
	// Call is the reported usage; ApplyReport sets its action and time.
	Call cachecontext.CallCache
}

// Report is one report of a call that ran.
type Report struct {
	// RouteID names the decision the report is for; empty applies it to the
	// latest pending decision.
	RouteID  string
	ReportID string
	Source   string
	// FinishedAt is when the call's response was complete.
	FinishedAt time.Time
	// Run maps what ran onto the package, given the action already recorded
	// or decided for the turn ("" for none): a report that names only a
	// model is read as that action when the model is its own.
	Run func(decidedAction string) ReportedRun
}

// ReportOutcome is what a report did to the episode.
type ReportOutcome struct {
	Recorded   bool
	Reason     string
	RouteID    string
	TurnIndex  uint64
	Attributed bool
}

// LastReport is the episode's most recent recorded report: what a later
// report of the same turn is matched against (a duplicate, or the gateway's
// report replacing a client's).
type LastReport struct {
	RouteID  string `json:"route_id,omitempty"`
	ReportID string `json:"report_id,omitempty"`
	// ClientReportID is, after the gateway replaced a client's report, the
	// client report's id, so the client's retry of it is still recognised.
	ClientReportID string `json:"client_report_id,omitempty"`
	Source         string `json:"source"`
	// TurnIndex is the episode's turn count right after the report.
	TurnIndex uint64 `json:"turn_index"`
	// Message is the ledger message the report recorded, -1 for none.
	Message int `json:"message"`
	// Arm is the arm the report recorded, -1 for none.
	Arm int `json:"arm"`
	// Decided is the decision's ledger entry for the turn (its action and
	// trained arm), which a correction of the turn resolves against; nil
	// when the decision recorded none.
	Decided *PolicyLedgerEntry `json:"decided,omitempty"`
	// Epoch is the policy context Message is a position in.
	Epoch int `json:"epoch"`
	// InputTokens is the turn's encoded input tokens, as its decision
	// recorded them, which a correction of the turn writes again.
	InputTokens int `json:"input_tokens"`
	// Replaced is, for a client's report on an arm, that arm's warmth
	// before it: the gateway's report for the turn restores it when it names
	// another arm.
	Replaced *ReplacedWarmth `json:"replaced,omitempty"`
}

// ReplacedWarmth is an arm's warmth before a client's report; Warmth is nil
// when the arm had none.
type ReplacedWarmth struct {
	Warmth *episodeWarmthWire `json:"warmth"`
}

// CloneLastReport is a copy the caller may change.
func CloneLastReport(last *LastReport) *LastReport {
	if last == nil {
		return nil
	}
	cloned := *last
	if last.Decided != nil {
		decided := *last.Decided
		decided.Prices = clonePriceAudit(decided.Prices)
		cloned.Decided = &decided
	}
	if last.Replaced != nil {
		replaced := ReplacedWarmth{Warmth: cloneWarmthWire(last.Replaced.Warmth)}
		cloned.Replaced = &replaced
	}
	return &cloned
}

func (last *LastReport) validate(state *EpisodeState) error {
	if last == nil {
		return nil
	}
	if (last.Source != ReportSourceGateway && last.Source != ReportSourceClient) ||
		len(last.RouteID) > MaxReportIDBytes || len(last.ReportID) > MaxReportIDBytes ||
		len(last.ClientReportID) > MaxReportIDBytes || (last.ClientReportID != "" && last.Source != ReportSourceGateway) ||
		last.TurnIndex > state.TurnIndex || last.Message < -1 || last.Arm < -1 || last.Arm >= len(state.Warmth) ||
		last.InputTokens < 0 || last.Epoch < 0 ||
		(last.Replaced != nil && (last.Source != ReportSourceClient || last.Arm < 0)) ||
		(last.Decided != nil && (!isLowerHex64(last.Decided.ActionID) || len(last.Decided.ArmID) > maxPolicyArmIDBytes)) {
		return errors.New("ARC episode last report is malformed")
	}
	return nil
}

// EpisodeHasHistory reports whether an episode records anything a report
// could apply to: a turn, a pending decision, or a policy context. A fresh
// state, which a read of an absent record returns, has none.
func EpisodeHasHistory(state *EpisodeState) bool {
	return state != nil &&
		(state.TurnIndex > 0 || state.Pending != nil || state.Policy != nil || state.PreviousArm != nil)
}

// ApplyReport is the episode after a report of a call that ran. state itself
// is never changed; it is returned as is when the report is ignored.
//
// The pending decision the report names becomes a recorded turn marked
// reported, for the model that ran. A report naming the decision the last
// report recorded is a duplicate, or -- the gateway's after a client's --
// replaces the client's: the gateway's report wins. A recorded turn after
// it clears the last report, so it is replaced only while it is the
// latest turn. A report naming no decision is used only to settle the
// pending decision (unnamedReport).
func ApplyReport(state *EpisodeState, report Report) (*EpisodeState, ReportOutcome, error) {
	if !EpisodeHasHistory(state) {
		return state, ignoredReport(ReportNoEpisode), nil
	}
	pending, last := state.Pending, state.LastReport
	if report.RouteID == "" {
		return unnamedReport(state, report)
	}
	switch {
	case pending != nil && pending.RouteID == report.RouteID:
		return resolveReported(state, report)
	case last != nil && last.RouteID == report.RouteID:
		return reportOfReportedTurn(state, report)
	}
	return state, staleRouteReport(), nil
}

// unnamedReport is the rule for a report naming no route id: it is used
// only to settle the pending decision, if the call finished after that
// decision was made. Anything else is not_pending, never a new turn: with no
// position in the conversation, the cell cannot tell a call it has not seen
// from a second report of one it has (the gateway's copy of a client's
// report, a retry, a helper call). A retry of the last report -- its report
// id, from the same reporter -- is a duplicate, matched first so it is never
// applied to a decision made after it.
func unnamedReport(state *EpisodeState, report Report) (*EpisodeState, ReportOutcome, error) {
	pending, last := state.Pending, state.LastReport
	if last != nil && report.ReportID != "" {
		if last.ReportID == report.ReportID && last.Source == report.Source {
			return state, ignoredReport(ReportDuplicate), nil
		}
		if last.ClientReportID == report.ReportID && report.Source == ReportSourceClient {
			return state, ignoredReport(ReportDuplicate), nil
		}
	}
	if pending != nil && !report.FinishedAt.Before(time.UnixMilli(pending.DecidedUnixMS)) {
		return resolveReported(state, report)
	}
	return state, ignoredReport(ReportNotPending), nil
}

// staleRouteReport is the rule for a report naming a decision that is no
// longer pending: dropped, never recorded as a new turn. The decision existed
// and is gone -- recorded as inferred by the next turn, replaced by a repeat
// ask, or dropped -- so recording the report would count a turn twice.
func staleRouteReport() ReportOutcome {
	return ignoredReport(ReportNotPending)
}

func ignoredReport(reason string) ReportOutcome {
	return ReportOutcome{Reason: reason}
}

// reportOfReportedTurn answers a report naming the route id the last report
// recorded -- the episode's latest turn, as a commit clears the last report:
// from the same source a duplicate; a client's after the gateway's is
// superseded; the gateway's after a client's replaces it.
func reportOfReportedTurn(state *EpisodeState, report Report) (*EpisodeState, ReportOutcome, error) {
	last := state.LastReport
	switch {
	case last.Source == report.Source:
		return state, ignoredReport(ReportDuplicate), nil
	case report.Source != ReportSourceGateway:
		return state, ignoredReport(ReportSuperseded), nil
	}
	return replaceClientReport(state, report)
}

// resolveReported records the pending decision as a turn that ran, for the
// reported run, in the context the decision was made in.
func resolveReported(state *EpisodeState, report Report) (*EpisodeState, ReportOutcome, error) {
	pending := state.Pending
	// A generation call through the cell decided the superseded decision's
	// turn itself and records it; a report of it would count it twice.
	if pending.Superseded {
		return state, ignoredReport(ReportNotPending), nil
	}
	decided := ""
	if pending.Policy != nil {
		decided = pending.Policy.Entry.ActionID
	}
	run := report.Run(decided)
	next := cloneEpisodeState(state)
	message := -1
	if pending.Policy != nil {
		message, _ = pending.reportedReply()
		next.Policy = pending.reportedPolicy(state.Policy, reportedEntry(pending.Policy.Entry, run))
	}
	replaced, err := commitReportedRun(next, run, pending.InputTokens, report)
	if err != nil {
		return state, ReportOutcome{}, err
	}
	next.LastReport = newLastReport(pending.RouteID, report, next.TurnIndex, message, run.Arm, replaced)
	next.LastReport.InputTokens = pending.InputTokens
	if next.Policy != nil {
		next.LastReport.Epoch = next.Policy.Epoch
	}
	if pending.Policy != nil {
		decided := pending.Policy.Entry
		decided.Prices = clonePriceAudit(decided.Prices)
		next.LastReport.Decided = &decided
	}
	return next, recordedReport(ReportPendingResolved, pending.RouteID, next, run), nil
}

// replaceClientReport rewrites the turn a client reported with the
// gateway's report of it. The turn count does not move; the client's arm
// gets back the warmth it had before the client's report when the gateway
// names another arm.
func replaceClientReport(state *EpisodeState, report Report) (*EpisodeState, ReportOutcome, error) {
	last := state.LastReport
	next := cloneEpisodeState(state)
	// The turn's ledger position is rewritten only in the context it was
	// recorded in: after a context change it describes no message of the
	// current one.
	sameContext := next.Policy != nil && last.Message >= 0 && next.Policy.Epoch == last.Epoch &&
		next.Policy.PrefixLen == last.Message
	if ledger := ledgerOf(next.Policy); sameContext && len(ledger) > 0 &&
		ledger[len(ledger)-1].Message == last.Message && ledger[len(ledger)-1].Basis == PolicyLedgerBasisReported {
		next.Policy.Ledger = ledger[:len(ledger)-1]
	}
	// The correction is resolved against the decision, as the gateway's
	// report would have been had it arrived first.
	decided := PolicyLedgerEntry{Message: last.Message}
	if last.Decided != nil {
		decided = *last.Decided
		decided.Message = last.Message
	}
	run := report.Run(decided.ActionID)
	if run.Arm >= len(next.Warmth) {
		return state, ReportOutcome{}, errors.New("reported arm is out of range")
	}
	if last.Arm >= 0 && last.Arm != run.Arm && last.Replaced != nil {
		next.Warmth[last.Arm] = warmthFromWire(last.Replaced.Warmth)
	}
	if entry := reportedEntry(decided, run); sameContext && entry != nil {
		next.Policy = next.Policy.withLedgerEntry(*entry, next.Policy.PrefixLen, next.Policy.PrefixDigest)
	}
	next.ServedModel = run.Model
	next.PreviousArm = nil
	if run.Arm >= 0 {
		arm := run.Arm
		next.PreviousArm = &arm
		next.Warmth[arm] = &cachecontext.WorkerWarmth{
			LastUsed: report.FinishedAt.UTC(), LastInputTokens: last.InputTokens, Call: reportedCall(run, report, last.contextEpoch()),
		}
	}
	// A decision pending after the corrected turn read the served model the
	// client's report left; a request that drops it restores that record.
	if next.Pending != nil {
		next.Pending.PriorServedModel = run.Model
	}
	next.LastReport = newLastReport(last.RouteID, report, state.TurnIndex, last.Message, run.Arm, nil)
	next.LastReport.InputTokens, next.LastReport.Epoch = last.InputTokens, last.Epoch
	next.LastReport.Decided = last.Decided
	next.LastReport.ClientReportID = last.ReportID
	return next, recordedReport(ReportClientReplaced, last.RouteID, next, run), nil
}

func ledgerOf(policy *PolicyEpisodeState) []PolicyLedgerEntry {
	if policy == nil {
		return nil
	}
	return policy.Ledger
}

// reportedEntry is the ledger entry of a reported run on a decision's turn:
// the decided entry when the run is the decided action, an entry for the
// action that ran otherwise (no price audit: the decision's prices scored
// another action), none when the action is not known.
func reportedEntry(decided PolicyLedgerEntry, run ReportedRun) *PolicyLedgerEntry {
	if run.ActionID == "" {
		return nil
	}
	// No decision named the trained arm of an action other than the decided
	// one: the entry has none, which the service reads as the action's
	// model only.
	entry := PolicyLedgerEntry{Message: decided.Message, ActionID: run.ActionID}
	if run.ActionID == decided.ActionID {
		entry = decided
		entry.Prices = clonePriceAudit(decided.Prices)
	}
	entry.Basis = PolicyLedgerBasisReported
	return &entry
}

// commitReportedRun records run as the turn that ran: the arm's warmth and
// last call, the held arm and the served model. A run with no arm counts the
// turn and clears the held arm, writing no warmth. For a client's report on
// an arm it returns that arm's warmth before it.
func commitReportedRun(next *EpisodeState, run ReportedRun, inputTokens int, report Report) (*ReplacedWarmth, error) {
	next.ServedModel = run.Model
	if run.Arm < 0 {
		next.PreviousArm = nil
		next.PolicyBoundary = nil
		next.Pending = nil
		if next.TurnIndex < math.MaxUint64 {
			next.TurnIndex++
		}
		return nil, nil
	}
	if run.Arm >= len(next.Warmth) {
		return nil, errors.New("reported arm is out of range")
	}
	before := next.Warmth[run.Arm]
	if err := next.Commit(run.Arm, inputTokens, report.FinishedAt.UTC()); err != nil {
		return nil, err
	}
	epoch := ""
	if next.Policy != nil {
		epoch = next.Policy.ContextEpoch()
	}
	next.Warmth[run.Arm].Call = reportedCall(run, report, epoch)
	if report.Source != ReportSourceClient {
		return nil, nil
	}
	return &ReplacedWarmth{Warmth: warmthToWire(before)}, nil
}

// reportedCall is the arm's last-call record for a reported run, in the
// context epoch its turn was recorded in ("" when unknown), as a served
// turn's call is.
func reportedCall(run ReportedRun, report Report, epoch string) *cachecontext.CallCache {
	call := run.Call
	call.ActionID = run.ActionID
	call.CompletedAt = report.FinishedAt.UTC()
	call.ContextEpoch = epoch
	return &call
}

// contextEpoch is the context epoch of the turn the report recorded, ""
// when it recorded none.
func (last *LastReport) contextEpoch() string {
	if last.Message < 0 {
		return ""
	}
	return strconv.Itoa(last.Epoch)
}

func newLastReport(routeID string, report Report, turnIndex uint64, message, arm int, replaced *ReplacedWarmth) *LastReport {
	return &LastReport{
		RouteID: routeID, ReportID: report.ReportID, Source: report.Source,
		TurnIndex: turnIndex, Message: message, Arm: arm, Replaced: replaced,
	}
}

func recordedReport(reason, routeID string, next *EpisodeState, run ReportedRun) ReportOutcome {
	return ReportOutcome{Recorded: true, Reason: reason, RouteID: routeID, TurnIndex: next.TurnIndex, Attributed: run.Arm >= 0}
}
