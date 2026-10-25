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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Hybrid routing v2 on /v1/routes (pathfinder #5904, arc_execution_v2):
// routing.policy and the v2 routing.execution envelope. VSR validates both,
// passes them to the policy service as decide-request members, and copies
// the service's receipt and the echoes back, never recomputing them. A
// lookup that sends neither is decided and answered exactly as before.

const routesExecutionVersionV2 = "arc-execution-context.v2"

var (
	routesPolicyPresets = []string{"arc-policy.package-default.v1", "arc-policy.hybrid-default.v1", "arc-policy.prefer-local.v1"}
	// routesSelectionPresets are the presets with a selection rule, the only
	// ones overrides.selection applies to.
	routesSelectionPresets     = []string{"arc-policy.prefer-local.v1"}
	routesScheduleClasses      = []string{"cache_boundaries", "user_messages", "all_turns"}
	routesExecutorEventKinds   = []string{"task_start", "user_message", "tool_result", "compaction", "subagent_start", "sidecall"}
	routesExecutorEventBases   = []string{"explicit_client", "marker_inferred", "unknown"}
	routesCapacityReasons      = []string{"context_capacity", "runtime_unavailable"}
	routesCapacitySources      = []string{"configured", "runtime_observed"}
	routesExecutionReconsults  = []string{"capacity", "veto"}
	routesExecutionPlacements  = []string{"cloud", "local"}
	routesExecutionProvenances = []string{provenanceProviderReported, provenanceLocalMeasured, provenanceCallerEstimated}
)

// --- routing.policy ----------------------------------------------------------

func routesPolicyInvalid(path, problem string) *routesContractError {
	return routesInvalid("policy_invalid", "routing.policy"+path+" "+problem+".")
}

// routesPolicyObject reads an object of optional members, none unknown and
// none null (a policy member is omitted, never null).
func routesPolicyObject(raw json.RawMessage, path string, allowed ...string) (map[string]json.RawMessage, *routesContractError) {
	var members map[string]json.RawMessage
	if json.Unmarshal(raw, &members) != nil || members == nil || routesExecutionIsNull(raw) {
		return nil, routesPolicyInvalid(path, "must be an object")
	}
	for key, value := range members {
		if !contains(allowed, key) {
			return nil, routesPolicyInvalid(path, "has an unsupported member: "+key)
		}
		if routesExecutionIsNull(value) {
			return nil, routesPolicyInvalid(path+"."+key, "is omitted, never null")
		}
	}
	return members, nil
}

func contains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

// readRoutesPolicy validates routing.policy (arc_execution_v2.parse_policy):
// a preset id and an optional schedule override per dimension, each a set
// of switch-point classes. Whether the cell admits the preset and the
// package the schedule is the service's to say.
func readRoutesPolicy(raw json.RawMessage) (*raylinearc.PolicyMember, *routesContractError) {
	members, failure := routesPolicyObject(raw, "", "preset", "overrides")
	if failure != nil {
		return nil, failure
	}
	presetRaw, ok := members["preset"]
	if !ok {
		return nil, routesPolicyInvalid("", "is missing preset")
	}
	var preset string
	if json.Unmarshal(presetRaw, &preset) != nil || !contains(routesPolicyPresets, preset) {
		return nil, routesPolicyInvalid(".preset", "must be one of "+strings.Join(routesPolicyPresets, ", "))
	}
	policy := &raylinearc.PolicyMember{Preset: preset}
	overridesRaw, ok := members["overrides"]
	if !ok {
		return policy, nil
	}
	overrides, failure := routesPolicyObject(overridesRaw, ".overrides", "schedule", "selection")
	if failure != nil {
		return nil, failure
	}
	policy.Overrides = &raylinearc.PolicyMemberOverride{}
	if selectionRaw, hasSelection := overrides["selection"]; hasSelection {
		if !contains(routesSelectionPresets, preset) {
			return nil, routesPolicyInvalid(".overrides.selection", "applies only to a preset with a selection rule ("+
				strings.Join(routesSelectionPresets, ", ")+")")
		}
		if policy.Overrides.Selection, failure = readRoutesPolicySelection(selectionRaw); failure != nil {
			return nil, failure
		}
	}
	scheduleRaw, ok := overrides["schedule"]
	if !ok {
		return policy, nil
	}
	schedule, failure := routesPolicyObject(scheduleRaw, ".overrides.schedule", "model", "thinking")
	if failure != nil {
		return nil, failure
	}
	policy.Overrides.Schedule = &raylinearc.PolicyMemberSchedule{}
	for _, dimension := range []struct {
		key    string
		target *[]string
	}{{"model", &policy.Overrides.Schedule.Model}, {"thinking", &policy.Overrides.Schedule.Thinking}} {
		classesRaw, ok := schedule[dimension.key]
		if !ok {
			continue
		}
		path := ".overrides.schedule." + dimension.key
		var classes []string
		if json.Unmarshal(classesRaw, &classes) != nil || classes == nil {
			return nil, routesPolicyInvalid(path, "must be an array of classes")
		}
		seen := map[string]bool{}
		for _, class := range classes {
			if !contains(routesScheduleClasses, class) {
				return nil, routesPolicyInvalid(path, "names an unknown class (one of "+strings.Join(routesScheduleClasses, ", ")+")")
			}
			if seen[class] {
				return nil, routesPolicyInvalid(path, "names a class once")
			}
			seen[class] = true
		}
		*dimension.target = classes
	}
	return policy, nil
}

// readRoutesPolicySelection reads prefer-local's thresholds: min_p_pass in
// [0, 1], max_rest_tokens an integer > 0, each optional, none null.
func readRoutesPolicySelection(raw json.RawMessage) (*raylinearc.PolicyMemberSelection, *routesContractError) {
	members, failure := routesPolicyObject(raw, ".overrides.selection", "min_p_pass", "max_rest_tokens")
	if failure != nil {
		return nil, failure
	}
	selection := &raylinearc.PolicyMemberSelection{}
	if value, ok := members["min_p_pass"]; ok {
		number, finite := routesFiniteNumber(value)
		if !finite || number < 0 || number > 1 {
			return nil, routesPolicyInvalid(".overrides.selection.min_p_pass", "must be a number in [0, 1]")
		}
		selection.MinPPass = &number
	}
	if value, ok := members["max_rest_tokens"]; ok {
		tokens, failure := routesExecutionInteger(value, ".overrides.selection.max_rest_tokens", 1, math.MaxInt32)
		if failure != nil {
			return nil, routesPolicyInvalid(".overrides.selection.max_rest_tokens", "must be an integer > 0")
		}
		selection.MaxRestTokens = &tokens
	}
	return selection, nil
}

// --- the v2 envelope -----------------------------------------------------------

type routesExecutionV2History struct {
	message  int
	actionID string
}

type routesExecutionV2LastServed struct {
	actionID string
	worker   string
	message  *int
}

type routesExecutionV2Capacity struct {
	fits                *bool
	reason              *string
	required, available *int
	availableSource     *string
	units               *string
	provenance          string
}

type routesExecutionV2Target struct {
	evidence *routesExecutionWorker // the cache evidence as a v1 worker; nil when cache is null
	actionID string
	worker   string
	place    string
	capacity *routesExecutionV2Capacity
}

// routesExecutionV2 is a validated v2 routing.execution.
type routesExecutionV2 struct {
	turnID         string
	turnAttempt    int
	turnEpoch      string
	turnRetryOf    *string
	turnReconsult  *string
	turnRaw        json.RawMessage
	event          raylinearc.PolicyExecutorEvent
	lastServed     *routesExecutionV2LastServed
	history        []routesExecutionV2History
	holdActionID   *string
	expectPackage  *string
	expectControls *string
	expectOffer    *string
	targets        []routesExecutionV2Target
}

func routesExecutionNullableIndex(raw json.RawMessage, path string) (*int, *routesContractError) {
	if routesExecutionIsNull(raw) {
		return nil, nil
	}
	value, failure := routesExecutionInteger(raw, path, 0, math.MaxInt32)
	if failure != nil {
		return nil, routesExecutionInvalid(path, "must be null or an integer >= 0")
	}
	return &value, nil
}

func routesExecutionNullableEnum(raw json.RawMessage, path string, values []string) (*string, *routesContractError) {
	if routesExecutionIsNull(raw) {
		return nil, nil
	}
	value, failure := routesExecutionEnum(raw, path, values...)
	if failure != nil {
		return nil, routesExecutionInvalid(path, "must be null or one of "+strings.Join(values, ", "))
	}
	return &value, nil
}

// readRoutesExecutionV2 validates the v2 envelope (arc_execution_v2.
// parse_execution_v2): shape and bounds, the counts, then history,
// last_served and a compaction's indices against the body's own messages.
func readRoutesExecutionV2(raw json.RawMessage, envelope map[string]json.RawMessage) (*routesExecutionV2, *routesContractError) {
	members, failure := routesExecutionObject(raw, "", "version", "turn", "event", "last_served", "history", "hold", "expect", "targets")
	if failure != nil {
		return nil, failure
	}
	execution := &routesExecutionV2{}
	if _, versionFailure := routesExecutionEnum(members["version"], ".version", routesExecutionVersionV2); versionFailure != nil {
		return nil, versionFailure
	}
	turn, failure := routesExecutionObject(members["turn"], ".turn", "id", "attempt", "epoch", "retry_of", "reconsult")
	if failure != nil {
		return nil, failure
	}
	if execution.turnID, failure = routesExecutionString(turn["id"], routesExecutionOpaqueID, ".turn.id"); failure != nil {
		return nil, failure
	}
	if execution.turnAttempt, failure = routesExecutionInteger(turn["attempt"], ".turn.attempt", 1, 64); failure != nil {
		return nil, failure
	}
	if execution.turnEpoch, failure = routesExecutionString(turn["epoch"], routesExecutionOpaqueID, ".turn.epoch"); failure != nil {
		return nil, failure
	}
	if execution.turnRetryOf, failure = routesExecutionNullableString(turn["retry_of"], routesExecutionOpaqueID, ".turn.retry_of"); failure != nil {
		return nil, failure
	}
	if execution.turnReconsult, failure = routesExecutionNullableEnum(turn["reconsult"], ".turn.reconsult", routesExecutionReconsults); failure != nil {
		return nil, failure
	}
	if execution.turnReconsult != nil && execution.turnRetryOf == nil {
		return nil, routesExecutionInvalid(".turn", "re-consults name the route they retry (retry_of)")
	}
	if failure = readRoutesExecutionV2Event(members["event"], execution); failure != nil {
		return nil, failure
	}
	if !routesExecutionIsNull(members["last_served"]) {
		var served map[string]json.RawMessage
		served, failure = routesExecutionObject(members["last_served"], ".last_served", "action_id", "worker", "message")
		if failure != nil {
			return nil, failure
		}
		entry := &routesExecutionV2LastServed{}
		if entry.actionID, failure = routesExecutionString(served["action_id"], routesExecutionSHA256, ".last_served.action_id"); failure != nil {
			return nil, failure
		}
		if entry.worker, failure = routesExecutionString(served["worker"], routesExecutionOpaqueID, ".last_served.worker"); failure != nil {
			return nil, failure
		}
		if entry.message, failure = routesExecutionNullableIndex(served["message"], ".last_served.message"); failure != nil {
			return nil, failure
		}
		execution.lastServed = entry
	}
	if execution.history, failure = readRoutesExecutionV2History(members["history"]); failure != nil {
		return nil, failure
	}
	if !routesExecutionIsNull(members["hold"]) {
		var hold map[string]json.RawMessage
		hold, failure = routesExecutionObject(members["hold"], ".hold", "scope", "action_id")
		if failure != nil {
			return nil, failure
		}
		if _, scopeFailure := routesExecutionEnum(hold["scope"], ".hold.scope", "action"); scopeFailure != nil {
			return nil, scopeFailure
		}
		var actionID string
		actionID, failure = routesExecutionString(hold["action_id"], routesExecutionSHA256, ".hold.action_id")
		if failure != nil {
			return nil, failure
		}
		execution.holdActionID = &actionID
	}
	if !routesExecutionIsNull(members["expect"]) {
		var expect map[string]json.RawMessage
		expect, failure = routesExecutionObject(members["expect"], ".expect", "package_sha256", "thinking_controls_sha256", "offer_sha256")
		if failure != nil {
			return nil, failure
		}
		for _, pin := range []struct {
			key    string
			target **string
		}{{"package_sha256", &execution.expectPackage}, {"thinking_controls_sha256", &execution.expectControls}, {"offer_sha256", &execution.expectOffer}} {
			if *pin.target, failure = routesExecutionNullableString(expect[pin.key], routesExecutionSHA256, ".expect."+pin.key); failure != nil {
				return nil, failure
			}
		}
	}
	if execution.targets, failure = readRoutesExecutionV2Targets(members["targets"]); failure != nil {
		return nil, failure
	}
	execution.turnRaw = members["turn"]
	evidence := make([]routesExecutionWorker, 0, len(execution.targets))
	for _, target := range execution.targets {
		if target.evidence != nil {
			evidence = append(evidence, *target.evidence)
		}
	}
	if failure = checkRoutesExecutionCounts(evidence); failure != nil {
		return nil, failure
	}
	return execution, checkRoutesExecutionV2Indices(execution, envelope)
}

func readRoutesExecutionV2Event(raw json.RawMessage, execution *routesExecutionV2) *routesContractError {
	event, failure := routesExecutionObject(raw, ".event", "kind", "basis", "compaction")
	if failure != nil {
		return failure
	}
	if execution.event.Kind, failure = routesExecutionNullableEnum(event["kind"], ".event.kind", routesExecutorEventKinds); failure != nil {
		return failure
	}
	if execution.event.Basis, failure = routesExecutionEnum(event["basis"], ".event.basis", routesExecutorEventBases...); failure != nil {
		return failure
	}
	if (execution.event.Basis == "unknown") != (execution.event.Kind == nil) {
		return routesExecutionInvalid(".event", "has no kind exactly when its basis is unknown")
	}
	if !routesExecutionIsNull(event["compaction"]) {
		compaction, failure := routesExecutionObject(event["compaction"], ".event.compaction", "summary_message", "retained_from", "new")
		if failure != nil {
			return failure
		}
		parsed := &raylinearc.PolicyExecutorCompaction{}
		if parsed.SummaryMessage, failure = routesExecutionNullableIndex(compaction["summary_message"], ".event.compaction.summary_message"); failure != nil {
			return failure
		}
		if parsed.RetainedFrom, failure = routesExecutionNullableIndex(compaction["retained_from"], ".event.compaction.retained_from"); failure != nil {
			return failure
		}
		trimmed := bytes.TrimSpace(compaction["new"])
		if !bytes.Equal(trimmed, []byte("true")) && !bytes.Equal(trimmed, []byte("false")) {
			return routesExecutionInvalid(".event.compaction.new", "must be true or false")
		}
		parsed.New = bytes.Equal(trimmed, []byte("true"))
		execution.event.Compaction = parsed
	}
	if (execution.event.Kind != nil && *execution.event.Kind == "compaction") != (execution.event.Compaction != nil) {
		return routesExecutionInvalid(".event", "carries a compaction block exactly when its kind is compaction")
	}
	return nil
}

func readRoutesExecutionV2History(raw json.RawMessage) ([]routesExecutionV2History, *routesContractError) {
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil || items == nil {
		return nil, routesExecutionInvalid(".history", "must be an array")
	}
	if len(items) > routesExecutionMaxAttribution {
		return nil, routesExecutionInvalid(".history", fmt.Sprintf("holds at most %d entries", routesExecutionMaxAttribution))
	}
	history := make([]routesExecutionV2History, 0, len(items))
	seen := map[int]bool{}
	for index, item := range items {
		path := fmt.Sprintf(".history[%d]", index)
		members, failure := routesExecutionObject(item, path, "message", "action_id")
		if failure != nil {
			return nil, failure
		}
		entry := routesExecutionV2History{}
		if entry.message, failure = routesExecutionInteger(members["message"], path+".message", 0, math.MaxInt32); failure != nil {
			return nil, failure
		}
		if seen[entry.message] {
			return nil, routesExecutionInvalid(path+".message", "repeats an index (each message at most once)")
		}
		seen[entry.message] = true
		if entry.actionID, failure = routesExecutionString(members["action_id"], routesExecutionSHA256, path+".action_id"); failure != nil {
			return nil, failure
		}
		history = append(history, entry)
	}
	return history, nil
}

func readRoutesExecutionV2Targets(raw json.RawMessage) ([]routesExecutionV2Target, *routesContractError) {
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil || items == nil {
		return nil, routesExecutionInvalid(".targets", "must be an array")
	}
	if len(items) > routesExecutionMaxWorkers {
		return nil, routesExecutionInvalid(".targets", fmt.Sprintf("holds at most %d entries", routesExecutionMaxWorkers))
	}
	targets := make([]routesExecutionV2Target, 0, len(items))
	seen := map[string]bool{}
	for index, item := range items {
		path := fmt.Sprintf(".targets[%d]", index)
		members, failure := routesExecutionObject(item, path, "action_id", "worker", "placement", "cache", "capacity")
		if failure != nil {
			return nil, failure
		}
		target := routesExecutionV2Target{}
		if target.actionID, failure = routesExecutionString(members["action_id"], routesExecutionSHA256, path+".action_id"); failure != nil {
			return nil, failure
		}
		if seen[target.actionID] {
			return nil, routesExecutionInvalid(path+".action_id", "repeats an action (one target per action)")
		}
		seen[target.actionID] = true
		if target.worker, failure = routesExecutionString(members["worker"], routesExecutionOpaqueID, path+".worker"); failure != nil {
			return nil, failure
		}
		if target.place, failure = routesExecutionEnum(members["placement"], path+".placement", routesExecutionPlacements...); failure != nil {
			return nil, failure
		}
		if !routesExecutionIsNull(members["cache"]) {
			var cache map[string]json.RawMessage
			cache, failure = routesExecutionObject(members["cache"], path+".cache",
				"context_epoch", "gap_s", "ttl", "residency", "usage", "current_prefix")
			if failure != nil {
				return nil, failure
			}
			worker := &routesExecutionWorker{actionID: target.actionID, worker: target.worker, placement: target.place}
			if failure = readRoutesExecutionEvidence(worker, cache, path+".cache"); failure != nil {
				return nil, failure
			}
			target.evidence = worker
		}
		if !routesExecutionIsNull(members["capacity"]) {
			if target.capacity, failure = readRoutesExecutionV2Capacity(members["capacity"], path+".capacity"); failure != nil {
				return nil, failure
			}
		}
		targets = append(targets, target)
	}
	return targets, nil
}

func readRoutesExecutionV2Capacity(raw json.RawMessage, path string) (*routesExecutionV2Capacity, *routesContractError) {
	members, failure := routesExecutionObject(raw, path, "fits", "reason", "required", "available", "available_source", "units", "provenance")
	if failure != nil {
		return nil, failure
	}
	capacity := &routesExecutionV2Capacity{}
	if !routesExecutionIsNull(members["fits"]) {
		trimmed := bytes.TrimSpace(members["fits"])
		if !bytes.Equal(trimmed, []byte("true")) && !bytes.Equal(trimmed, []byte("false")) {
			return nil, routesExecutionInvalid(path+".fits", "must be null, true or false")
		}
		fits := bytes.Equal(trimmed, []byte("true"))
		capacity.fits = &fits
	}
	if capacity.reason, failure = routesExecutionNullableEnum(members["reason"], path+".reason", routesCapacityReasons); failure != nil {
		return nil, failure
	}
	if capacity.required, failure = routesExecutionCount(members["required"], path+".required"); failure != nil {
		return nil, failure
	}
	if capacity.available, failure = routesExecutionCount(members["available"], path+".available"); failure != nil {
		return nil, failure
	}
	if capacity.availableSource, failure = routesExecutionNullableEnum(members["available_source"], path+".available_source", routesCapacitySources); failure != nil {
		return nil, failure
	}
	if capacity.units, failure = routesExecutionNullableString(members["units"], routesExecutionUnits, path+".units"); failure != nil {
		return nil, failure
	}
	if capacity.provenance, failure = routesExecutionEnum(members["provenance"], path+".provenance", routesExecutionProvenances...); failure != nil {
		return nil, failure
	}
	if (capacity.fits != nil && !*capacity.fits) != (capacity.reason != nil) {
		return nil, routesExecutionInvalid(path, "gives a reason exactly when the target does not fit")
	}
	return capacity, nil
}

// checkRoutesExecutionV2Indices refuses a history or last_served index that
// is not an assistant message of this body, and a compaction index past it.
func checkRoutesExecutionV2Indices(execution *routesExecutionV2, envelope map[string]json.RawMessage) *routesContractError {
	named := make([]routesExecutionAttribution, 0, len(execution.history)+1)
	for _, entry := range execution.history {
		named = append(named, routesExecutionAttribution{message: entry.message})
	}
	if execution.lastServed != nil && execution.lastServed.message != nil {
		named = append(named, routesExecutionAttribution{message: *execution.lastServed.message})
	}
	if failure := checkRoutesExecutionAttribution(named, envelope); failure != nil {
		return failure
	}
	if compaction := execution.event.Compaction; compaction != nil {
		count := routesExecutionMessageCount(envelope)
		for _, index := range []*int{compaction.SummaryMessage, compaction.RetainedFrom} {
			if index != nil && *index >= count {
				return routesExecutionInvalid(".event.compaction", "indexes past the request")
			}
		}
	}
	return nil
}

func routesExecutionMessageCount(envelope map[string]json.RawMessage) int {
	_, hasInput := envelope["input"]
	_, hasMessages := envelope["messages"]
	raw := envelope["messages"]
	if hasInput && !hasMessages {
		raw = envelope["input"]
	}
	var items []json.RawMessage
	_ = json.Unmarshal(raw, &items)
	return len(items)
}

// routesTurnKey is the stable key a turn's decision and draw derive from,
// across attempts and lookups (arc_execution_v2.turn_key).
func routesTurnKey(turnID string) string {
	digest := sha256.Sum256([]byte("arc-turn-key.v1\n" + turnID))
	return hex.EncodeToString(digest[:])
}

// routesExecutionV2Controls maps the envelope onto decide-request members
// (arc_execution_v2.decide_inputs_v2): v1's admission for each target's
// cache evidence; capacity fits true admitted only from local_measured or
// provider_reported (else unknown, named), fits false from any provenance.
func routesExecutionV2Controls(execution *routesExecutionV2) (*selection.RaylineARCRouteExecutionV2, []routesExecutionNotAdmitted) {
	controls := &selection.RaylineARCRouteExecutionV2{
		ContextEpoch:           execution.turnEpoch,
		Attribution:            make([]raylinearc.PolicyAttribution, 0, len(execution.history)),
		HoldActionID:           execution.holdActionID,
		Event:                  execution.event,
		TurnKey:                routesTurnKey(execution.turnID),
		Reconsult:              execution.turnReconsult,
		ExpectPackage:          execution.expectPackage,
		ExpectThinkingControls: execution.expectControls,
		ExpectOffer:            execution.expectOffer,
	}
	for _, entry := range execution.history {
		controls.Attribution = append(controls.Attribution, raylinearc.PolicyAttribution{Message: entry.message, ActionID: entry.actionID})
		controls.ActionIDs = append(controls.ActionIDs, entry.actionID)
	}
	sort.Slice(controls.Attribution, func(i, j int) bool { return controls.Attribution[i].Message < controls.Attribution[j].Message })
	refused := []routesExecutionNotAdmitted{}
	for _, target := range execution.targets {
		controls.ActionIDs = append(controls.ActionIDs, target.actionID)
		if target.evidence != nil {
			entry, notAdmitted := admitRoutesExecutionWorker(*target.evidence)
			if controls.CacheContext == nil {
				controls.CacheContext = &cachecontext.PolicyCacheContext{}
			}
			controls.CacheContext.Actions = append(controls.CacheContext.Actions, entry)
			refused = append(refused, notAdmitted...)
		}
		// Every target is one capacity entry: the service bounds the offer
		// to the targets, and a target with no fact is unknown (fits and
		// reason null), carrying its placement (pathfinder #5915).
		if target.capacity == nil {
			controls.Capacity = append(controls.Capacity, raylinearc.PolicyActionCapacity{
				ActionID: target.actionID, Placement: target.place,
			})
		}
		if fact := target.capacity; fact != nil {
			fits := fact.fits
			if fits != nil && *fits && fact.provenance == provenanceCallerEstimated {
				fits = nil
				refused = append(refused, routesExecutionNotAdmitted{ActionID: target.actionID, Field: "capacity", Reason: "estimated_not_admitted"})
			}
			controls.Capacity = append(controls.Capacity, raylinearc.PolicyActionCapacity{
				ActionID: target.actionID, Placement: target.place, Fits: fits, Reason: fact.reason,
			})
			if target.capacity.fits != nil && !*target.capacity.fits {
				controls.Infeasible = append(controls.Infeasible, target.actionID)
			}
		}
	}
	if served := execution.lastServed; served != nil {
		actionID := served.actionID
		controls.LastServedActionID = &actionID
		controls.LastServedMessage = served.message
		controls.ActionIDs = append(controls.ActionIDs, actionID)
	}
	if execution.holdActionID != nil {
		controls.ActionIDs = append(controls.ActionIDs, *execution.holdActionID)
	}
	return controls, refused
}

// --- the receipt and the echoes ------------------------------------------------

// raylineRoutesReceipt is one public result's receipt
// (arc_execution_v2.result_receipt): the decide's decision.receipt with the
// result's own action and control, copied, never recomputed.
type raylineRoutesReceipt struct {
	ActionID        string                             `json:"action_id"`
	ControlID       *string                            `json:"control_id"`
	Policy          *string                            `json:"policy"`
	DecidedBy       raylinearc.PolicyReceiptDecidedBy  `json:"decided_by"`
	Eligible        raylinearc.PolicyReceiptEligible   `json:"eligible"`
	HeldFrom        *string                            `json:"held_from"`
	Excluded        []raylinearc.PolicyReceiptExcluded `json:"excluded"`
	CapacityUnknown []string                           `json:"capacity_unknown"`
	Basis           *string                            `json:"basis"`
	// Selection is a prefer-local decision's comparison, copied verbatim;
	// omitted otherwise.
	Selection json.RawMessage `json:"selection,omitempty"`
}

func raylineRoutesReceiptOf(receipt *raylinearc.PolicyDecisionReceipt, actionID, controlID string) *raylineRoutesReceipt {
	if receipt == nil || actionID == "" {
		return nil
	}
	rendered := &raylineRoutesReceipt{
		ActionID: actionID, Policy: receipt.Policy, DecidedBy: receipt.DecidedBy, Eligible: receipt.Eligible,
		HeldFrom: receipt.HeldFrom, Excluded: receipt.Excluded, CapacityUnknown: receipt.CapacityUnknown, Basis: receipt.Basis,
		Selection: receipt.Selection,
	}
	if controlID != "" {
		rendered.ControlID = &controlID
	}
	if rendered.Excluded == nil {
		rendered.Excluded = []raylinearc.PolicyReceiptExcluded{}
	}
	if rendered.CapacityUnknown == nil {
		rendered.CapacityUnknown = []string{}
	}
	return rendered
}

// raylineRoutesExecutionEchoV2 is the v2 routing.execution echo
// (arc_execution_v2.ExecutionEchoV2): applied exactly when the decide
// carried a receipt; the turn as sent; the pins (null when unsupported).
// There are no per-worker rows in v2: each result's receipt says how it was
// decided.
type raylineRoutesExecutionEchoV2 struct {
	Status          string            `json:"status"`
	Version         string            `json:"version"`
	Turn            json.RawMessage   `json:"turn"`
	Pinned          map[string]string `json:"pinned"`
	EpisodeIdentity *string           `json:"episode_identity"`
}

func raylineRoutesExecutionEchoV2Of(execution *routesExecutionV2, decision routerruntime.RouteDecision) *raylineRoutesExecutionEchoV2 {
	turn := json.RawMessage(routesCompactJSON(execution.turnRaw))
	echo := &raylineRoutesExecutionEchoV2{Status: "unsupported", Version: routesExecutionVersionV2, Turn: turn}
	if decision.Receipt == nil {
		return echo
	}
	echo.Status = "applied"
	episode := routesExecutionEpisodePerLookup
	echo.EpisodeIdentity = &episode
	echo.Pinned = map[string]string{}
	for key, pin := range map[string]*string{
		"package_sha256": execution.expectPackage, "thinking_controls_sha256": execution.expectControls,
		"offer_sha256": execution.expectOffer,
	} {
		echo.Pinned[key] = "unpinned"
		if pin != nil {
			echo.Pinned[key] = "matched"
		}
	}
	return echo
}

// raylineRoutesPolicyEcho is the routing.policy echo
// (arc_execution_v2.PolicyEcho): applied when the decide carried a receipt.
type raylineRoutesPolicyEcho struct {
	Status string `json:"status"`
	Preset string `json:"preset"`
}

func raylineRoutesPolicyEchoOf(policy *raylinearc.PolicyMember, decision routerruntime.RouteDecision) *raylineRoutesPolicyEcho {
	status := "unsupported"
	if decision.Receipt != nil {
		status = "applied"
	}
	return &raylineRoutesPolicyEcho{Status: status, Preset: policy.Preset}
}

func routesCompactJSON(raw json.RawMessage) []byte {
	var out bytes.Buffer
	if json.Compact(&out, raw) != nil {
		return raw
	}
	return out.Bytes()
}
