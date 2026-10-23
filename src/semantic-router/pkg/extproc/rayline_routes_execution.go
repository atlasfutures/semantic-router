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
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// routing.execution (pathfinder #5874, proposed ADR 0154): the execution
// state of a caller that runs turns itself, carried on one Decisions lookup.
// It is request-scoped: it shapes only this lookup's decide request and
// answer, and never enters the router's session state (no warmth, call
// cache, ledger or episode write). Ids, counts, durations and enums only.

const (
	routesExecutionVersion = "arc-execution-context.v1"
	// routesExecutionMaxAttribution and routesExecutionMaxWorkers are the
	// schema's bounds; the second is the service's MaxCacheContextActions.
	routesExecutionMaxAttribution = 4096
	routesExecutionMaxWorkers     = raylinearc.MaxCacheContextActions
	routesExecutionMaxCount       = 100_000_000
	routesExecutionMaxGapS        = 604_800
	routesExecutionMaxTTLS        = 86_400
	// routesUserTurnCacheMultiwarm is the selection.user_turn_cache mode the
	// lookup asks for when it carries admitted cache evidence.
	routesUserTurnCacheMultiwarm = "repartition_multiwarm"
)

const (
	provenanceProviderReported = "provider_reported"
	provenanceLocalMeasured    = "local_measured"
	provenanceCallerEstimated  = "caller_estimated"
)

var (
	routesExecutionOpaqueID = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,128}$`)
	routesExecutionUnits    = regexp.MustCompile(`^[A-Za-z0-9._:@+/-]{1,160}$`)
	routesExecutionTTLLabel = regexp.MustCompile(`^[A-Za-z0-9 ._:()-]{1,96}$`)
	routesExecutionSHA256   = regexp.MustCompile(`^[a-f0-9]{64}$`)
)

// routesExecution is a validated routing.execution member. Every nullable
// member is nil when the caller sent null; nothing is defaulted.
type routesExecution struct {
	contextEpoch string
	turnID       string
	turnAttempt  int
	// expect is the caller's pinned identity: package, thinking controls,
	// offer (nil: not pinned).
	expectPackage, expectThinkingControls, expectOffer *string
	attribution                                        []routesExecutionAttribution
	hold                                               *routesExecutionHold
	workers                                            []routesExecutionWorker
}

type routesExecutionAttribution struct {
	message  int
	actionID string
	armID    *string
	worker   string
	routeID  *string
	result   *string
}

type routesExecutionHold struct {
	scope    string // model or action
	actionID string
}

type routesExecutionTTL struct {
	seconds float64
	source  string
	basis   string
}

type routesExecutionResidency struct {
	state      string
	provenance string
}

type routesExecutionUsage struct {
	promptTokens, cacheReadTokens, cacheWriteTokens *int
	units, provenance                               string
}

type routesExecutionPrefix struct {
	tokens            int
	overlapTokens     *int
	units, provenance string
}

type routesExecutionWorker struct {
	actionID, worker, placement string
	contextEpoch                *string
	gapS                        *float64
	ttl                         *routesExecutionTTL
	residency                   *routesExecutionResidency
	usage                       *routesExecutionUsage
	currentPrefix               *routesExecutionPrefix
}

// routesExecutionInvalid is the schema refusal: a wrong shape, an unknown
// or missing key, a type or bound violated, a bad id, a repeated action in
// workers or a repeated attribution index.
func routesExecutionInvalid(path, problem string) *routesContractError {
	return routesInvalid("execution_invalid", "routing.execution"+path+" "+problem+".")
}

// routesExecutionObject reads an object with exactly the given keys: every
// key is required (unknown is null, never omitted) and no other is allowed.
func routesExecutionObject(raw json.RawMessage, path string, keys ...string) (map[string]json.RawMessage, *routesContractError) {
	var members map[string]json.RawMessage
	if json.Unmarshal(raw, &members) != nil || members == nil || bytes.HasPrefix(bytes.TrimSpace(raw), []byte("null")) {
		return nil, routesExecutionInvalid(path, "must be an object")
	}
	allowed := make(map[string]bool, len(keys))
	for _, key := range keys {
		allowed[key] = true
		if _, ok := members[key]; !ok {
			return nil, routesExecutionInvalid(path, "is missing "+key+" (an unknown value is null, never omitted)")
		}
	}
	var extra []string
	for key := range members {
		if !allowed[key] {
			extra = append(extra, key)
		}
	}
	if len(extra) > 0 {
		sort.Strings(extra)
		return nil, routesExecutionInvalid(path, "has unsupported member(s): "+strings.Join(extra, ", "))
	}
	return members, nil
}

func routesExecutionIsNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

func routesExecutionString(raw json.RawMessage, pattern *regexp.Regexp, path string) (string, *routesContractError) {
	var value string
	if json.Unmarshal(raw, &value) != nil || !pattern.MatchString(value) {
		return "", routesExecutionInvalid(path, "must be a string matching "+pattern.String())
	}
	return value, nil
}

func routesExecutionNullableString(raw json.RawMessage, pattern *regexp.Regexp, path string) (*string, *routesContractError) {
	if routesExecutionIsNull(raw) {
		return nil, nil
	}
	value, failure := routesExecutionString(raw, pattern, path)
	if failure != nil {
		return nil, routesExecutionInvalid(path, "must be null or a string matching "+pattern.String())
	}
	return &value, nil
}

func routesExecutionEnum(raw json.RawMessage, path string, values ...string) (string, *routesContractError) {
	var value string
	if json.Unmarshal(raw, &value) == nil {
		for _, allowed := range values {
			if value == allowed {
				return value, nil
			}
		}
	}
	return "", routesExecutionInvalid(path, "must be one of "+strings.Join(values, ", "))
}

// routesExecutionInteger reads a JSON integer in [minimum, maximum]; a float
// spelling of an integer (2.0) is not one.
func routesExecutionInteger(raw json.RawMessage, path string, minimum, maximum int) (int, *routesContractError) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value interface{}
	if decoder.Decode(&value) == nil {
		if number, ok := value.(json.Number); ok {
			if parsed, err := strconv.ParseInt(number.String(), 10, 64); err == nil && parsed >= int64(minimum) && parsed <= int64(maximum) {
				return int(parsed), nil
			}
		}
	}
	return 0, routesExecutionInvalid(path, fmt.Sprintf("must be an integer in [%d, %d]", minimum, maximum))
}

func routesExecutionCount(raw json.RawMessage, path string) (*int, *routesContractError) {
	if routesExecutionIsNull(raw) {
		return nil, nil
	}
	value, failure := routesExecutionInteger(raw, path, 0, routesExecutionMaxCount)
	if failure != nil {
		return nil, routesExecutionInvalid(path, fmt.Sprintf("must be null or an integer in [0, %d]", routesExecutionMaxCount))
	}
	return &value, nil
}

// readRoutesExecution validates routing.execution against the schema
// (execution_invalid), then its counts (execution_counts_inconsistent), then
// its attribution against the body's messages (execution_attribution_invalid),
// in the contract's refusal order. The checks that need the serving package
// (catalog, expect, hold) are the selector's.
func readRoutesExecution(raw json.RawMessage, envelope map[string]json.RawMessage) (*routesExecution, *routesContractError) {
	members, failure := routesExecutionObject(raw, "", "version", "context_epoch", "turn", "expect", "attribution", "hold", "workers")
	if failure != nil {
		return nil, failure
	}
	execution := &routesExecution{}
	if _, versionFailure := routesExecutionEnum(members["version"], ".version", routesExecutionVersion); versionFailure != nil {
		return nil, versionFailure
	}
	if execution.contextEpoch, failure = routesExecutionString(members["context_epoch"], routesExecutionOpaqueID, ".context_epoch"); failure != nil {
		return nil, failure
	}
	turn, failure := routesExecutionObject(members["turn"], ".turn", "id", "attempt")
	if failure != nil {
		return nil, failure
	}
	if execution.turnID, failure = routesExecutionString(turn["id"], routesExecutionOpaqueID, ".turn.id"); failure != nil {
		return nil, failure
	}
	if execution.turnAttempt, failure = routesExecutionInteger(turn["attempt"], ".turn.attempt", 1, 64); failure != nil {
		return nil, failure
	}
	expect, failure := routesExecutionObject(members["expect"], ".expect", "package_sha256", "thinking_controls_sha256", "offer_sha256")
	if failure != nil {
		return nil, failure
	}
	for _, pin := range []struct {
		key    string
		target **string
	}{
		{"package_sha256", &execution.expectPackage},
		{"thinking_controls_sha256", &execution.expectThinkingControls},
		{"offer_sha256", &execution.expectOffer},
	} {
		if *pin.target, failure = routesExecutionNullableString(expect[pin.key], routesExecutionSHA256, ".expect."+pin.key); failure != nil {
			return nil, failure
		}
	}
	if execution.attribution, failure = readRoutesExecutionAttribution(members["attribution"]); failure != nil {
		return nil, failure
	}
	if execution.hold, failure = readRoutesExecutionHold(members["hold"]); failure != nil {
		return nil, failure
	}
	if execution.workers, failure = readRoutesExecutionWorkers(members["workers"]); failure != nil {
		return nil, failure
	}
	if failure := checkRoutesExecutionCounts(execution.workers); failure != nil {
		return nil, failure
	}
	if failure := checkRoutesExecutionAttribution(execution.attribution, envelope); failure != nil {
		return nil, failure
	}
	return execution, nil
}

func readRoutesExecutionAttribution(raw json.RawMessage) ([]routesExecutionAttribution, *routesContractError) {
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil || items == nil {
		return nil, routesExecutionInvalid(".attribution", "must be an array")
	}
	if len(items) > routesExecutionMaxAttribution {
		return nil, routesExecutionInvalid(".attribution", fmt.Sprintf("holds at most %d entries", routesExecutionMaxAttribution))
	}
	attribution := make([]routesExecutionAttribution, 0, len(items))
	seen := map[int]bool{}
	for index, item := range items {
		path := fmt.Sprintf(".attribution[%d]", index)
		members, failure := routesExecutionObject(item, path, "message", "action_id", "arm_id", "worker", "route_id", "result")
		if failure != nil {
			return nil, failure
		}
		entry := routesExecutionAttribution{}
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
		if entry.armID, failure = routesExecutionNullableString(members["arm_id"], routesExecutionSHA256, path+".arm_id"); failure != nil {
			return nil, failure
		}
		if entry.worker, failure = routesExecutionString(members["worker"], routesExecutionOpaqueID, path+".worker"); failure != nil {
			return nil, failure
		}
		if entry.routeID, failure = routesExecutionNullableString(members["route_id"], routesExecutionOpaqueID, path+".route_id"); failure != nil {
			return nil, failure
		}
		if !routesExecutionIsNull(members["result"]) {
			result, failure := routesExecutionEnum(members["result"], path+".result", "top", "swift", "sage", "custom")
			if failure != nil {
				return nil, routesExecutionInvalid(path+".result", "must be null or one of top, swift, sage, custom")
			}
			entry.result = &result
		}
		attribution = append(attribution, entry)
	}
	return attribution, nil
}

func readRoutesExecutionHold(raw json.RawMessage) (*routesExecutionHold, *routesContractError) {
	if routesExecutionIsNull(raw) {
		return nil, nil
	}
	members, failure := routesExecutionObject(raw, ".hold", "scope", "action_id")
	if failure != nil {
		return nil, routesExecutionInvalid(".hold", "must be null or an object {scope, action_id}")
	}
	hold := &routesExecutionHold{}
	if hold.scope, failure = routesExecutionEnum(members["scope"], ".hold.scope", "model", "action"); failure != nil {
		return nil, failure
	}
	if hold.actionID, failure = routesExecutionString(members["action_id"], routesExecutionSHA256, ".hold.action_id"); failure != nil {
		return nil, failure
	}
	return hold, nil
}

func readRoutesExecutionWorkers(raw json.RawMessage) ([]routesExecutionWorker, *routesContractError) {
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil || items == nil {
		return nil, routesExecutionInvalid(".workers", "must be an array")
	}
	if len(items) > routesExecutionMaxWorkers {
		return nil, routesExecutionInvalid(".workers", fmt.Sprintf("holds at most %d entries", routesExecutionMaxWorkers))
	}
	workers := make([]routesExecutionWorker, 0, len(items))
	seen := map[string]bool{}
	for index, item := range items {
		worker, failure := readRoutesExecutionWorker(item, fmt.Sprintf(".workers[%d]", index))
		if failure != nil {
			return nil, failure
		}
		if seen[worker.actionID] {
			return nil, routesExecutionInvalid(fmt.Sprintf(".workers[%d].action_id", index), "repeats an action (one entry per action)")
		}
		seen[worker.actionID] = true
		workers = append(workers, worker)
	}
	return workers, nil
}

func readRoutesExecutionWorker(raw json.RawMessage, path string) (routesExecutionWorker, *routesContractError) {
	worker := routesExecutionWorker{}
	members, failure := routesExecutionObject(raw, path,
		"action_id", "worker", "placement", "context_epoch", "gap_s", "ttl", "residency", "usage", "current_prefix")
	if failure != nil {
		return worker, failure
	}
	if worker.actionID, failure = routesExecutionString(members["action_id"], routesExecutionSHA256, path+".action_id"); failure != nil {
		return worker, failure
	}
	if worker.worker, failure = routesExecutionString(members["worker"], routesExecutionOpaqueID, path+".worker"); failure != nil {
		return worker, failure
	}
	if worker.placement, failure = routesExecutionEnum(members["placement"], path+".placement", "cloud", "local"); failure != nil {
		return worker, failure
	}
	if worker.contextEpoch, failure = routesExecutionNullableString(members["context_epoch"], routesExecutionOpaqueID, path+".context_epoch"); failure != nil {
		return worker, failure
	}
	if !routesExecutionIsNull(members["gap_s"]) {
		gap, finite := routesFiniteNumber(members["gap_s"])
		if !finite || gap < 0 || gap > routesExecutionMaxGapS {
			return worker, routesExecutionInvalid(path+".gap_s", fmt.Sprintf("must be null or a number in [0, %d]", routesExecutionMaxGapS))
		}
		worker.gapS = &gap
	}
	provenance := func(raw json.RawMessage, at string) (string, *routesContractError) {
		return routesExecutionEnum(raw, at, provenanceProviderReported, provenanceLocalMeasured, provenanceCallerEstimated)
	}
	if !routesExecutionIsNull(members["ttl"]) {
		ttl, failure := routesExecutionObject(members["ttl"], path+".ttl", "seconds", "source", "basis")
		if failure != nil {
			return worker, failure
		}
		parsed := &routesExecutionTTL{}
		seconds, finite := routesFiniteNumber(ttl["seconds"])
		if !finite || seconds <= 0 || seconds > routesExecutionMaxTTLS {
			return worker, routesExecutionInvalid(path+".ttl.seconds", fmt.Sprintf("must be a number in (0, %d]", routesExecutionMaxTTLS))
		}
		parsed.seconds = seconds
		if parsed.source, failure = routesExecutionString(ttl["source"], routesExecutionTTLLabel, path+".ttl.source"); failure != nil {
			return worker, failure
		}
		if parsed.basis, failure = routesExecutionEnum(ttl["basis"], path+".ttl.basis", "wire_declared", "provider_documented", "caller_estimated"); failure != nil {
			return worker, failure
		}
		worker.ttl = parsed
	}
	if !routesExecutionIsNull(members["residency"]) {
		residency, failure := routesExecutionObject(members["residency"], path+".residency", "state", "provenance")
		if failure != nil {
			return worker, failure
		}
		parsed := &routesExecutionResidency{}
		if parsed.state, failure = routesExecutionEnum(residency["state"], path+".residency.state", "resident", "evicted"); failure != nil {
			return worker, failure
		}
		if parsed.provenance, failure = provenance(residency["provenance"], path+".residency.provenance"); failure != nil {
			return worker, failure
		}
		worker.residency = parsed
	}
	if !routesExecutionIsNull(members["usage"]) {
		usage, failure := routesExecutionObject(members["usage"], path+".usage",
			"prompt_tokens", "cache_read_tokens", "cache_write_tokens", "units", "provenance")
		if failure != nil {
			return worker, failure
		}
		parsed := &routesExecutionUsage{}
		for _, count := range []struct {
			key    string
			target **int
		}{
			{"prompt_tokens", &parsed.promptTokens},
			{"cache_read_tokens", &parsed.cacheReadTokens},
			{"cache_write_tokens", &parsed.cacheWriteTokens},
		} {
			if *count.target, failure = routesExecutionCount(usage[count.key], path+".usage."+count.key); failure != nil {
				return worker, failure
			}
		}
		if parsed.units, failure = routesExecutionString(usage["units"], routesExecutionUnits, path+".usage.units"); failure != nil {
			return worker, failure
		}
		if parsed.provenance, failure = provenance(usage["provenance"], path+".usage.provenance"); failure != nil {
			return worker, failure
		}
		worker.usage = parsed
	}
	if !routesExecutionIsNull(members["current_prefix"]) {
		prefix, failure := routesExecutionObject(members["current_prefix"], path+".current_prefix",
			"tokens", "overlap_tokens", "units", "provenance")
		if failure != nil {
			return worker, failure
		}
		parsed := &routesExecutionPrefix{}
		if parsed.tokens, failure = routesExecutionInteger(prefix["tokens"], path+".current_prefix.tokens", 1, routesExecutionMaxCount); failure != nil {
			return worker, failure
		}
		if parsed.overlapTokens, failure = routesExecutionCount(prefix["overlap_tokens"], path+".current_prefix.overlap_tokens"); failure != nil {
			return worker, failure
		}
		if parsed.units, failure = routesExecutionString(prefix["units"], routesExecutionUnits, path+".current_prefix.units"); failure != nil {
			return worker, failure
		}
		if parsed.provenance, failure = provenance(prefix["provenance"], path+".current_prefix.provenance"); failure != nil {
			return worker, failure
		}
		worker.currentPrefix = parsed
	}
	return worker, nil
}

// checkRoutesExecutionCounts refuses counts that contradict each other: a
// known cache leg, or both, above prompt_tokens; overlap above the prefix.
func checkRoutesExecutionCounts(workers []routesExecutionWorker) *routesContractError {
	inconsistent := func(index int, problem string) *routesContractError {
		return routesInvalid("execution_counts_inconsistent",
			fmt.Sprintf("routing.execution.workers[%d]: %s.", index, problem))
	}
	for index, worker := range workers {
		if usage := worker.usage; usage != nil && usage.promptTokens != nil {
			prompt, cached := *usage.promptTokens, 0
			for _, leg := range []*int{usage.cacheReadTokens, usage.cacheWriteTokens} {
				if leg != nil {
					if *leg > prompt {
						return inconsistent(index, "a cache leg exceeds usage.prompt_tokens (which includes them)")
					}
					cached += *leg
				}
			}
			if cached > prompt {
				return inconsistent(index, "the cache legs together exceed usage.prompt_tokens")
			}
		}
		if prefix := worker.currentPrefix; prefix != nil && prefix.overlapTokens != nil && *prefix.overlapTokens > prefix.tokens {
			return inconsistent(index, "current_prefix.overlap_tokens exceeds current_prefix.tokens")
		}
	}
	return nil
}

// checkRoutesExecutionAttribution refuses an attribution index outside the
// body's messages (input for Responses) or naming a non-assistant one.
func checkRoutesExecutionAttribution(
	attribution []routesExecutionAttribution,
	envelope map[string]json.RawMessage,
) *routesContractError {
	if len(attribution) == 0 {
		return nil
	}
	raw, responses := envelope["input"]
	if !responses {
		raw = envelope["messages"]
	}
	var items []json.RawMessage
	_ = json.Unmarshal(raw, &items)
	for _, entry := range attribution {
		if entry.message >= len(items) {
			return routesInvalid("execution_attribution_invalid", fmt.Sprintf(
				"routing.execution.attribution names message %d; the request has %d.", entry.message, len(items)))
		}
		if !routesExecutionAssistantItem(items[entry.message], responses) {
			return routesInvalid("execution_attribution_invalid", fmt.Sprintf(
				"routing.execution.attribution names message %d, which is not an assistant message.", entry.message))
		}
	}
	return nil
}

// routesExecutionAssistantItem reports whether one messages entry (or
// Responses input item) is the assistant's output.
func routesExecutionAssistantItem(raw json.RawMessage, responses bool) bool {
	var item struct {
		Role string `json:"role"`
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &item) != nil {
		return false
	}
	if item.Role == "assistant" {
		return true
	}
	if !responses {
		return false
	}
	switch item.Type {
	case "function_call", "custom_tool_call", "reasoning", "web_search_call", "file_search_call",
		"computer_call", "code_interpreter_call", "local_shell_call", "image_generation_call":
		return true
	}
	return false
}

// --- admission ---------------------------------------------------------------

// routesExecutionNotAdmitted names a value passed on as unknown, and why.
type routesExecutionNotAdmitted struct {
	Field  string `json:"field"`
	Reason string `json:"reason"`
}

// admitRoutesExecutionWorker is one worker's ActionCache after admission
// (pathfinder #5874, "Admission"): caller evidence moves a price only when
// its provenance may, and is never relabelled. A value not admitted is sent
// null and named, with its reason, in the echo. admitted reports whether
// any cache evidence (TTL, residency, prompt tokens, prefix) was admitted.
func admitRoutesExecutionWorker(worker routesExecutionWorker) (raylinearc.PolicyActionCache, []routesExecutionNotAdmitted, bool) {
	entry := raylinearc.PolicyActionCache{ActionID: worker.actionID, GapS: worker.gapS, ContextEpoch: worker.contextEpoch}
	notAdmitted := []routesExecutionNotAdmitted{}
	admitted := false
	refuse := func(field, reason string) {
		notAdmitted = append(notAdmitted, routesExecutionNotAdmitted{Field: field, Reason: reason})
	}
	if ttl := worker.ttl; ttl != nil {
		if ttl.basis == "wire_declared" || ttl.basis == "provider_documented" {
			seconds, source := ttl.seconds, ttl.source
			entry.TTLS, entry.TTLSource = &seconds, &source
			admitted = true
		} else {
			refuse("ttl", "estimated_not_admitted")
		}
	}
	if residency := worker.residency; residency != nil {
		switch {
		case residency.provenance == provenanceProviderReported,
			residency.provenance == provenanceLocalMeasured && worker.placement == "local":
			state := residency.state
			entry.Residency = &state
			admitted = true
		case residency.provenance == provenanceLocalMeasured:
			refuse("residency", "local_measured_not_local")
		default:
			refuse("residency", "estimated_not_admitted")
		}
	}
	usageAdmitted := false
	if usage := worker.usage; usage != nil {
		if usage.provenance == provenanceCallerEstimated {
			refuse("usage", "estimated_not_admitted")
		} else {
			usageAdmitted = true
			entry.PromptTokens, entry.CacheReadTokens, entry.CacheWriteTokens = usage.promptTokens, usage.cacheReadTokens, usage.cacheWriteTokens
			admitted = admitted || usage.promptTokens != nil
		}
	}
	if prefix := worker.currentPrefix; prefix != nil {
		switch {
		case prefix.provenance == provenanceCallerEstimated:
			refuse("current_prefix", "estimated_not_admitted")
		case usageAdmitted && worker.usage.units != prefix.units:
			// Two counts are compared only in the same units: the prefix
			// stays unknown, so the worker is never credited warmth it
			// cannot be shown to have.
			refuse("current_prefix", "units_mismatch")
		default:
			tokens := prefix.tokens
			entry.CurrentPrefixTokens, entry.OverlapTokens = &tokens, prefix.overlapTokens
			admitted = true
		}
	}
	return entry, notAdmitted, admitted
}

// routesExecutionControls is the lookup's execution context as decide-request
// members: the caller's epoch and attribution, its hold, and the admitted
// cache evidence (with user_turn_cache repartition_multiwarm when any is).
func routesExecutionControls(execution *routesExecution) (*selection.RaylineARCRouteExecution, map[string][]routesExecutionNotAdmitted) {
	controls := &selection.RaylineARCRouteExecution{
		ContextEpoch:           execution.contextEpoch,
		Attribution:            make([]raylinearc.PolicyAttribution, 0, len(execution.attribution)),
		ExpectPackage:          execution.expectPackage,
		ExpectThinkingControls: execution.expectThinkingControls,
		ExpectOffer:            execution.expectOffer,
	}
	for _, entry := range execution.attribution {
		controls.Attribution = append(controls.Attribution, raylinearc.PolicyAttribution{
			Message: entry.message, ActionID: entry.actionID, ArmID: entry.armID,
		})
		controls.ActionIDs = append(controls.ActionIDs, entry.actionID)
	}
	sort.Slice(controls.Attribution, func(i, j int) bool { return controls.Attribution[i].Message < controls.Attribution[j].Message })
	if hold := execution.hold; hold != nil {
		controls.HoldScope, controls.HoldActionID = hold.scope, hold.actionID
		controls.ActionIDs = append(controls.ActionIDs, hold.actionID)
	}
	notAdmitted := make(map[string][]routesExecutionNotAdmitted, len(execution.workers))
	anyAdmitted := false
	if len(execution.workers) > 0 {
		controls.CacheContext = &raylinearc.PolicyCacheContext{Actions: make([]raylinearc.PolicyActionCache, 0, len(execution.workers))}
	}
	for _, worker := range execution.workers {
		entry, refused, admitted := admitRoutesExecutionWorker(worker)
		controls.CacheContext.Actions = append(controls.CacheContext.Actions, entry)
		controls.ActionIDs = append(controls.ActionIDs, worker.actionID)
		notAdmitted[worker.actionID] = refused
		anyAdmitted = anyAdmitted || admitted
	}
	if anyAdmitted {
		controls.UserTurnCache = routesUserTurnCacheMultiwarm
	}
	return controls, notAdmitted
}

// routesOfferSHA256 is the offer's identity, sha256(JCS({schema:
// "arc-offer.v1", package_sha256, action_ids})) over the actions offered to
// this request, sorted.
func routesOfferSHA256(packageSHA256 string, actionIDs []string) string {
	sorted := append([]string(nil), actionIDs...)
	sort.Slice(sorted, func(i, j int) bool { return jcsLess(sorted[i], sorted[j]) })
	var out strings.Builder
	out.WriteString(`{"action_ids":[`)
	for index, id := range sorted {
		if index > 0 {
			out.WriteByte(',')
		}
		out.WriteString(jcsString(id))
	}
	out.WriteString(`],"package_sha256":`)
	out.WriteString(jcsString(packageSHA256))
	out.WriteString(`,"schema":"arc-offer.v1"}`)
	digest := sha256.Sum256([]byte(out.String()))
	return hex.EncodeToString(digest[:])
}

// --- the echo ----------------------------------------------------------------

// raylineRoutesExecutionEcho is routing.execution in the routing echo (gate 6
// of the sidecar plan): whether it was applied, the turn, the pinned
// identity, the effective user_turn_cache and boundary, and per worker how
// the decision read its evidence. Values come from the service's
// decision.cache_trace, never recomputed; null when it did not say.
type raylineRoutesExecutionEcho struct {
	Status        string                             `json:"status"`
	Turn          raylineRoutesExecutionTurn         `json:"turn"`
	Pinned        map[string]string                  `json:"pinned"`
	UserTurnCache *string                            `json:"user_turn_cache"`
	Boundary      *string                            `json:"boundary"`
	Workers       []raylineRoutesExecutionWorkerEcho `json:"workers"`
}

type raylineRoutesExecutionTurn struct {
	ID      string `json:"id"`
	Attempt int    `json:"attempt"`
}

type raylineRoutesExecutionWorkerEcho struct {
	ActionID         string                       `json:"action_id"`
	Worker           string                       `json:"worker"`
	Read             *bool                        `json:"read"`
	Reason           *string                      `json:"reason"`
	Basis            *string                      `json:"basis"`
	PWarm            *float64                     `json:"p_warm"`
	PrefixTokensUsed *int                         `json:"prefix_tokens_used"`
	PrefixBasis      *string                      `json:"prefix_basis"`
	NotAdmitted      []routesExecutionNotAdmitted `json:"not_admitted"`
}

func raylineRoutesExecutionEchoOf(routing *raylineRoutesRouting, decision routerruntime.RouteDecision) *raylineRoutesExecutionEcho {
	execution := routing.execution
	echo := &raylineRoutesExecutionEcho{
		Status:  "unsupported",
		Turn:    raylineRoutesExecutionTurn{ID: execution.turnID, Attempt: execution.turnAttempt},
		Pinned:  map[string]string{},
		Workers: make([]raylineRoutesExecutionWorkerEcho, 0, len(execution.workers)),
	}
	if decision.ExecutionApplied {
		echo.Status = "applied"
	}
	// A mismatch never reaches an answer (it is refused): a pinned member
	// matched.
	for key, pin := range map[string]*string{
		"package_sha256": execution.expectPackage, "thinking_controls_sha256": execution.expectThinkingControls,
		"offer_sha256": execution.expectOffer,
	} {
		echo.Pinned[key] = "unpinned"
		if pin != nil {
			echo.Pinned[key] = "matched"
		}
	}
	traced := map[string]raylinearc.PolicyCacheTraceAction{}
	if trace := decision.CacheTrace; trace != nil {
		echo.UserTurnCache, echo.Boundary = trace.UserTurnCache, trace.Boundary
		for _, action := range trace.Actions {
			traced[action.ActionID] = action
		}
	}
	for _, worker := range execution.workers {
		entry := raylineRoutesExecutionWorkerEcho{
			ActionID: worker.actionID, Worker: worker.worker,
			NotAdmitted: routing.executionNotAdmitted[worker.actionID],
		}
		if entry.NotAdmitted == nil {
			entry.NotAdmitted = []routesExecutionNotAdmitted{}
		}
		if action, ok := traced[worker.actionID]; ok {
			entry.Read, entry.Reason, entry.Basis = action.Read, action.Reason, action.Basis
			entry.PWarm, entry.PrefixTokensUsed, entry.PrefixBasis = action.PWarm, action.PrefixTokensUsed, action.PrefixBasis
		}
		echo.Workers = append(echo.Workers, entry)
	}
	return echo
}
