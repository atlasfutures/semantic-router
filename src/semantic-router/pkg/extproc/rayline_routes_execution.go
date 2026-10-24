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
	return worker, readRoutesExecutionEvidence(&worker, members, path)
}

// readRoutesExecutionEvidence reads a worker's cache evidence (context_epoch,
// gap_s, ttl, residency, usage, current_prefix) from members already checked
// for their key set: v1's workers[] entry, or a v2 target's cache.
func readRoutesExecutionEvidence(worker *routesExecutionWorker, members map[string]json.RawMessage, path string) *routesContractError {
	var failure *routesContractError
	if worker.contextEpoch, failure = routesExecutionNullableString(members["context_epoch"], routesExecutionOpaqueID, path+".context_epoch"); failure != nil {
		return failure
	}
	if !routesExecutionIsNull(members["gap_s"]) {
		gap, finite := routesFiniteNumber(members["gap_s"])
		if !finite || gap < 0 || gap > routesExecutionMaxGapS {
			return routesExecutionInvalid(path+".gap_s", fmt.Sprintf("must be null or a number in [0, %d]", routesExecutionMaxGapS))
		}
		worker.gapS = &gap
	}
	provenance := func(raw json.RawMessage, at string) (string, *routesContractError) {
		return routesExecutionEnum(raw, at, provenanceProviderReported, provenanceLocalMeasured, provenanceCallerEstimated)
	}
	if !routesExecutionIsNull(members["ttl"]) {
		ttl, failure := routesExecutionObject(members["ttl"], path+".ttl", "seconds", "source", "basis")
		if failure != nil {
			return failure
		}
		parsed := &routesExecutionTTL{}
		seconds, finite := routesFiniteNumber(ttl["seconds"])
		if !finite || seconds <= 0 || seconds > routesExecutionMaxTTLS {
			return routesExecutionInvalid(path+".ttl.seconds", fmt.Sprintf("must be a number in (0, %d]", routesExecutionMaxTTLS))
		}
		parsed.seconds = seconds
		if parsed.source, failure = routesExecutionString(ttl["source"], routesExecutionTTLLabel, path+".ttl.source"); failure != nil {
			return failure
		}
		if parsed.basis, failure = routesExecutionEnum(ttl["basis"], path+".ttl.basis", "wire_declared", "provider_documented", "caller_estimated"); failure != nil {
			return failure
		}
		worker.ttl = parsed
	}
	if !routesExecutionIsNull(members["residency"]) {
		residency, failure := routesExecutionObject(members["residency"], path+".residency", "state", "provenance")
		if failure != nil {
			return failure
		}
		parsed := &routesExecutionResidency{}
		if parsed.state, failure = routesExecutionEnum(residency["state"], path+".residency.state", "resident", "evicted"); failure != nil {
			return failure
		}
		if parsed.provenance, failure = provenance(residency["provenance"], path+".residency.provenance"); failure != nil {
			return failure
		}
		worker.residency = parsed
	}
	if !routesExecutionIsNull(members["usage"]) {
		usage, failure := routesExecutionObject(members["usage"], path+".usage",
			"prompt_tokens", "cache_read_tokens", "cache_write_tokens", "units", "provenance")
		if failure != nil {
			return failure
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
				return failure
			}
		}
		if parsed.units, failure = routesExecutionString(usage["units"], routesExecutionUnits, path+".usage.units"); failure != nil {
			return failure
		}
		if parsed.provenance, failure = provenance(usage["provenance"], path+".usage.provenance"); failure != nil {
			return failure
		}
		worker.usage = parsed
	}
	if !routesExecutionIsNull(members["current_prefix"]) {
		prefix, failure := routesExecutionObject(members["current_prefix"], path+".current_prefix",
			"tokens", "overlap_tokens", "units", "provenance")
		if failure != nil {
			return failure
		}
		parsed := &routesExecutionPrefix{}
		if parsed.tokens, failure = routesExecutionInteger(prefix["tokens"], path+".current_prefix.tokens", 1, routesExecutionMaxCount); failure != nil {
			return failure
		}
		if parsed.overlapTokens, failure = routesExecutionCount(prefix["overlap_tokens"], path+".current_prefix.overlap_tokens"); failure != nil {
			return failure
		}
		if parsed.units, failure = routesExecutionString(prefix["units"], routesExecutionUnits, path+".current_prefix.units"); failure != nil {
			return failure
		}
		if parsed.provenance, failure = provenance(prefix["provenance"], path+".current_prefix.provenance"); failure != nil {
			return failure
		}
		worker.currentPrefix = parsed
	}
	return nil
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
	_, hasInput := envelope["input"]
	_, hasMessages := envelope["messages"]
	responses := hasInput && !hasMessages
	raw := envelope["messages"]
	if responses {
		raw = envelope["input"]
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
	if !responses {
		return item.Role == "assistant"
	}
	// Responses: an assistant-output item (a message with role assistant, a
	// function_call or a reasoning item).
	switch item.Type {
	case "", "message":
		return item.Role == "assistant"
	case "function_call", "reasoning":
		return true
	}
	return false
}

// --- admission ---------------------------------------------------------------

// routesExecutionNotAdmitted names a value passed on as unknown, and why
// (pathfinder #5875, arc_execution_context.NotAdmitted).
type routesExecutionNotAdmitted struct {
	ActionID string `json:"-"`
	Field    string `json:"field"`
	Reason   string `json:"reason"`
}

// admitRoutesExecutionWorker is one worker's ActionCache after admission
// (arc_execution_context.decide_inputs): a value not admitted is sent
// unknown and named, never relabelled; missing evidence never makes a switch
// cheaper.
//
//   - ttl: wire_declared and provider_documented only;
//   - residency: local_measured on a local worker only;
//   - usage and current_prefix: any provenance but caller_estimated. A usage
//     in other units than the current prefix is not comparable with it and is
//     left out (units_mismatch); the current prefix stays.
func admitRoutesExecutionWorker(worker routesExecutionWorker) (raylinearc.PolicyActionCache, []routesExecutionNotAdmitted) {
	refused := []routesExecutionNotAdmitted{}
	refuse := func(field, reason string) {
		refused = append(refused, routesExecutionNotAdmitted{ActionID: worker.actionID, Field: field, Reason: reason})
	}
	ttl := worker.ttl
	if ttl != nil && ttl.basis == "caller_estimated" {
		refuse("ttl", "estimated_not_admitted")
		ttl = nil
	}
	residency := worker.residency
	switch {
	case residency != nil && residency.provenance != provenanceLocalMeasured:
		refuse("residency", "residency_requires_local_measured")
		residency = nil
	case residency != nil && worker.placement != "local":
		refuse("residency", "residency_requires_local_placement")
		residency = nil
	}
	usage, prefix := worker.usage, worker.currentPrefix
	if usage != nil && usage.provenance == provenanceCallerEstimated {
		refuse("usage", "estimated_not_admitted")
		usage = nil
	}
	if prefix != nil && prefix.provenance == provenanceCallerEstimated {
		refuse("current_prefix", "estimated_not_admitted")
		prefix = nil
	}
	if usage != nil && prefix != nil && usage.units != prefix.units {
		refuse("usage", "units_mismatch")
		usage = nil
	}
	entry := raylinearc.PolicyActionCache{ActionID: worker.actionID, GapS: worker.gapS, ContextEpoch: worker.contextEpoch}
	if ttl != nil {
		seconds, source := ttl.seconds, ttl.source
		entry.TTLS, entry.TTLSource = &seconds, &source
	}
	if usage != nil {
		entry.PromptTokens, entry.CacheReadTokens, entry.CacheWriteTokens = usage.promptTokens, usage.cacheReadTokens, usage.cacheWriteTokens
	}
	if prefix != nil {
		tokens := prefix.tokens
		entry.CurrentPrefixTokens, entry.OverlapTokens = &tokens, prefix.overlapTokens
	}
	if residency != nil {
		state := residency.state
		entry.Residency = &state
	}
	return entry, refused
}

// routesExecutionControls is the lookup's execution context as decide-request
// members: the caller's epoch and attribution, its hold, and the admitted
// cache evidence. The cache mode is the service's to decide.
func routesExecutionControls(execution *routesExecution) (*selection.RaylineARCRouteExecution, []routesExecutionNotAdmitted) {
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
	for _, worker := range execution.workers {
		controls.ActionIDs = append(controls.ActionIDs, worker.actionID)
	}
	if hold := execution.hold; hold != nil {
		controls.HoldScope, controls.HoldActionID = hold.scope, hold.actionID
		controls.ActionIDs = append(controls.ActionIDs, hold.actionID)
	}
	refused := []routesExecutionNotAdmitted{}
	if len(execution.workers) > 0 {
		controls.CacheContext = &raylinearc.PolicyCacheContext{Actions: make([]raylinearc.PolicyActionCache, 0, len(execution.workers))}
	}
	for _, worker := range execution.workers {
		entry, notAdmitted := admitRoutesExecutionWorker(worker)
		controls.CacheContext.Actions = append(controls.CacheContext.Actions, entry)
		refused = append(refused, notAdmitted...)
	}
	return controls, refused
}

// routesOfferSHA256 is the offer's identity, sha256(JCS({schema:
// "arc-offer.v1", package_sha256, action_ids})) over the sorted, distinct
// actions offered to the request.
func routesOfferSHA256(packageSHA256 string, actionIDs []string) string {
	distinct := map[string]bool{}
	sorted := make([]string, 0, len(actionIDs))
	for _, id := range actionIDs {
		if !distinct[id] {
			distinct[id] = true
			sorted = append(sorted, id)
		}
	}
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

const (
	routesUserTurnCacheMultiwarm = "repartition_multiwarm"
	// routesExecutionEpisodePerLookup: an execution lookup is decided under
	// a fresh random episode id (ephemeral), never one derived from the
	// caller's session alone.
	routesExecutionEpisodePerLookup = "per_lookup_random"
)

// raylineRoutesExecutionEcho is routing.execution as the serving path
// applied it (arc_execution_context.ExecutionEcho). unsupported: the cell
// reported no cache_trace, so every applied field is null and no worker row
// is stated.
type raylineRoutesExecutionEcho struct {
	Status string                     `json:"status"`
	Turn   raylineRoutesExecutionTurn `json:"turn"`
	// EpisodeIdentity is the basis of the episode the lookup was decided
	// under: per_lookup_random, a fresh random id per execution lookup
	// (tenant-safe, no encoder-session reuse). Account-scoped reuse waits on
	// a trusted gateway account header.
	// Null when unsupported.
	EpisodeIdentity     *string                            `json:"episode_identity"`
	Pinned              map[string]string                  `json:"pinned"`
	Boundary            *string                            `json:"boundary"`
	UserTurnCache       *string                            `json:"user_turn_cache"`
	UserTurnCacheSource *string                            `json:"user_turn_cache_source"`
	Multiwarm           *bool                              `json:"multiwarm"`
	Workers             []raylineRoutesExecutionWorkerEcho `json:"workers"`
}

type raylineRoutesExecutionTurn struct {
	ID      string `json:"id"`
	Attempt int    `json:"attempt"`
}

// raylineRoutesExecutionWorkerEcho is one stated worker as the decision
// priced it (arc_execution_context.WorkerEcho).
type raylineRoutesExecutionWorkerEcho struct {
	ActionID     string                       `json:"action_id"`
	Worker       string                       `json:"worker"`
	Held         *bool                        `json:"held"`
	Pricing      string                       `json:"pricing"`
	Read         bool                         `json:"read"`
	Reason       *string                      `json:"reason"`
	Basis        *string                      `json:"basis"`
	PWarm        *float64                     `json:"p_warm"`
	PrefixTokens *int                         `json:"prefix_tokens"`
	PrefixBasis  *string                      `json:"prefix_basis"`
	NotAdmitted  []routesExecutionNotAdmitted `json:"not_admitted"`
}

// raylineRoutesExecutionEchoOf mirrors arc_execution_context.execution_echo:
// admission's refusals joined to the decision's cache_trace, values copied,
// one row per stated worker in request order.
func raylineRoutesExecutionEchoOf(routing *raylineRoutesRouting, decision routerruntime.RouteDecision) *raylineRoutesExecutionEcho {
	execution := routing.execution
	echo := &raylineRoutesExecutionEcho{
		Status:  "unsupported",
		Turn:    raylineRoutesExecutionTurn{ID: execution.turnID, Attempt: execution.turnAttempt},
		Workers: []raylineRoutesExecutionWorkerEcho{},
	}
	trace := decision.CacheTrace
	if trace == nil || !decision.ExecutionApplied {
		return echo
	}
	echo.Status = "applied"
	episode := routesExecutionEpisodePerLookup
	echo.EpisodeIdentity = &episode
	echo.Pinned = map[string]string{}
	for key, pin := range map[string]*string{
		"package_sha256": execution.expectPackage, "thinking_controls_sha256": execution.expectThinkingControls,
		"offer_sha256": execution.expectOffer,
	} {
		// A mismatch never reaches an answer: it is refused.
		echo.Pinned[key] = "unpinned"
		if pin != nil {
			echo.Pinned[key] = "matched"
		}
	}
	echo.Boundary, echo.UserTurnCache, echo.UserTurnCacheSource = trace.Boundary, trace.UserTurnCache, trace.Source
	multiwarm := trace.UserTurnCache != nil && *trace.UserTurnCache == routesUserTurnCacheMultiwarm
	echo.Multiwarm = &multiwarm
	priced := trace.Boundary != nil && *trace.Boundary == "user_message"
	byAction := map[string]raylinearc.PolicyCacheTraceCandidate{}
	for _, candidate := range trace.Candidates {
		for _, actionID := range candidate.ActionIDs {
			byAction[actionID] = candidate
		}
	}
	reason := func(value string) *string { return &value }
	for _, worker := range execution.workers {
		row, found := byAction[worker.actionID]
		// A model this request did not offer: no switch to it was priced.
		found = found && priced && (row.Offered || row.Held)
		read := found && row.EvidenceActionID != nil && *row.EvidenceActionID == worker.actionID
		entry := raylineRoutesExecutionWorkerEcho{
			ActionID: worker.actionID, Worker: worker.worker, Read: read,
			NotAdmitted: []routesExecutionNotAdmitted{},
		}
		switch {
		case read:
		case !priced:
			entry.Reason = reason("boundary")
		case !found:
			entry.Reason = reason("not_in_offer")
		case row.EvidenceActionID != nil:
			entry.Reason = reason("superseded")
		default:
			entry.Reason = reason("package_mode")
		}
		switch {
		case !found:
			entry.Pricing = "not_priced"
		case read && row.Basis != nil && strings.HasPrefix(*row.Basis, "observed_"):
			entry.Pricing = "observed"
		case row.Held:
			entry.Pricing = "default_held_warm"
		default:
			entry.Pricing = "default_rival_cold"
		}
		if found {
			held := row.Held
			entry.Held, entry.Basis, entry.PWarm, entry.PrefixTokens, entry.PrefixBasis = &held, row.Basis, row.PWarm, row.PrefixTokens, row.PrefixBasis
		}
		for _, refused := range routing.executionNotAdmitted {
			if refused.ActionID == worker.actionID {
				entry.NotAdmitted = append(entry.NotAdmitted, refused)
			}
		}
		echo.Workers = append(echo.Workers, entry)
	}
	return echo
}
