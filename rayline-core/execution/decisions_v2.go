package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
)

// DecisionsExecutionContextV2 selects the Go core <-> policy-service envelope of
// the hybrid v2 spec (pathfinder docs/plans/2026-10-24-decisions-execution-context-v2.md):
// routing.policy plus routing.execution "arc-execution-context.v2". The core
// reports facts; the service selects. The core never reselects a model.
const DecisionsExecutionContextV2 = "execution_context_v2"

const executionContextV2 = "arc-execution-context.v2"

// maxTargetsV2 is the contract's targets[] bound (pathfinder #5904 schema).
const maxTargetsV2 = 16

// policyPresets are the presets the policy service serves (pathfinder #5904;
// arc-policy.prefer-local.v1 since #5913).
var policyPresets = map[string]bool{
	"arc-policy.hybrid-default.v1":  true,
	"arc-policy.prefer-local.v1":    true,
	"arc-policy.package-default.v1": true,
}

var scheduleClasses = map[string]bool{"cache_boundaries": true, "user_messages": true, "all_turns": true}

type policyOverrides struct {
	Schedule *struct {
		Model    []string `json:"model,omitempty"`
		Thinking []string `json:"thinking,omitempty"`
	} `json:"schedule,omitempty"`
	Selection *struct {
		MinPPass *float64 `json:"min_p_pass"`
	} `json:"selection,omitempty"`
}

type routingPolicy struct {
	Preset    string           `json:"preset"`
	Overrides *policyOverrides `json:"overrides,omitempty"`
}

// validateRoutingPolicy admits only the published shape: a known preset and
// optional schedule and prefer-local threshold overrides. Package
// admission of a class is the service's refusal, not a local guess.
func validateRoutingPolicy(raw json.RawMessage) (routingPolicy, error) {
	var policy routingPolicy
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&policy); err != nil {
		return policy, fmt.Errorf("routing.policy must contain only preset and overrides.{schedule,selection}: %v", err)
	}
	if !policyPresets[policy.Preset] {
		return policy, fmt.Errorf("routing.policy.preset %q is not a published preset", policy.Preset)
	}
	if overrides := policy.Overrides; overrides != nil {
		if schedule := overrides.Schedule; schedule != nil {
			for name, classes := range map[string][]string{"model": schedule.Model, "thinking": schedule.Thinking} {
				if classes == nil {
					continue
				}
				seen := map[string]bool{}
				if len(classes) == 0 {
					return policy, fmt.Errorf("routing.policy schedule.%s must name at least one class", name)
				}
				for _, class := range classes {
					if !scheduleClasses[class] || seen[class] {
						return policy, fmt.Errorf("routing.policy schedule.%s has unknown or duplicate class %q", name, class)
					}
					seen[class] = true
				}
			}
		}
	}
	if overrides := policy.Overrides; overrides != nil && overrides.Selection != nil {
		threshold := overrides.Selection.MinPPass
		if threshold == nil || !(*threshold > 0 && *threshold < 1) {
			return policy, fmt.Errorf("routing.policy selection.min_p_pass must be in (0, 1)")
		}
		if policy.Preset != "arc-policy.prefer-local.v1" {
			return policy, fmt.Errorf("routing.policy selection.min_p_pass applies only to arc-policy.prefer-local.v1")
		}
	}
	return policy, nil
}

type turnV2 struct {
	ID        string  `json:"id"`
	Attempt   int     `json:"attempt"`
	Epoch     string  `json:"epoch"`
	RetryOf   *string `json:"retry_of"`
	Reconsult *string `json:"reconsult"`
}

type lastServedV2 struct {
	ActionID string `json:"action_id"`
	Worker   string `json:"worker"`
	Message  *int   `json:"message"`
}

type historyV2 struct {
	Message  int    `json:"message"`
	ActionID string `json:"action_id"`
}

type expectV2 struct {
	PackageSHA256          *string `json:"package_sha256"`
	ThinkingControlsSHA256 *string `json:"thinking_controls_sha256"`
	OfferSHA256            *string `json:"offer_sha256"`
}

// cacheV2 is v1's worker evidence unchanged, minus the identity members that
// moved up to the target.
type cacheV2 struct {
	ContextEpoch  *string         `json:"context_epoch"`
	Gap           *float64        `json:"gap_s"`
	TTL           *externalTTL    `json:"ttl"`
	Residency     *Residency      `json:"residency"`
	Usage         *externalUsage  `json:"usage"`
	CurrentPrefix *externalPrefix `json:"current_prefix"`
}

type targetV2 struct {
	ActionID  string        `json:"action_id"`
	Worker    string        `json:"worker"`
	Placement string        `json:"placement"`
	Cache     *cacheV2      `json:"cache"`
	Capacity  *CapacityFact `json:"capacity"`
}

type contextV2 struct {
	Version    string         `json:"version"`
	Turn       turnV2         `json:"turn"`
	Event      ExecutionEvent `json:"event"`
	LastServed *lastServedV2  `json:"last_served"`
	History    []historyV2    `json:"history"`
	Hold       *holdV2        `json:"hold"`
	// Expect is null for a package-less (derived_actions) profile: it pins nothing.
	Expect  *expectV2  `json:"expect"`
	Targets []targetV2 `json:"targets"`
}

// holdV2 is a mandatory action-scope constraint. The core has no current
// source of one; the field exists so a future constraint is explicit.
type holdV2 struct {
	Scope    string `json:"scope"`
	ActionID string `json:"action_id"`
}

func (provider *HTTPDecisions) projectV2(input DecisionInput) contextV2 {
	external := contextV2{Version: executionContextV2, Event: input.Event, History: []historyV2{}, Targets: []targetV2{}}
	external.Turn = turnV2{ID: fingerprint([]byte(input.SessionID + "\x00" + input.BranchID + "\x00" + input.RequestID)), Attempt: 1, Epoch: input.State.ContextEpoch}
	if reconsult := input.Reconsult; reconsult != nil {
		kind, retryOf := reconsult.Kind, reconsult.RetryOf
		external.Turn.Attempt, external.Turn.Reconsult, external.Turn.RetryOf = reconsult.Attempt, &kind, &retryOf
	}
	// A package-less profile pins the registry (and any offer), not a package.
	if provider.config.PackageSHA256 != "" || provider.config.RegistrySHA256 != "" {
		external.Expect = &expectV2{}
		if provider.config.PackageSHA256 != "" {
			external.Expect.PackageSHA256 = &provider.config.PackageSHA256
		}
		if provider.config.RegistrySHA256 != "" {
			registry := provider.config.RegistrySHA256
			external.Expect.ThinkingControlsSHA256 = &registry
		}
		if provider.config.Decisions.OfferSHA256 != "" {
			offer := provider.config.Decisions.OfferSHA256
			external.Expect.OfferSHA256 = &offer
		}
	}
	if !input.Event.is(EventSidecall) {
		for _, entry := range input.State.Attribution {
			external.History = append(external.History, historyV2{Message: entry.Message, ActionID: entry.ActionID})
		}
	}
	// last_served is the truthful record of what last ran. It survives prefix
	// invalidation and is sent even when that target is not offered or fits.
	if held := input.State.HeldActionID; held != "" {
		served := &lastServedV2{ActionID: held, Worker: input.State.HeldWorker}
		if served.Worker == "" {
			served.Worker = provider.actions[held].Worker
		}
		for _, entry := range external.History {
			if entry.ActionID == held && (served.Message == nil || entry.Message > *served.Message) {
				message := entry.Message
				served.Message = &message
			}
		}
		external.LastServed = served
	}
	v1 := provider.project(input)
	cache := make(map[string]*cacheV2, len(v1.Workers))
	for _, worker := range v1.Workers {
		cache[worker.ActionID] = &cacheV2{ContextEpoch: worker.ContextEpoch, Gap: worker.Gap, TTL: worker.TTL, Residency: worker.Residency, Usage: worker.Usage, CurrentPrefix: worker.CurrentPrefix}
	}
	ids := make([]string, 0, len(provider.actions))
	for id, action := range provider.actions {
		if action.DeploymentID != "" {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	for _, id := range ids {
		action := provider.actions[id]
		external.Targets = append(external.Targets, targetV2{ActionID: id, Worker: action.Worker, Placement: provider.deployments[action.DeploymentID].Kind, Cache: cache[id], Capacity: input.Capacity[id]})
	}
	return external
}

// DecisionReceipt is the service's per-result explanation (v2 only).
type DecisionReceipt struct {
	ActionID  string  `json:"action_id"`
	ControlID string  `json:"control_id"`
	Policy    *string `json:"policy"`
	DecidedBy struct {
		// Model is null for a one-stage package.
		Model    *string `json:"model"`
		Thinking string  `json:"thinking"`
	} `json:"decided_by"`
	Eligible      map[string]*string `json:"eligible"`
	HeldFrom      *string            `json:"held_from"`
	Excluded      []excludedTarget   `json:"excluded"`
	Basis         string             `json:"basis"`
	CapacityBasis *string            `json:"capacity_basis,omitempty"`
	// CapacityUnknown lists offered actions admitted with unknown capacity.
	CapacityUnknown []string `json:"capacity_unknown,omitempty"`
}

type excludedTarget struct {
	ActionID string `json:"action_id"`
	Reason   string `json:"reason"`
}

type executionEchoV2 struct {
	Status  string            `json:"status"`
	Version string            `json:"version"`
	Turn    turnV2            `json:"turn"`
	Pinned  map[string]string `json:"pinned"`
}

type policyEchoV2 struct {
	Status string `json:"status"`
	Preset string `json:"preset"`
}

func sameTurn(left, right turnV2) bool {
	equal := func(a, b *string) bool { return (a == nil) == (b == nil) && (a == nil || *a == *b) }
	return left.ID == right.ID && left.Attempt == right.Attempt && left.Epoch == right.Epoch && equal(left.RetryOf, right.RetryOf) && equal(left.Reconsult, right.Reconsult)
}

// validateV2 checks the service applied this turn's envelope and policy and
// that its receipt explains the selected action without contradicting a
// capacity fact the core reported. It never substitutes another action.
func validateV2(raw []byte, selected publicDecision, identity *decisionIdentity, sent contextV2, preset string) (*DecisionReceipt, error) {
	var echo struct {
		Routing struct {
			Execution *executionEchoV2 `json:"execution"`
			Policy    *policyEchoV2    `json:"policy"`
		} `json:"routing"`
	}
	if err := json.Unmarshal(raw, &echo); err != nil {
		return nil, fmt.Errorf("invalid Decisions response")
	}
	execution := echo.Routing.Execution
	if execution == nil || execution.Status != "applied" || execution.Version != executionContextV2 || !sameTurn(execution.Turn, sent.Turn) {
		return nil, fmt.Errorf("Decisions did not apply this v2 execution context")
	}
	if sent.Expect != nil && execution.Pinned == nil {
		return nil, fmt.Errorf("Decisions applied the v2 execution context without confirming its pins")
	}
	if sent.Expect != nil {
		pins := []string{"thinking_controls_sha256"}
		if sent.Expect.PackageSHA256 != nil {
			pins = append(pins, "package_sha256")
		}
		if sent.Expect.OfferSHA256 != nil {
			pins = append(pins, "offer_sha256")
		}
		for _, name := range pins {
			if execution.Pinned[name] != "matched" {
				return nil, fmt.Errorf("Decisions did not confirm pinned %s", name)
			}
		}
	}
	if policy := echo.Routing.Policy; policy == nil || policy.Status != "applied" || policy.Preset != preset {
		return nil, fmt.Errorf("Decisions did not apply the requested routing.policy preset")
	}
	receipt := selected.Receipt
	if receipt == nil || receipt.ActionID != identity.ActionID || receipt.ControlID != identity.ControlID || (receipt.Policy != nil && *receipt.Policy != preset) {
		return nil, fmt.Errorf("Decisions result receipt does not name the selected action and applied policy")
	}
	if model := receipt.DecidedBy.Model; model != nil {
		switch *model {
		case "schedule", "held", "capacity_handoff", "held_unavailable", "prefer_local_threshold", "utility_fallback":
		default:
			return nil, fmt.Errorf("Decisions receipt has unknown decided_by.model %q", *model)
		}
	}
	switch receipt.DecidedBy.Thinking {
	case "schedule", "held":
	default:
		return nil, fmt.Errorf("Decisions receipt has unknown decided_by.thinking %q", receipt.DecidedBy.Thinking)
	}
	excluded := make(map[string]string, len(receipt.Excluded))
	for _, entry := range receipt.Excluded {
		excluded[entry.ActionID] = entry.Reason
	}
	for _, target := range sent.Targets {
		if target.Capacity == nil || target.Capacity.Fits {
			continue
		}
		if target.ActionID == identity.ActionID {
			return nil, fmt.Errorf("Decisions selected a target the core reported does not fit")
		}
		if reason := excluded[target.ActionID]; target.Capacity.Reason == nil || reason != *target.Capacity.Reason {
			return nil, fmt.Errorf("Decisions receipt omitted an infeasible target's exclusion")
		}
	}
	return receipt, nil
}

// capacitySummary is the known capacity facts for logs, keyed by action.
func capacitySummary(targets []targetV2) map[string]*CapacityFact {
	facts := map[string]*CapacityFact{}
	for _, target := range targets {
		if target.Capacity != nil {
			facts[target.ActionID] = target.Capacity
		}
	}
	return facts
}

// stringOrNil logs an optional string by value rather than by address.
func stringOrNil(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

// decidedBy renders the receipt's per-dimension decision for headers and logs.
func (receipt *DecisionReceipt) decidedBy() string {
	model := "one_stage"
	if receipt.DecidedBy.Model != nil {
		model = *receipt.DecidedBy.Model
	}
	return model + "/" + receipt.DecidedBy.Thinking
}
