package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// --- Stopgap: a plain-control target scored as its model's steer-none action ---
//
// An executor derives its targets from its thinking-controls registry. On a
// wire format where the registry admits no steering instruction for a model
// (Chat Completions and Responses for many models), the executor can offer
// that model only under its plain control: its base with no instruction. A
// v5 package may bind the model only under steering controls, so the plain
// target is unknown here and, before this stopgap, was left out of the offer:
// the model was never offered on those formats.
//
// When a v2 lookup's target is the plain action of a model whose package
// binds that base's "none" steering action and not the plain one, and the
// lookup does not already offer that "none" action, the target is scored as
// the "none" action (same model, same base; "none" adds no instruction except
// a neutral one when the level changes) and answered back under the plain
// action and control, which is what the executor offered and runs. The score
// is borrowed; what executes is exactly the plain action. Each lookup where
// it applies logs rayline_arc_route_target_stand_in.
//
// Remove this file, its two call sites and its tests once the serving
// package binds the plain action itself: plainStandInsOf then finds nothing,
// so the event stops firing.

// plainStandIn is one plain action the package lacks, the "none" steering
// action it is scored as, and the plain control the answer names.
type plainStandIn struct {
	standInActionID string
	plainControlID  string
	model           string
}

// plainStandInsOf maps each plain action the cell does not bind to the bound
// "none" steering action of the same model and base, for v5 bindings only;
// nil when there is none.
func plainStandInsOf(policy *config.RaylineARCPolicyServiceConfig) map[string]plainStandIn {
	if policy == nil {
		return nil
	}
	bound := make(map[string]bool, len(policy.Bindings))
	for _, binding := range policy.Bindings {
		bound[binding.ActionID] = true
	}
	standIns := map[string]plainStandIn{}
	for actionID := range bound {
		action, ok := policy.PackageV5Action(actionID)
		if !ok || action.Control.Instruction == nil || action.Control.Instruction.Level != "none" {
			continue
		}
		plain := action.Control
		plain.Instruction = nil
		controlID := plain.ID()
		plainID := v5ActionID(controlID, action.Model)
		if bound[plainID] {
			continue
		}
		standIns[plainID] = plainStandIn{standInActionID: actionID, plainControlID: controlID, model: action.Model}
	}
	if len(standIns) == 0 {
		return nil
	}
	return standIns
}

// v5ActionID is a v5 action's identity: sha256 of the JCS form of
// {"control_id","model"}.
func v5ActionID(controlID, model string) string {
	raw, err := json.Marshal(struct {
		ControlID string `json:"control_id"`
		Model     string `json:"model"`
	}{controlID, model})
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// applyPlainStandIns rewrites a v2 envelope before it is decided, once per
// lookup and before identical lookups are coalesced, so each caller's own
// envelope says what it was answered under. A target that is a plain action
// the cell lacks becomes its stand-in (its capacity, infeasibility and cache
// evidence with it), unless the lookup already offers that stand-in; each
// such target is recorded on the envelope for the answer and logged. A plain
// action the executor attributes, last served or holds becomes its stand-in
// whatever the targets: it ran as the stand-in's model and base, and an
// unknown claim would refuse the lookup.
func applyPlainStandIns(v2 *selection.RaylineARCRouteExecutionV2, standIns map[string]plainStandIn, episodeIDHash string) {
	if v2 == nil || len(standIns) == 0 {
		return
	}
	targets := map[string]string{}
	for _, target := range v2.TargetActionIDs {
		standIn, ok := standIns[target]
		if !ok || slices.Contains(v2.TargetActionIDs, standIn.standInActionID) {
			continue
		}
		targets[target] = standIn.standInActionID
		if v2.StandIns == nil {
			v2.StandIns = map[string]selection.RaylineARCPlainStandIn{}
		}
		v2.StandIns[standIn.standInActionID] = selection.RaylineARCPlainStandIn{
			PlainActionID: target, PlainControlID: standIn.plainControlID,
		}
		logging.ComponentWarnEvent("extproc", "rayline_arc_route_target_stand_in", map[string]interface{}{
			"model": standIn.model, "plain_action_id": target, "stand_in_action_id": standIn.standInActionID,
			"episode_id_hash": episodeIDHash,
		})
	}
	target := func(actionID string) string {
		if to, ok := targets[actionID]; ok {
			return to
		}
		return actionID
	}
	claim := func(actionID string) string {
		if standIn, ok := standIns[actionID]; ok {
			return standIn.standInActionID
		}
		return actionID
	}
	for index := range v2.TargetActionIDs {
		v2.TargetActionIDs[index] = target(v2.TargetActionIDs[index])
	}
	for index := range v2.Infeasible {
		v2.Infeasible[index] = target(v2.Infeasible[index])
	}
	for index := range v2.Capacity {
		v2.Capacity[index].ActionID = target(v2.Capacity[index].ActionID)
	}
	if v2.CacheContext != nil {
		for index := range v2.CacheContext.Actions {
			v2.CacheContext.Actions[index].ActionID = target(v2.CacheContext.Actions[index].ActionID)
		}
	}
	for index := range v2.ActionIDs {
		v2.ActionIDs[index] = claim(v2.ActionIDs[index])
	}
	for index := range v2.Attribution {
		v2.Attribution[index].ActionID = claim(v2.Attribution[index].ActionID)
	}
	for _, field := range []**string{&v2.LastServedActionID, &v2.HoldActionID} {
		if *field != nil {
			renamed := claim(**field)
			*field = &renamed
		}
	}
}

// restorePlainStandIns answers each stand-in the decision names under the
// plain action and control the executor offered. The arm and the thinking
// level, which describe the stand-in's training and steering, are cleared:
// the plain action has no trained arm and no steering level.
func restorePlainStandIns(facts *routerruntime.RouteDecision, standIns map[string]selection.RaylineARCPlainStandIn) {
	if len(standIns) == 0 {
		return
	}
	if plain, ok := standIns[facts.PolicyActionID]; ok {
		facts.PolicyActionID, facts.ControlID, facts.ArmID, facts.ThinkingLevel = plain.PlainActionID, plain.PlainControlID, "", ""
	}
	for index := range facts.Results {
		result := &facts.Results[index]
		if plain, ok := standIns[result.PolicyActionID]; ok {
			result.PolicyActionID, result.ControlID, result.ArmID, result.ThinkingLevel = plain.PlainActionID, plain.PlainControlID, "", ""
		}
	}
	if facts.Receipt == nil {
		return
	}
	receipt := *facts.Receipt
	receipt.Excluded = slices.Clone(facts.Receipt.Excluded)
	for index, entry := range receipt.Excluded {
		if plain, ok := standIns[entry.ActionID]; ok {
			receipt.Excluded[index].ActionID = plain.PlainActionID
		}
	}
	receipt.CapacityUnknown = slices.Clone(facts.Receipt.CapacityUnknown)
	for index, actionID := range receipt.CapacityUnknown {
		if plain, ok := standIns[actionID]; ok {
			receipt.CapacityUnknown[index] = plain.PlainActionID
		}
	}
	if held := facts.Receipt.HeldFrom; held != nil {
		if plain, ok := standIns[*held]; ok {
			value := plain.PlainActionID
			receipt.HeldFrom = &value
		}
	}
	facts.Receipt = &receipt
}
