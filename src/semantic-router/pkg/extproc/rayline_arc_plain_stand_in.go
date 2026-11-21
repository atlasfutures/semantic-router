package extproc

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// --- Stopgap: a plain-control target scored as its model's none-level action ---
//
// An executor derives its targets from its thinking-controls registry. On a
// wire format where the registry admits no steering instruction for a model
// (Chat Completions and Responses for many models), the executor can offer
// and run that model only under its plain control: its base with no
// instruction. A v5 package may bind the model only under steering controls,
// so the plain target is unknown here and is left out of the offer: the model
// is never offered on those formats.
//
// The stopgap is one alias per such model: its plain action is read as its
// bound "none" steering action (same model and base; "none" adds no
// instruction except a neutral one when the level changes). A v2 lookup that
// names the plain action and never the "none" one is rewritten on the way in,
// every action id field at once, so the envelope's own checks run on it as
// they would on a cell that bound the plain action; the answer is rewritten
// back on the way out, so the executor sees only the plain action it offered
// and runs. The score is borrowed from the "none" action; what executes is
// exactly the plain action. An envelope that names the "none" action anywhere
// is left as it is, and decides as without this stopgap. Each lookup that used
// an alias logs rayline_arc_route_target_stand_in after its decision.
//
// Remove this file, its call sites and its tests once the serving package
// binds the plain action itself: plainAliasesOf then finds nothing for that
// model, so the event stops firing.

// plainAlias is the "none" action a plain action is read as, and the plain
// control the answer names.
type plainAlias struct {
	standInActionID string
	plainControlID  string
	model           string
}

// plainAliasCache holds plainAliasesOf by what it depends on, the package and
// the cell's bound action ids, never by config: a reload builds a new config,
// which must not be kept alive here. Entries are only the small alias maps,
// one per distinct package and binding set the process has served.
var plainAliasCache sync.Map

// plainAliasesOf maps each plain action that has a stand-in on this cell to
// it; nil when none has.
func plainAliasesOf(policy *config.RaylineARCPolicyServiceConfig) map[string]plainAlias {
	if policy == nil || !policy.IsPackageV5() {
		return nil // a v4 package names no controls: nothing to alias, nothing to load
	}
	ids := make([]string, 0, len(policy.Bindings))
	for _, binding := range policy.Bindings {
		ids = append(ids, binding.ActionID)
	}
	slices.Sort(ids)
	key := policy.PackageSHA256 + "\x00" + strings.Join(ids, ",")
	if cached, ok := plainAliasCache.Load(key); ok {
		return cached.(map[string]plainAlias)
	}
	bound := make(map[string]bool, len(ids))
	actions := map[string]config.RaylineARCPolicyActionV5{}
	for _, actionID := range ids {
		bound[actionID] = true
		if action, ok := policy.PackageV5Action(actionID); ok {
			actions[actionID] = action
		}
	}
	if len(actions) == 0 {
		// No v5 action read: not a v5 package, or one that did not load
		// this time. Nothing is cached, so a later load is read again.
		return nil
	}
	inPackage := func(actionID string) bool {
		_, ok := policy.PackageV5Action(actionID)
		return ok
	}
	aliases := plainAliasesOfActions(actions, func(actionID string) bool { return bound[actionID] || inPackage(actionID) })
	plainAliasCache.Store(key, aliases)
	return aliases
}

// plainAliasesOfActions maps the plain action of each bound v5 "none" action
// to that action, unless the plain action is known (bound, or in the package
// though the cell leaves it unbound) or more than one "none" action reads as
// it (another neutral text, say): it then keeps no alias.
func plainAliasesOfActions(actions map[string]config.RaylineARCPolicyActionV5, known func(actionID string) bool) map[string]plainAlias {
	aliases := map[string]plainAlias{}
	ambiguous := map[string]bool{}
	for actionID, action := range actions {
		if action.Control.Instruction == nil || action.Control.Instruction.Level != "none" {
			continue
		}
		plain := action.Control
		plain.Instruction = nil
		controlID := plain.ID()
		plainID := v5ActionID(controlID, action.Model)
		if known(plainID) {
			continue
		}
		if _, seen := aliases[plainID]; seen {
			ambiguous[plainID] = true
		}
		aliases[plainID] = plainAlias{standInActionID: actionID, plainControlID: controlID, model: action.Model}
	}
	for plainID := range ambiguous {
		delete(aliases, plainID)
	}
	if len(aliases) == 0 {
		return nil
	}
	return aliases
}

// v5ActionID is a v5 action's identity: sha256 of the RFC 8785 form of
// {"control_id","model"}.
func v5ActionID(controlID, model string) string {
	sum := sha256.Sum256([]byte(`{"control_id":` + jcsString(controlID) + `,"model":` + jcsString(model) + `}`))
	return hex.EncodeToString(sum[:])
}

// applyPlainAliases rewrites a v2 envelope before it is decided, once per
// lookup and before identical lookups are coalesced, so each caller's own
// envelope records what it is answered under. For each alias whose plain
// action the envelope names and whose "none" action it never names, every
// action id field is rewritten from the plain action to the "none" one and
// the alias is recorded on the envelope (StandIns).
func applyPlainAliases(v2 *selection.RaylineARCRouteExecutionV2, aliases map[string]plainAlias) {
	if v2 == nil || len(aliases) == 0 {
		return
	}
	named := v2Named(v2)
	renames := map[string]string{}
	for plainID, alias := range aliases {
		if named[plainID] && !named[alias.standInActionID] {
			renames[plainID] = alias.standInActionID
			if v2.StandIns == nil {
				v2.StandIns = map[string]selection.RaylineARCPlainStandIn{}
			}
			v2.StandIns[alias.standInActionID] = selection.RaylineARCPlainStandIn{
				PlainActionID: plainID, PlainControlID: alias.plainControlID, Model: alias.model,
			}
		}
	}
	if len(renames) > 0 {
		v2RenameActions(v2, func(actionID string) string {
			if to, ok := renames[actionID]; ok {
				return to
			}
			return actionID
		})
	}
}

// v2Named is every action id a v2 envelope names, in any field.
func v2Named(v2 *selection.RaylineARCRouteExecutionV2) map[string]bool {
	named := map[string]bool{}
	v2RenameActions(v2, func(actionID string) string {
		named[actionID] = true
		return actionID
	})
	return named
}

// v2RenameActions applies rename to every action id field of a v2 envelope.
func v2RenameActions(v2 *selection.RaylineARCRouteExecutionV2, rename func(string) string) {
	for _, ids := range [][]string{v2.TargetActionIDs, v2.ActionIDs, v2.Infeasible} {
		for index := range ids {
			ids[index] = rename(ids[index])
		}
	}
	for index := range v2.Capacity {
		v2.Capacity[index].ActionID = rename(v2.Capacity[index].ActionID)
	}
	for index := range v2.Attribution {
		v2.Attribution[index].ActionID = rename(v2.Attribution[index].ActionID)
	}
	if v2.CacheContext != nil {
		for index := range v2.CacheContext.Actions {
			v2.CacheContext.Actions[index].ActionID = rename(v2.CacheContext.Actions[index].ActionID)
		}
	}
	for _, field := range []**string{&v2.LastServedActionID, &v2.HoldActionID} {
		if *field != nil {
			renamed := rename(**field)
			*field = &renamed
		}
	}
}

// logPlainAliases records, once per lookup that used an alias, that it did
// and how its decision ended: decided, or failed (preparation and selection
// failures alike). A routes answer that fails after a decided lookup is
// logged by the routes failure event of the same request.
func logPlainAliases(v2 *selection.RaylineARCRouteExecutionV2, episodeIDHash string, err error) {
	if v2 == nil || len(v2.StandIns) == 0 {
		return
	}
	outcome := "decided"
	if err != nil {
		outcome = "failed"
	}
	standInIDs := make([]string, 0, len(v2.StandIns))
	for standInID := range v2.StandIns {
		standInIDs = append(standInIDs, standInID)
	}
	slices.Sort(standInIDs)
	models, plainIDs := make([]string, 0, len(standInIDs)), make([]string, 0, len(standInIDs))
	for _, standInID := range standInIDs {
		models = append(models, v2.StandIns[standInID].Model)
		plainIDs = append(plainIDs, v2.StandIns[standInID].PlainActionID)
	}
	// One event per lookup, naming every alias it used, so the count is the
	// number of affected lookups.
	logging.ComponentEvent("extproc", "rayline_arc_route_target_stand_in", map[string]interface{}{
		"models": models, "plain_action_ids": plainIDs, "stand_in_action_ids": standInIDs,
		"outcome": outcome, "episode_id_hash": episodeIDHash,
	})
}

// renamePlainAliasesInAnswer rewrites a routes answer so it names only what
// the executor offered: every occurrence of a stand-in action id in what the
// cell decided becomes its plain action id, and in an object that named the
// stand-in as its action (action_id or policy_action_id), the control becomes
// the plain control and the arm and steering level are cleared: the plain
// action has no trained arm and no steering level. The routing echo is the
// caller's own members (opaque turn ids among them) and the signals block are
// answered as they were written. The answer is returned unchanged without stand-ins.
func renamePlainAliasesInAnswer(answer []byte, standIns map[string]selection.RaylineARCPlainStandIn) ([]byte, error) {
	if len(standIns) == 0 {
		return answer, nil
	}
	var tree map[string]any
	decoder := json.NewDecoder(bytes.NewReader(answer))
	decoder.UseNumber() // numbers are kept as the answer wrote them
	if err := decoder.Decode(&tree); err != nil {
		return nil, err
	}
	// The routing echo (the caller's own members) and the signals block (the
	// service's, passed through as written) are opaque: never rewritten.
	opaque := map[string]any{}
	for _, key := range []string{"routing", "signals"} {
		if value, present := tree[key]; present {
			opaque[key] = value
			delete(tree, key)
		}
	}
	tree = renamePlainAliasesIn(tree, standIns).(map[string]any)
	for key, value := range opaque {
		tree[key] = value
	}
	return json.Marshal(tree)
}

func renamePlainAliasesIn(node any, standIns map[string]selection.RaylineARCPlainStandIn) any {
	switch value := node.(type) {
	case string:
		if plain, ok := standIns[value]; ok {
			return plain.PlainActionID
		}
		return value
	case []any:
		for index := range value {
			value[index] = renamePlainAliasesIn(value[index], standIns)
		}
		return value
	case map[string]any:
		var named *selection.RaylineARCPlainStandIn
		for _, key := range []string{"action_id", "policy_action_id"} {
			if id, ok := value[key].(string); ok {
				if plain, found := standIns[id]; found {
					named = &plain
				}
			}
		}
		for key := range value {
			value[key] = renamePlainAliasesIn(value[key], standIns)
		}
		if named != nil {
			for key, cleared := range map[string]any{"control_id": named.PlainControlID, "arm_id": nil} {
				if _, present := value[key]; present {
					value[key] = cleared
				}
			}
			// The answer omits a steering level an action does not have.
			delete(value, "thinking_level")
		}
		return value
	default:
		return node
	}
}
