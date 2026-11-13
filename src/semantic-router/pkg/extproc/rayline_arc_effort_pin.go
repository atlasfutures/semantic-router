package extproc

import (
	"sort"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A worker's named effort is fixed for the episode (pathfinder ADR 0085,
// decision 1): the native base and budget a request sends are part of the
// provider's prompt-cache key, so a change between turns restarts the
// worker's cached transcript after the tools and system prompt. A v5 package
// can bind one worker to actions at several bases, and a policy drawing among
// them turn by turn would pay that miss on every switch. Once a worker's
// transcript has a placer, its offer keeps only the actions dispatched at
// that transcript's base and budget; the per-turn lever (the steering
// instruction) still varies freely.

// effortBase is a control's request-level shape: its native base and
// budget.
type effortBase struct {
	native   string
	budget   int64
	budgeted bool
}

func effortBaseOf(control thinkingcontrol.Control) effortBase {
	base := effortBase{native: control.Native}
	if control.BudgetTokens != nil {
		base.budget, base.budgeted = *control.BudgetTokens, true
	}
	return base
}

// effortPins are the bases this episode's workers are fixed at: per worker,
// the first control of its most recently dispatched transcript.
func effortPins(controls []raylinearc.ControlPlacement) map[string]effortBase {
	pins := map[string]effortBase{}
	for _, placement := range controls {
		worker, _, found := strings.Cut(placement.Key, "|")
		if !found || placement.State.First == nil {
			continue
		}
		pins[worker] = effortBaseOf(*placement.State.First)
	}
	return pins
}

// effortPinned reports whether the offer leaves actionID out: a v5 action
// dispatched on a worker this episode has fixed at another base.
func (scorer *policyServiceScorer) effortPinned(actionID string, pins map[string]effortBase) bool {
	binding := scorer.bindings[actionID]
	if binding.base == nil || binding.dispatchArm < 0 || binding.dispatchArm >= len(scorer.workerIDs) {
		return false
	}
	pin, ok := pins[scorer.workerIDs[binding.dispatchArm]]
	return ok && pin != *binding.base
}

// logRaylineARCEffortPin records a turn whose offer the pins narrowed, or
// whose pins were lifted because they left nothing to offer (that turn
// switches base, a cache boundary, rather than failing).
func logRaylineARCEffortPin(
	arcContext *selection.RaylineARCSelectionContext, pins map[string]effortBase, masked int, lifted bool,
) {
	if masked == 0 && !lifted {
		return
	}
	workers := make([]string, 0, len(pins))
	for worker := range pins {
		workers = append(workers, worker)
	}
	sort.Strings(workers)
	logging.ComponentEvent("extproc", "rayline_arc_effort_pin", map[string]interface{}{
		"episode_id_hash": arcContext.EpisodeIDHash,
		"pinned_workers":  workers,
		"masked_actions":  masked,
		"lifted":          lifted,
	})
}
