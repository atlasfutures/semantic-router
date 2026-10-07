package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// emptyOffer is what a policy turn knew when it was left nothing to offer.
// A turn fails policy_no_available_action before any service call, and the
// failure line alone does not say which constraint emptied the offer: a held
// model, the hard mask (vision, capability, context, operator), the tool-loop
// family hold, or the fallback's exclusions. On the 2026-10-07 lab run these
// 503s came on held turns with the fallback off, and nothing logged could
// tell them apart.
type emptyOffer struct {
	// stage is where the offer emptied: "offer", before any service call,
	// or "derived_hold", after a two-stage package refused a cold turn.
	stage    string
	held     int
	retained bool
	sideCall bool
	fallback bool
	excluded []bool
	hard     []bool
	cellOut  map[int]string
	turn     *raylinearc.PolicyEpisodeState
	// derivedModel and derivedSource name the hold a two-stage package's
	// cold-turn retry derived; it is the model whose exclusion emptied a
	// derived_hold offer, and no arm is held there.
	derivedModel  string
	derivedSource string
}

// logRaylineARCEmptyOffer records an empty offer's inputs, every one of them
// a router-owned fact (no request content), so the next occurrence names its
// cause.
func logRaylineARCEmptyOffer(
	arcContext *selection.RaylineARCSelectionContext,
	scorer *policyServiceScorer,
	offer emptyOffer,
) {
	heldModel := ""
	if offer.held >= 0 && offer.held < len(scorer.workerIDs) {
		heldModel = scorer.armModel(offer.held)
	}
	actionsPerArm := make([]int, len(scorer.workerIDs))
	for _, actionID := range scorer.actionOrder {
		if arm := scorer.bindings[actionID].arm; arm >= 0 && arm < len(actionsPerArm) {
			actionsPerArm[arm]++
		}
	}
	routes := map[string]string{}
	for arm, class := range offer.cellOut {
		if arm >= 0 && arm < len(scorer.workerIDs) {
			routes[scorer.workerIDs[arm]] = class
		}
	}
	excludedModels := 0
	if offer.turn != nil {
		excludedModels = len(offer.turn.Exclusions)
	}
	logging.ComponentWarnEvent("extproc", "rayline_arc_empty_offer", map[string]interface{}{
		"empty_at":           offer.stage,
		"episode_id_hash":    arcContext.EpisodeIDHash,
		"request_format":     arcContext.RequestFormat,
		"held_arm":           offer.held,
		"held_model":         heldModel,
		"derived_model":      offer.derivedModel,
		"derived_source":     offer.derivedSource,
		"boundary_retained":  offer.retained,
		"side_call":          offer.sideCall,
		"fallback":           offer.fallback,
		"image_bearing":      arcContext.ImageBearing,
		"excluded_arms":      offer.excluded,
		"hard_excluded_arms": offer.hard,
		"actions_per_arm":    actionsPerArm,
		"workers":            scorer.workerIDs,
		"excluded_routes":    routes,
		"context_exclusions": excludedModels,
		"tool_loop_family":   arcContext.ToolLoopFamily,
		"tool_loop_arm":      arcContext.ToolLoopArm,
	})
}
