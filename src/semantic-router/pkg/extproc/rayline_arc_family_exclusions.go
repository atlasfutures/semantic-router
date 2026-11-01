package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// familyExclusionTimeout bounds a session family read or write. Both are
// advice: a slow or failed store leaves the turn with the context
// exclusions it has, never fails it.
const familyExclusionTimeout = 150 * time.Millisecond

// loadRaylineARCFamilyExclusions reads the families this request's session
// excludes, when the cell's fallback excludes families and the request has a
// session.
func (r *OpenAIRouter) loadRaylineARCFamilyExclusions(
	policy *config.RaylineARCPolicyServiceConfig, reqCtx *RequestContext,
) []string {
	if !policy.FamilyExclusionEnabled() || reqCtx == nil || reqCtx.RaylineARCSessionKey == "" {
		return nil
	}
	store, ok := r.raylineARCEpisodeStoreFor(reqCtx).(raylinearc.SessionFamilyExclusionStore)
	if !ok {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), familyExclusionTimeout)
	defer cancel()
	families, err := store.LoadSessionFamilyExclusions(ctx, reqCtx.RaylineARCSessionKey)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "rayline_arc_family_exclusion_read_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
		})
		return nil
	}
	return families
}

// noteRaylineARCFamilyExclusion excludes the refusing model's family for
// the session, when the cell's fallback excludes families. It returns the
// family excluded, or "".
func noteRaylineARCFamilyExclusion(ctx *RequestContext, model string) string {
	if ctx == nil || ctx.RaylineARCSessionKey == "" || ctx.RaylineARCTransaction == nil {
		return ""
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil {
		return ""
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	family := raylinearc.ModelFamily(model)
	if !policy.FamilyExclusionEnabled() || family == "" {
		return ""
	}
	store, ok := ctx.RaylineARCTransaction.store.(raylinearc.SessionFamilyExclusionStore)
	if !ok {
		return ""
	}
	writeContext, cancel := context.WithTimeout(context.Background(), familyExclusionTimeout)
	defer cancel()
	ttl := policy.FamilyExclusionTTL()
	fields := map[string]interface{}{
		"request_id": ctx.RequestID, "family": family, "model": model, "ttl_seconds": ttl.Seconds(),
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
	}
	if err := store.AddSessionFamilyExclusion(writeContext, ctx.RaylineARCSessionKey, family, ttl); err != nil {
		logging.ComponentWarnEvent("extproc", "rayline_arc_family_exclusion_write_failed", fields)
		return ""
	}
	logging.ComponentEvent("extproc", "rayline_arc_family_exclusion", fields)
	return family
}
