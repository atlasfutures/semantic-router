package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A private (privacy or zdr) turn is decided by the policy service like any
// other, but the service is to hold its content for that decide only. The
// turn binds ephemeral (policySessionBindingFor), never stages or appends
// (policyDeltaEligible), and once it is decided, whether or not the decide
// succeeded, the service is asked to drop the episode (POST
// /v1/rayline/arc/policy/release) on every session or replica the turn
// called, and the sessions the turn started are ended. The release runs off
// the turn's path, bounded by policyPrivateReleaseTimeout per call; a
// service that predates it answers 404 and keeps the episode until it
// evicts it, as before.

// policyPrivateReleaseTimeout bounds each release call.
const policyPrivateReleaseTimeout = 5 * time.Second

// runPolicyPrivateRelease runs a private turn's release. It is a variable so
// tests can run the release inline and read what it sent.
var runPolicyPrivateRelease = func(release func()) { go release() }

// releasePrivatePolicyEpisode asks the policy service to drop what it holds
// for a private turn's episode and ends the sessions the turn started.
func releasePrivatePolicyEpisode(
	ctx context.Context,
	arcContext *selection.RaylineARCSelectionContext,
	request raylinearc.PolicyDecisionRequest,
	decider policyDecider,
	replicaBinding *raylinearc.PolicyServiceBinding,
	sessionBinding *raylinearc.PolicySessionBinding,
) {
	release := raylinearc.PolicyReleaseRequest{
		SchemaVersion: raylinearc.PolicyReleaseRequestSchema,
		Package:       request.Package,
		EpisodeIDHash: arcContext.EpisodeIDHash,
	}
	requestID := arcContext.RequestID
	detached := context.WithoutCancel(ctx)
	runPolicyPrivateRelease(func() {
		outcome := raylinearc.PolicyReleaseNone
		switch {
		case sessionBinding != nil:
			outcome = sessionBinding.ReleaseEpisode(detached, release)
			sessionBinding.Release(detached)
		case replicaBinding != nil:
			outcome = replicaBinding.ReleaseEpisode(detached, release, policyPrivateReleaseTimeout)
		default:
			if client, ok := decider.(*raylinearc.PolicyServiceClient); ok && client != nil {
				releaseCtx, cancel := context.WithTimeout(detached, policyPrivateReleaseTimeout)
				outcome = raylinearc.PolicyReleaseOutcomeOf(client.Release(releaseCtx, release))
				cancel()
			}
		}
		fields := map[string]interface{}{
			"request_id": requestID, "episode_id_hash": release.EpisodeIDHash, "outcome": outcome,
		}
		if outcome == raylinearc.PolicyReleaseFailed {
			logging.ComponentWarnEvent("extproc", "rayline_arc_policy_private_release", fields)
			return
		}
		logging.ComponentEvent("extproc", "rayline_arc_policy_private_release", fields)
	})
}
