package extproc

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A private (privacy or zdr) turn is decided by the policy service like any
// other, but the service is to hold its content for that decide only. The
// turn names a fresh episode to the service (policyServiceEpisode), binds
// ephemeral (policySessionBindingFor), never stages or appends
// (policyDeltaEligible), and once it is decided, whether or not the decide
// succeeded, the service is asked to drop the episode (POST
// /v1/rayline/arc/policy/release) on every session or replica the turn
// called, and the sessions the turn started are ended. The release runs off
// the turn's path, bounded by policyPrivateReleaseTimeout per call; a
// service that predates it answers 404 and keeps the episode until it
// evicts it, as before.

// policyServiceEpisode is the episode a turn's decide names to the policy
// service. A standard turn names its own. A private turn names a fresh
// random one, so the service can neither join it to the episode's other
// turns nor touch the episode's standard-mode state, even on a container
// that holds it, and its release drops exactly what that decide left.
func policyServiceEpisode(arcContext *selection.RaylineARCSelectionContext) string {
	if !arcContext.PrivateTurn {
		return arcContext.EpisodeIDHash
	}
	var random [32]byte
	if _, err := rand.Read(random[:]); err != nil {
		// crypto/rand does not fail on supported platforms; a fixed id would
		// join private turns, so fall back to a time-and-request derived one.
		sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", arcContext.RequestID, time.Now().UnixNano())))
		return hex.EncodeToString(sum[:])
	}
	return hex.EncodeToString(random[:])
}

// policyPrivateReleaseTimeout bounds each release call, its one retry
// included.
const policyPrivateReleaseTimeout = raylinearc.PolicyReleaseTimeout

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
		EpisodeIDHash: request.EpisodeIDHash,
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
