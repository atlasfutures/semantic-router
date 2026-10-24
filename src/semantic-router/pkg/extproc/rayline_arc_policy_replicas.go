package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// policyDecider is what a turn's decide calls go through: the cell's one
// policy-service client, or the turn's binding to the episode's replica.
type policyDecider interface {
	Decide(context.Context, raylinearc.PolicyDecisionRequest) (*raylinearc.PolicyDecisionResponse, error)
}

// newRaylineARCPolicyServicePool builds the replica pool of a cell that
// configures policy_service.replicas.
func newRaylineARCPolicyServicePool(
	policy *config.RaylineARCPolicyServiceConfig,
	clientFor func(string) *raylinearc.PolicyServiceClient,
) (*raylinearc.PolicyServicePool, error) {
	replicas := make([]raylinearc.PolicyServiceReplica, 0, len(policy.Replicas))
	for _, configured := range policy.Replicas {
		replicas = append(replicas, raylinearc.PolicyServiceReplica{
			ID:     configured.ID,
			State:  raylinearc.EncoderReplicaState(configured.State),
			Client: clientFor(configured.BaseURL),
		})
	}
	return raylinearc.NewPolicyServicePool(replicas, raylinearc.PolicyServicePoolConfig{
		SchemaVersion:          policy.Failover.SchemaVersion,
		UnavailableStatusCodes: append([]int(nil), policy.Failover.UnavailableStatusCodes...),
		UnavailableCooldown:    time.Duration(policy.Failover.UnavailableCooldownSeconds) * time.Second,
		MaxRemaps:              policy.Failover.MaxRemaps,
	})
}

// policyServiceMode reports whether the cell decides through a policy
// service, one client or a replica pool.
func (armed *raylineARCArmedComponents) policyServiceMode() bool {
	return armed != nil && (armed.policy != nil || armed.policyPool != nil)
}

// policyDeciderFor is the turn's decider. With replicas it is a binding to
// the episode's owner: the replica the episode state persisted (the state's
// encoder owner, which in this mode names the policy replica holding the
// episode's retained encoder session), else the rendezvous choice. The
// binding is returned so the turn can record where it was decided; the
// episode commit persists that owner. A side call commits nothing, so it
// reads the owner but never moves it. Without replicas it is the cell's one
// client and the binding is nil.
func (armed *raylineARCArmedComponents) policyDeciderFor(
	episodeIDHash string,
	state *raylinearc.EpisodeState,
) (policyDecider, *raylinearc.PolicyServiceBinding) {
	if armed.policyPool == nil {
		return armed.policy, nil
	}
	affinity := raylinearc.EncoderAffinity{}
	if state != nil {
		affinity.Owner = state.EncoderOwner
		affinity.Visited = state.EncoderVisitedOwners
	}
	binding := armed.policyPool.Bind(episodeIDHash, affinity)
	return binding, binding
}

// recordPolicyReplica puts the replica this turn was decided on into its
// trace: the selection log's replica index, attempts and failover, and the
// owner and visited set the episode commit persists (never logged). A turn
// decided without replicas is unchanged.
func recordPolicyReplica(trace *selection.RaylineARCTrace, binding *raylinearc.PolicyServiceBinding) {
	if trace == nil || binding == nil {
		return
	}
	var routed raylinearc.EncoderResult
	binding.Annotate(&routed)
	if routed.ReplicaID == "" {
		return
	}
	trace.EncoderReplicaID = routed.ReplicaID
	trace.EncoderReplicaIndex = routed.ReplicaIndex
	trace.EncoderAttempts = routed.ReplicaAttempts
	trace.EncoderFailover = routed.ReplicaFailover
	trace.EncoderVisitedReplicaIDs = routed.VisitedReplicaIDs
}
