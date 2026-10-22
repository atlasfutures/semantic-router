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
	"context"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A snapshot cell serves several dated ARC snapshots, each one policy
// package, from one policy service (config.RaylineARCSnapshotsConfig). The
// request's snapshot is resolved before the lease
// (buildRaylineARCSelectionContext), decided by that snapshot's scorer
// (selectViaPolicyService), dispatched from that snapshot's view of the
// policy block, and named in x-vsr-arc-snapshot and x-vsr-arc-release.

// forSnapshot is the scorer that decides a turn served under this snapshot:
// the cell's own when it has no snapshot table.
func (scorer *policyServiceScorer) forSnapshot(snapshot string) (*policyServiceScorer, bool) {
	if scorer.snapshots == nil {
		return scorer, snapshot == ""
	}
	served, ok := scorer.snapshots[snapshot]
	return served, ok
}

// armBound reports whether any of this package's actions dispatch to arm.
func (scorer *policyServiceScorer) armBound(arm int) bool {
	for _, binding := range scorer.bindings {
		if binding.arm == arm {
			return true
		}
	}
	return false
}

// snapshotAttribution keeps the attributions whose action this snapshot's
// package knows. A conversation whose earlier turns another snapshot decided
// carries that package's actions in its ledger; an action id is
// content-addressed, so one this package also has means the same action,
// and the others stay unattributed, as after a prefix break, rather than
// being refused as out of the package's catalog. Trained arm ids are per
// package, and the service refuses one outside the action's: an arm id this
// package's manifest does not list for the action, or any arm id when VSR
// does not know the package's arms (v4), is sent as null, which the service
// reads as the action's first trained arm.
func (scorer *policyServiceScorer) snapshotAttribution(attribution []raylinearc.PolicyAttribution) []raylinearc.PolicyAttribution {
	kept := make([]raylinearc.PolicyAttribution, 0, len(attribution))
	for _, entry := range attribution {
		binding, bound := scorer.bindings[entry.ActionID]
		if !bound {
			continue
		}
		if entry.ArmID != nil && !slices.Contains(binding.trainedArms, *entry.ArmID) {
			entry.ArmID = nil
		}
		kept = append(kept, entry)
	}
	return kept
}

// probeRaylineARCPolicyCell is a policy cell's readiness: the service has
// loaded the cell's package and serves what the cell sends, and a relaxed
// cell's service decides relaxed. A snapshot cell is ready once that holds
// for every snapshot: a store-mode service loads each package on the pin
// that names it, so no pinned turn finds its package unloaded.
func probeRaylineARCPolicyCell(
	ctx context.Context,
	client *raylinearc.PolicyServiceClient,
	policy *config.RaylineARCPolicyServiceConfig,
	routing *config.RaylineARCRoutesRoutingConfig,
	scorer *policyServiceScorer,
) error {
	base := scorer
	probeOne := func(policy *config.RaylineARCPolicyServiceConfig, scorer *policyServiceScorer) error {
		// One service serves every snapshot: its capability list is the
		// cell's, recorded on the snapshot's scorer and the cell's own.
		advertised := func(capabilities []string) {
			scorer.setAdvertised(capabilities)
			if scorer != base {
				base.setAdvertised(capabilities)
			}
		}
		if err := requireRaylineARCPolicyServing(ctx, client, policy, routing, advertised); err != nil {
			return err
		}
		if scorer.episodeMode == raylinearc.PolicyEpisodeModeRelaxed {
			return probeRelaxedPolicyDecide(ctx, client, scorer)
		}
		return nil
	}
	if policy.Snapshots == nil {
		return probeOne(policy, scorer)
	}
	for _, id := range policy.SnapshotIDs() {
		served, _, _ := policy.SnapshotPolicy(id)
		snapshotScorer, ok := scorer.forSnapshot(id)
		if !ok {
			return &raylinearc.PolicyServiceError{Class: "snapshot_scorer"}
		}
		if err := probeOne(served, snapshotScorer); err != nil {
			return err
		}
	}
	return nil
}

// raylineARCServedPolicy is the policy_service block as it served this turn:
// on a snapshot cell, the view of the snapshot that decided it, whose
// package names the turn's action; otherwise the block itself.
func raylineARCServedPolicy(ctx *RequestContext) *config.RaylineARCPolicyServiceConfig {
	policy := ctx.VSRSelectedDecision.Algorithm.RaylineARC.PolicyService
	if ctx.VSRRaylineARC != nil && ctx.VSRRaylineARC.Snapshot != "" {
		if served, _, ok := policy.SnapshotPolicy(ctx.VSRRaylineARC.Snapshot); ok {
			return served
		}
	}
	return policy
}
