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
// for its default snapshot, the block's own package; every other snapshot is
// verified the same way on the first turn that pins it
// (raylineARCSnapshotReadiness), so readiness does not grow with the table.
func probeRaylineARCPolicyCell(
	ctx context.Context,
	client *raylinearc.PolicyServiceClient,
	policy *config.RaylineARCPolicyServiceConfig,
	scorer *policyServiceScorer,
) error {
	if policy.Snapshots == nil {
		return probeRaylineARCPolicyPackage(ctx, client, policy, scorer)
	}
	return probeRaylineARCSnapshot(ctx, client, policy, scorer, policy.Snapshots.Default)
}

// probeRaylineARCSnapshot verifies one snapshot of a snapshot cell: the
// service loads and serves its package (a store-mode service loads it on
// the listing that names it).
func probeRaylineARCSnapshot(
	ctx context.Context,
	client *raylinearc.PolicyServiceClient,
	policy *config.RaylineARCPolicyServiceConfig,
	scorer *policyServiceScorer,
	id string,
) error {
	served, _, ok := policy.SnapshotPolicy(id)
	snapshotScorer, found := scorer.forSnapshot(id)
	if !ok || !found {
		return &raylinearc.PolicyServiceError{Class: "snapshot_scorer"}
	}
	return probeRaylineARCPolicyPackage(ctx, client, served, snapshotScorer)
}

// probeRaylineARCPolicyPackage verifies one package: the service has loaded
// it and serves what the cell sends, and decides relaxed for a relaxed cell.
func probeRaylineARCPolicyPackage(
	ctx context.Context,
	client *raylinearc.PolicyServiceClient,
	policy *config.RaylineARCPolicyServiceConfig,
	scorer *policyServiceScorer,
) error {
	if err := requireRaylineARCPolicyServing(ctx, client, policy); err != nil {
		return err
	}
	if scorer.episodeMode == raylinearc.PolicyEpisodeModeRelaxed {
		return probeRelaxedPolicyDecide(ctx, client, scorer)
	}
	return nil
}

// lazySnapshotIDs are a snapshot cell's ids but its default: the rows
// verified on first pin.
func lazySnapshotIDs(policy *config.RaylineARCPolicyServiceConfig) []string {
	if policy == nil || policy.Snapshots == nil {
		return nil
	}
	ids := make([]string, 0, len(policy.Snapshots.Entries))
	for _, id := range policy.SnapshotIDs() {
		if id != policy.Snapshots.Default {
			ids = append(ids, id)
		}
	}
	return ids
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
