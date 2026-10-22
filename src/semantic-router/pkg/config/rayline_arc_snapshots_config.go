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

package config

import (
	"fmt"
	"regexp"
	"sort"
	"sync"
)

// RaylineARCSnapshotsConfig serves several dated ARC snapshots from one cell
// (pathfinder docs/plans/2026-10-22-arc-dated-snapshots.md). A snapshot is
// one immutable policy package, published as one release, at this cell's
// tier cost weight. The gateway pins a request to a snapshot with the
// x-rayline-arc-snapshot header; a request without it is served the default
// snapshot, and one naming a snapshot the table does not hold is refused.
// Every snapshot is decided by the same policy service, which loads each
// package by its alias and sha256.
//
// Absent, the cell serves its one package exactly as before: the header is
// not read, and no snapshot headers are sent.
type RaylineARCSnapshotsConfig struct {
	// Default is the snapshot a request without the header is served. Its
	// package must be the policy_service block's own package_alias and
	// package_sha256, so the block's fields describe it.
	Default string `yaml:"default"`
	// Entries are the snapshots this cell serves, keyed by snapshot id
	// (e.g. arc-1.0-20261022).
	Entries map[string]RaylineARCSnapshotEntry `yaml:"entries"`
}

// RaylineARCSnapshotEntry is one snapshot: its release and package, and what
// the cell sends with it. A member left empty is the policy_service block's
// own, which is right whenever the snapshot's package is the block's
// package; a snapshot of another package states what differs.
type RaylineARCSnapshotEntry struct {
	// Release names the published release, e.g. rayline-ai/arc-1.0@20261022.
	// It is returned as x-vsr-arc-release.
	Release       string `yaml:"release"`
	PackageAlias  string `yaml:"package_alias"`
	PackageSHA256 string `yaml:"package_sha256"`
	// PackageManifest is this package's manifest; required for a v5 cell
	// when the package is not the block's.
	PackageManifest string `yaml:"package_manifest,omitempty"`
	// CostWeight is this cell's tier cost weight under this snapshot, sent as
	// selection.operating_point.cost_weight in place of decisions_v2's. A
	// tier's V is re-solved per release, so each snapshot states its own.
	// Absent, decisions_v2's applies (none on a configurable cell).
	CostWeight *float64 `yaml:"cost_weight,omitempty"`
	// Exploration replaces the block's exploration for this snapshot: tau is
	// calibrated per package, so a snapshot of another package on an
	// exploring cell states its own.
	Exploration *RaylineARCExplorationConfig `yaml:"exploration,omitempty"`
	// Bindings and TrainedModels replace the block's for this snapshot.
	// Action ids are content-addressed, so a package with the block's
	// catalog keeps its bindings; a package that adds or drops an action
	// states its own.
	Bindings      []RaylineARCPolicyBinding `yaml:"bindings,omitempty"`
	TrainedModels map[string]string         `yaml:"trained_models,omitempty"`
}

const (
	// maxRaylineARCSnapshots bounds the table. Readiness loads only the
	// default snapshot (the others are verified on first pin), so the bound
	// is not the store app's package budget; it keeps a 180-day retention at
	// a snapshot every few days within one table.
	maxRaylineARCSnapshots = 64
	// RaylineARCSnapshotHeader is the request header that pins a snapshot.
	RaylineARCSnapshotHeader = "x-rayline-arc-snapshot"
)

var (
	// raylineARCSnapshotIDPattern is a dated snapshot id: lowercase, no
	// spaces, at most 64 bytes (it is returned as a header and logged).
	raylineARCSnapshotIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)
	// raylineARCReleasePattern is an HF release name, repo@revision.
	raylineARCReleasePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}@[A-Za-z0-9._-]{1,64}$`)
)

// SnapshotIDs are the configured snapshot ids, sorted; nil without a table.
func (cfg *RaylineARCPolicyServiceConfig) SnapshotIDs() []string {
	if cfg == nil || cfg.Snapshots == nil {
		return nil
	}
	ids := make([]string, 0, len(cfg.Snapshots.Entries))
	for id := range cfg.Snapshots.Entries {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// ResolveSnapshot is the snapshot a request's header value names: the
// default when the value is empty. ok is false for an id the table does not
// hold, and for any value on a cell without a table.
func (cfg *RaylineARCPolicyServiceConfig) ResolveSnapshot(requested string) (string, bool) {
	if cfg == nil || cfg.Snapshots == nil {
		return "", false
	}
	if requested == "" {
		requested = cfg.Snapshots.Default
	}
	_, ok := cfg.Snapshots.Entries[requested]
	return requested, ok
}

// SnapshotPolicy is the policy_service block as it serves one snapshot: the
// block with the entry's package, manifest, cost weight, exploration and
// bindings in place of its own, and no snapshot table. Every request member
// the cell sends for this snapshot is read from it.
//
// The view is built once per snapshot and kept, so a snapshot's manifest is
// read and verified once, as the block's is; the block is not changed after
// load.
func (cfg *RaylineARCPolicyServiceConfig) SnapshotPolicy(id string) (*RaylineARCPolicyServiceConfig, RaylineARCSnapshotEntry, bool) {
	if cfg == nil || cfg.Snapshots == nil {
		return nil, RaylineARCSnapshotEntry{}, false
	}
	entry, ok := cfg.Snapshots.Entries[id]
	if !ok {
		return nil, RaylineARCSnapshotEntry{}, false
	}
	raylineARCSnapshotPolicyLock.Lock()
	defer raylineARCSnapshotPolicyLock.Unlock()
	if derived, built := cfg.snapshotPolicies[id]; built {
		return derived, entry, true
	}
	derived := cfg.snapshotPolicy(entry)
	if cfg.snapshotPolicies == nil {
		cfg.snapshotPolicies = map[string]*RaylineARCPolicyServiceConfig{}
	}
	cfg.snapshotPolicies[id] = derived
	return derived, entry, true
}

// raylineARCSnapshotPolicyLock guards every block's built snapshot views.
var raylineARCSnapshotPolicyLock sync.Mutex

func (cfg *RaylineARCPolicyServiceConfig) snapshotPolicy(entry RaylineARCSnapshotEntry) *RaylineARCPolicyServiceConfig {
	derived := *cfg
	derived.Snapshots, derived.snapshotPolicies = nil, nil
	derived.PackageAlias, derived.PackageSHA256 = entry.PackageAlias, entry.PackageSHA256
	if entry.PackageManifest != "" && entry.PackageManifest != cfg.PackageManifest {
		derived.PackageManifest = entry.PackageManifest
		derived.packageV5 = nil
	}
	if entry.PackageSHA256 != cfg.PackageSHA256 && entry.PackageManifest == "" {
		// Validation refuses a v5 cell's snapshot of another package without
		// its own manifest; never reuse the block's loaded one for it.
		derived.packageV5 = nil
	}
	if entry.CostWeight != nil {
		v2 := RaylineARCDecisionsV2Config{}
		if cfg.DecisionsV2 != nil {
			v2 = *cfg.DecisionsV2
		}
		weight := *entry.CostWeight
		v2.CostWeight = &weight
		derived.DecisionsV2 = &v2
	}
	if entry.Exploration != nil {
		derived.Exploration = entry.Exploration
	}
	if len(entry.Bindings) > 0 {
		derived.Bindings = entry.Bindings
	}
	if len(entry.TrainedModels) > 0 {
		derived.TrainedModels = entry.TrainedModels
	}
	return &derived
}

// validateRaylineARCSnapshotsConfig checks the table's shape and each
// snapshot's policy as the block's own is checked.
func validateRaylineARCSnapshotsConfig(cfg *RaylineARCPolicyServiceConfig) error {
	table := cfg.Snapshots
	if table == nil {
		return nil
	}
	if len(table.Entries) == 0 || len(table.Entries) > maxRaylineARCSnapshots {
		return fmt.Errorf("snapshots: entries must hold 1 to %d snapshots", maxRaylineARCSnapshots)
	}
	defaultEntry, ok := table.Entries[table.Default]
	if !ok {
		return fmt.Errorf("snapshots: default %q is not an entry", table.Default)
	}
	if defaultEntry.PackageAlias != cfg.PackageAlias || defaultEntry.PackageSHA256 != cfg.PackageSHA256 {
		return fmt.Errorf("snapshots: the default snapshot's package must be package_alias and package_sha256")
	}
	if (defaultEntry.PackageManifest != "" && defaultEntry.PackageManifest != cfg.PackageManifest) ||
		len(defaultEntry.Bindings) > 0 || len(defaultEntry.TrainedModels) > 0 || defaultEntry.Exploration != nil {
		return fmt.Errorf("snapshots: the default snapshot is the policy_service block's package; state its manifest, bindings, trained_models and exploration there")
	}
	for _, id := range cfg.SnapshotIDs() {
		entry := table.Entries[id]
		if !raylineARCSnapshotIDPattern.MatchString(id) {
			return fmt.Errorf("snapshots: id %q must match %s", id, raylineARCSnapshotIDPattern)
		}
		if !raylineARCReleasePattern.MatchString(entry.Release) {
			return fmt.Errorf("snapshots %s: release must be repo@revision", id)
		}
		if entry.CostWeight != nil && !validRaylineARCCostWeight(*entry.CostWeight) {
			return fmt.Errorf("snapshots %s: cost_weight must be a finite number >= 0", id)
		}
		samePackage := entry.PackageSHA256 == cfg.PackageSHA256
		if !samePackage && cfg.IsPackageV5() && entry.PackageManifest == "" {
			return fmt.Errorf("snapshots %s: a package other than package_sha256 needs its own package_manifest", id)
		}
		if !samePackage && entry.PackageManifest == cfg.PackageManifest && entry.PackageManifest != "" {
			return fmt.Errorf("snapshots %s: package_manifest is the block's, whose sha256 is package_sha256", id)
		}
		if !samePackage && cfg.Exploration != nil && entry.Exploration == nil {
			return fmt.Errorf("snapshots %s: exploration is calibrated per package, so a snapshot of another package states its own", id)
		}
		derived, _, _ := cfg.SnapshotPolicy(id)
		if err := validateRaylineARCPolicyServiceConfig(derived); err != nil {
			return fmt.Errorf("snapshots %s: %w", id, err)
		}
	}
	return nil
}

// validateRaylineARCSnapshotDecisions checks each snapshot's bindings against
// the decision, as the block's are: every binding names a modelRef and every
// action is dispatchable. A snapshot need not serve every modelRef: the
// default does, and an older or newer package may serve fewer models.
func validateRaylineARCSnapshotDecisions(cfg *RouterConfig, decision Decision) error {
	policy := decision.Algorithm.RaylineARC.PolicyService
	for _, id := range policy.SnapshotIDs() {
		derived, _, _ := policy.SnapshotPolicy(id)
		refs := make(map[string]bool, len(decision.ModelRefs))
		for _, modelRef := range decision.ModelRefs {
			refs[modelRef.Model] = true
		}
		for _, binding := range derived.Bindings {
			if !refs[binding.Worker] {
				return fmt.Errorf("snapshots %s: binding names worker %q, which is not a modelRef", id, binding.Worker)
			}
		}
		arc := *decision.Algorithm.RaylineARC
		arc.PolicyService = derived
		algorithm := *decision.Algorithm
		algorithm.RaylineARC = &arc
		snapshotDecision := decision
		snapshotDecision.Algorithm = &algorithm
		if derived.FallbackEnabled() {
			if err := derived.fallbackWorkerModels(); err != nil {
				return fmt.Errorf("snapshots %s: %w", id, err)
			}
		}
		if err := validateRaylineARCPolicyDispatch(cfg, snapshotDecision); err != nil {
			return fmt.Errorf("snapshots %s: %w", id, err)
		}
	}
	return nil
}
