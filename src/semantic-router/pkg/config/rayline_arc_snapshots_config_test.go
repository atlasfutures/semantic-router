package config

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const (
	snapshotTestFirst  = "arc-1.0-20261022"
	snapshotTestSecond = "arc-1.0-20261105"
)

// snapshotTestTable is a two-snapshot table over the dispatch fixture: the
// default is the block's own package, and the second another package with
// the same catalog.
func snapshotTestTable(policy *RaylineARCPolicyServiceConfig) *RaylineARCSnapshotsConfig {
	swift, later := 10.0, 9.5
	return &RaylineARCSnapshotsConfig{
		Default: snapshotTestFirst,
		Entries: map[string]RaylineARCSnapshotEntry{
			snapshotTestFirst: {
				Release: "rayline-ai/arc-1.0@20261022", PackageAlias: policy.PackageAlias,
				PackageSHA256: policy.PackageSHA256, CostWeight: &swift,
			},
			snapshotTestSecond: {
				Release: "rayline-ai/arc-1.0@20261105", PackageAlias: "rayline/arc-later",
				PackageSHA256: strings.Repeat("b", 64), CostWeight: &later,
			},
		},
	}
}

func TestRaylineARCSnapshotsParseAndValidate(t *testing.T) {
	var policy RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte(`
snapshots:
  default: arc-1.0-20261022
  entries:
    arc-1.0-20261022:
      release: rayline-ai/arc-1.0@20261022
      package_alias: rayline/arc-073b-speed-pf099
      package_sha256: 1a03f5079ac2e64e2200e788edeb0a8314c2a72941b37ec13ecd784b5f1a759b
      cost_weight: 0.2222222222222222
`), &policy); err != nil {
		t.Fatal(err)
	}
	entry := policy.Snapshots.Entries[snapshotTestFirst]
	if policy.Snapshots.Default != snapshotTestFirst || entry.Release != "rayline-ai/arc-1.0@20261022" ||
		entry.PackageAlias != "rayline/arc-073b-speed-pf099" || entry.CostWeight == nil || *entry.CostWeight != 0.2222222222222222 {
		t.Fatalf("parsed %+v", policy.Snapshots)
	}

	cfg, decision := policyDispatchFixture()
	block := decision.Algorithm.RaylineARC.PolicyService
	if block.SnapshotIDs() != nil {
		t.Fatal("a cell without a table lists snapshots")
	}
	if _, ok := block.ResolveSnapshot(""); ok {
		t.Fatal("a cell without a table resolved a snapshot")
	}
	block.Snapshots = snapshotTestTable(block)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid table refused: %v", err)
	}
	if got := block.SnapshotIDs(); len(got) != 2 || got[0] != snapshotTestFirst || got[1] != snapshotTestSecond {
		t.Fatalf("SnapshotIDs = %v", got)
	}
	for requested, want := range map[string]string{"": snapshotTestFirst, snapshotTestSecond: snapshotTestSecond} {
		if got, ok := block.ResolveSnapshot(requested); !ok || got != want {
			t.Fatalf("ResolveSnapshot(%q) = %q, %v; want %q", requested, got, ok, want)
		}
	}
	if _, ok := block.ResolveSnapshot("arc-1.0-20991231"); ok {
		t.Fatal("an unknown snapshot resolved")
	}

	// Each snapshot's view has its own package and V, the block's bindings,
	// and no table; the block keeps its own.
	second, entry, ok := block.SnapshotPolicy(snapshotTestSecond)
	if !ok || entry.Release != "rayline-ai/arc-1.0@20261105" || second.PackageAlias != "rayline/arc-later" ||
		second.PackageSHA256 != strings.Repeat("b", 64) || second.Snapshots != nil ||
		second.DecisionsV2 == nil || *second.DecisionsV2.CostWeight != 9.5 || len(second.Bindings) != len(block.Bindings) {
		t.Fatalf("SnapshotPolicy = %+v, %+v, %v", second, entry, ok)
	}
	if again, _, _ := block.SnapshotPolicy(snapshotTestSecond); again != second {
		t.Fatal("a snapshot's view was built twice")
	}
	if block.DecisionsV2 != nil || block.PackageAlias != "rayline/test" {
		t.Fatalf("the block changed: %+v", block)
	}
}

func TestRaylineARCSnapshotsRefusals(t *testing.T) {
	weight := func(value float64) *float64 { return &value }
	for name, tc := range map[string]struct {
		edit func(*RaylineARCPolicyServiceConfig, *RaylineARCSnapshotsConfig)
		want string
	}{
		"empty table": {func(_ *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			table.Entries = map[string]RaylineARCSnapshotEntry{}
		}, "entries"},
		"default not an entry": {func(_ *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			table.Default = "arc-1.0-20991231"
		}, "default"},
		"default of another package": {func(_ *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			table.Default = snapshotTestSecond
		}, "default snapshot's package"},
		"default restates bindings": {func(policy *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			entry := table.Entries[snapshotTestFirst]
			entry.Bindings = policy.Bindings
			table.Entries[snapshotTestFirst] = entry
		}, "default snapshot is the policy_service block's"},
		"uppercase id": {func(_ *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			table.Entries["ARC-1.0"] = table.Entries[snapshotTestSecond]
		}, "id"},
		"release without revision": {func(_ *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			entry := table.Entries[snapshotTestSecond]
			entry.Release = "rayline-ai/arc-1.0"
			table.Entries[snapshotTestSecond] = entry
		}, "release"},
		"negative V": {func(_ *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			entry := table.Entries[snapshotTestSecond]
			entry.CostWeight = weight(-1)
			table.Entries[snapshotTestSecond] = entry
		}, "cost_weight"},
		"bad package sha": {func(_ *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			entry := table.Entries[snapshotTestSecond]
			entry.PackageSHA256 = "B"
			table.Entries[snapshotTestSecond] = entry
		}, "package_sha256"},
		"binding off the modelRefs": {func(policy *RaylineARCPolicyServiceConfig, table *RaylineARCSnapshotsConfig) {
			entry := table.Entries[snapshotTestSecond]
			entry.Bindings = append([]RaylineARCPolicyBinding(nil), policy.Bindings...)
			entry.Bindings[2].Worker = "arm-gone"
			table.Entries[snapshotTestSecond] = entry
		}, "not a modelRef"},
		"another package keeps the block's exploration": {func(policy *RaylineARCPolicyServiceConfig, _ *RaylineARCSnapshotsConfig) {
			tau, tauMax, r := 0.01, 0.05, 0.05
			policy.Exploration = &RaylineARCExplorationConfig{
				Tau: &tau, TauMax: &tauMax, Salt: "s",
				Basis: RaylineARCExplorationBasis{R: &r, PackageSHA256: policy.PackageSHA256, ReadoutPin: "pin"},
			}
		}, "exploration is calibrated per package"},
	} {
		cfg, decision := policyDispatchFixture()
		policy := decision.Algorithm.RaylineARC.PolicyService
		policy.Snapshots = snapshotTestTable(policy)
		tc.edit(policy, policy.Snapshots)
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one naming %q", name, err, tc.want)
		}
	}
}

// A v5 cell's snapshot of another package needs its own manifest, whose
// sha256 is that package's; with it, the snapshot's view reads its actions
// from that manifest.
func TestRaylineARCSnapshotsOfAnotherV5Package(t *testing.T) {
	manifest := readPolicyV5Fixture(t)
	cfg, decision := policyV5Decision(t, manifest)
	policy := decision.Algorithm.RaylineARC.PolicyService
	later := append(append([]byte(nil), manifest...), '\n')
	path := filepath.Join(t.TempDir(), "later.json")
	if err := os.WriteFile(path, later, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(later)
	laterSHA := hex.EncodeToString(sum[:])
	policy.Snapshots = &RaylineARCSnapshotsConfig{
		Default: snapshotTestFirst,
		Entries: map[string]RaylineARCSnapshotEntry{
			snapshotTestFirst: {Release: "rayline-ai/arc-1.0@20261022", PackageAlias: policy.PackageAlias, PackageSHA256: policy.PackageSHA256},
			snapshotTestSecond: {
				Release: "rayline-ai/arc-1.0@20261105", PackageAlias: "rayline/test-v5-later", PackageSHA256: laterSHA,
			},
		},
	}
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "needs its own package_manifest") {
		t.Fatalf("a v5 snapshot without its manifest: err = %v", err)
	}

	cfg, decision = policyV5Decision(t, manifest)
	policy = decision.Algorithm.RaylineARC.PolicyService
	policy.Snapshots = &RaylineARCSnapshotsConfig{
		Default: snapshotTestFirst,
		Entries: map[string]RaylineARCSnapshotEntry{
			snapshotTestFirst: {Release: "rayline-ai/arc-1.0@20261022", PackageAlias: policy.PackageAlias, PackageSHA256: policy.PackageSHA256},
			snapshotTestSecond: {
				Release: "rayline-ai/arc-1.0@20261105", PackageAlias: "rayline/test-v5-later", PackageSHA256: laterSHA,
				PackageManifest: path,
			},
		},
	}
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a v5 snapshot with its manifest refused: %v", err)
	}
	view, _, _ := policy.SnapshotPolicy(snapshotTestSecond)
	if view.PackageManifest != path {
		t.Fatalf("the view's manifest is %q", view.PackageManifest)
	}
	if action, ok := view.PackageV5Action(policyV5GLMUp); !ok || action.Model != "glm-5.3-flash" {
		t.Fatalf("the view's action = %+v, %v", action, ok)
	}

	// A manifest whose sha256 is not the snapshot's package is refused.
	cfg, decision = policyV5Decision(t, manifest)
	policy = decision.Algorithm.RaylineARC.PolicyService
	policy.Snapshots = &RaylineARCSnapshotsConfig{
		Default: snapshotTestFirst,
		Entries: map[string]RaylineARCSnapshotEntry{
			snapshotTestFirst: {Release: "rayline-ai/arc-1.0@20261022", PackageAlias: policy.PackageAlias, PackageSHA256: policy.PackageSHA256},
			snapshotTestSecond: {
				Release: "rayline-ai/arc-1.0@20261105", PackageAlias: "rayline/test-v5-later", PackageSHA256: strings.Repeat("c", 64),
				PackageManifest: path,
			},
		},
	}
	if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), "not package_sha256") {
		t.Fatalf("a manifest of another package: err = %v", err)
	}
}
