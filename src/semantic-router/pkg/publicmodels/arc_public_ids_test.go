package publicmodels

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The ARC public ids (ARC launch, 2026-10-21): first-party ids are bare --
// arc-swift, arc-sage and the configurable arc-1.0 -- and the prefixed
// rayline/arc-* spellings stay accepted aliases. A cell serves one of them at
// its own operating point, so its model surface is that product's ids and
// nothing else: auto_model_names is the closed allow-list a turn must name.
var arcTierCellIDs = map[string][]string{
	"swift": {"arc-swift", "rayline/arc-swift"},
	"sage":  {"arc-sage", "rayline/arc-sage"},
	"1.0":   {"arc-1.0", "rayline/arc-1.0", "arc-1", "rayline/arc-1"},
}

// arcTierCellConfig is the candidates cell with the tier's ids, plus the
// internal decision alias the gateway dispatches every ARC turn under today.
func arcTierCellConfig(t *testing.T, ids []string) *config.RouterConfig {
	t.Helper()
	t.Setenv("PUBLICMODELS_CANDIDATES_TEST_KEY", "public-candidates-test-key")
	names := append([]string{"rayline/arc-dev"}, ids...)
	quoted := make([]string, 0, len(names))
	for _, name := range names {
		quoted = append(quoted, `"`+name+`"`)
	}
	yaml := strings.Replace(
		candidatesConfig,
		`auto_model_names: ["rayline/arc-dev"]`,
		`auto_model_names: [`+strings.Join(quoted, ", ")+`]`,
		1,
	)
	cfg, err := config.ParseYAMLBytes([]byte(yaml))
	if err != nil {
		t.Fatalf("the %v cell config did not load: %v", ids, err)
	}
	return cfg
}

// A tier cell admits its bare id and its prefixed alias alike, lists both
// with the same ARC candidates, and refuses every other tier's ids: a tier
// turn is never answered by another tier's cell.
func TestARCTierCellAdmitsItsBareAndPrefixedIDsOnly(t *testing.T) {
	for tier, ids := range arcTierCellIDs {
		t.Run(tier, func(t *testing.T) {
			cfg := arcTierCellConfig(t, ids)
			listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
			want := len(candidatesOf(t, listing, "rayline/arc-dev"))
			for _, id := range ids {
				if !cfg.IsAutoModelName(id) {
					t.Fatalf("%q is not admitted by the %s cell", id, tier)
				}
				if got := len(candidatesOf(t, listing, id)); got != want || got == 0 {
					t.Fatalf("%q lists %d candidates, want the decision's %d", id, got, want)
				}
			}
			for other, otherIDs := range arcTierCellIDs {
				if other == tier {
					continue
				}
				for _, id := range otherIDs {
					if cfg.IsAutoModelName(id) {
						t.Fatalf("the %s cell admits %q, the %s tier's id", tier, id, other)
					}
				}
			}
			if cfg.IsAutoModelName("arc") {
				t.Fatalf("the %s cell admits the bare word arc, which names no product", tier)
			}
		})
	}
}
