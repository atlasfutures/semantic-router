package extproc

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A worker moved to Vertex AI under the same name and format does not resume
// the placer it had on another endpoint: that placer's ledger can hold
// instructions the Vertex AI cell refuses.
func TestRaylineARCPlacementKeySeparatesVertexAI(t *testing.T) {
	openRouter := &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}
	vertex := &config.ProviderProfile{Type: "vertex-ai", BaseURL: "https://aiplatform.googleapis.com/v1/projects/p/locations/global"}
	if got := raylineARCPlacementKey("w", "chat", openRouter); got != "w|chat" {
		t.Fatalf("an OpenRouter dispatch's key = %q, want the plain worker|format key", got)
	}
	if got := raylineARCPlacementKey("w", "chat", nil); got != "w|chat" {
		t.Fatalf("a dispatch with no profile's key = %q", got)
	}
	if got := raylineARCPlacementKey("w", "chat", vertex); got == "w|chat" {
		t.Fatal("a Vertex AI dispatch resumes the worker's other placer")
	}
}

// Only a named scope is a scope: an older key's worker|format|shape...
// segments are an unscoped placement.
func TestRaylineARCPlacementKeyPartsKeepLegacyKeysUnscoped(t *testing.T) {
	for key, want := range map[string][2]string{
		"opus|messages":             {"opus", ""},
		"opus|messages|high|-|true": {"opus", ""},
		"opus|vertex-ai|chat":       {"opus", "vertex-ai"},
	} {
		worker, scope, ok := raylineARCPlacementKeyParts(key)
		if !ok || worker != want[0] || scope != want[1] {
			t.Fatalf("%q = (%q, %q, %v), want %v", key, worker, scope, ok, want)
		}
	}
}

// A worker whose placements moved to a scope drops those it left elsewhere,
// so they do not push other workers' placements out of the bounded record.
func TestRaylineARCRetirePlacementsDropsOnlyTheWorkersOtherScopes(t *testing.T) {
	placements := []raylinearc.ControlPlacement{
		{Key: "w|messages"}, {Key: "w|chat|high|-|true"}, {Key: "w|vertex-ai|chat"}, {Key: "x|messages"}, {Key: "wx|chat"},
	}
	var keys []string
	for _, placement := range raylineARCRetirePlacements(placements, "w", "vertex-ai") {
		keys = append(keys, placement.Key)
	}
	if want := []string{"w|vertex-ai|chat", "x|messages", "wx|chat"}; !slices.Equal(keys, want) {
		t.Fatalf("kept %v, want %v", keys, want)
	}
	if len(placements) != 5 {
		t.Fatal("the input was changed")
	}
}
