package raylinearc

import (
	"strings"
	"testing"
)

// pathfinder#3677's listing names a package's fallback beside it: the fixture
// holds one entry with the pair and one without, and both decode.
func TestDecodePolicyPackagesResponseReadsTheFallbackPair(t *testing.T) {
	response, err := DecodePolicyPackagesResponse(readPolicyFixture(t, "packages_response.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Packages) != 2 {
		t.Fatalf("%d packages, want 2", len(response.Packages))
	}
	with, without := response.Packages[0], response.Packages[1]
	if with.FallbackActionID != "569bf3701d341b1945954bf38e0686b23e404ab2a857983402706a40b2a0fad9" ||
		with.FallbackModel != "anthropic/claude-opus-5" {
		t.Fatalf("listed fallback = %q %q", with.FallbackActionID, with.FallbackModel)
	}
	if without.FallbackActionID != "" || without.FallbackModel != "" {
		t.Fatalf("an entry without the pair decoded %q %q", without.FallbackActionID, without.FallbackModel)
	}
}

// The pair is stated whole or not at all, never null, with a sha256 action
// id and a model.
func TestDecodePolicyPackagesResponseRefusesABrokenFallbackPair(t *testing.T) {
	for name, edit := range map[string]func(entry map[string]any){
		"action id alone":   func(e map[string]any) { delete(e, "fallback_model") },
		"model alone":       func(e map[string]any) { delete(e, "fallback_action_id") },
		"null action id":    func(e map[string]any) { e["fallback_action_id"] = nil },
		"null model":        func(e map[string]any) { e["fallback_model"] = nil },
		"both null":         func(e map[string]any) { e["fallback_action_id"], e["fallback_model"] = nil, nil },
		"uppercase hex":     func(e map[string]any) { e["fallback_action_id"] = strings.ToUpper(e["fallback_action_id"].(string)) },
		"short action id":   func(e map[string]any) { e["fallback_action_id"] = "569bf3701d34" },
		"empty model":       func(e map[string]any) { e["fallback_model"] = "" },
		"numeric model":     func(e map[string]any) { e["fallback_model"] = 7 },
		"empty action id":   func(e map[string]any) { e["fallback_action_id"] = "" },
		"unknown neighbour": func(e map[string]any) { e["fallback_effort"] = "high" },
	} {
		t.Run(name, func(t *testing.T) {
			body := fixtureWith(t, "packages_response.v1.json", func(body map[string]any) {
				edit(body["packages"].([]any)[0].(map[string]any))
			})
			if _, err := DecodePolicyPackagesResponse(body); err == nil {
				t.Fatal("decoded")
			}
		})
	}
}
