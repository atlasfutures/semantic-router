package execution

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCurrentProfileRequiresResolvedIdentityAndActualPriceApplication(t *testing.T) {
	for _, scenario := range []string{
		"latest", "pinned", "optional digest pin", "missing profile", "unsupported profile",
		"wrong profile", "wrong version", "invalid content hash", "missing prices",
		"unsupported prices", "invalid price digest", "wrong action price digest",
		"wrong configured digest", "wrong exclusive", "missing action prices",
	} {
		t.Run(scenario, func(t *testing.T) {
			config := currentTestConfig(t)
			config.Decisions.Routing = json.RawMessage(`{"price_profile_id":"local-qwen","price_profile_version":2}`)
			priceHash := strings.Repeat("d", 64)
			profile := map[string]any{"status": "applied", "id": "local-qwen", "version": 2, "content_hash": "sha256:" + strings.Repeat("e", 64)}
			prices := map[string]any{"status": "applied", "price_sha256": priceHash, "exclusive": false}
			actionPrices := map[string]any{"basis": "caller", "price_sha256": priceHash, "exclusive": false}
			action := config.Actions[0]
			identity := map[string]any{"package_sha256": config.PackageSHA256, "thinking_controls_sha256": config.RegistrySHA256, "action_id": action.ID, "policy_action_id": action.ID, "control_id": action.ControlID, "prices": actionPrices}
			routing := map[string]any{"price_profile": profile, "prices": prices, "unapplied_keys": []string{}}
			response := map[string]any{"route_id": "route-profile", "model": action.Model, "worker": action.Worker, "policy_action_id": action.ID, "identity": identity, "routing": routing}
			switch scenario {
			case "latest":
				config.Decisions.Routing = json.RawMessage(`{"price_profile_id":"local-qwen"}`)
			case "optional digest pin":
				config.Decisions.PriceSHA256 = priceHash
			case "missing profile":
				delete(routing, "price_profile")
			case "unsupported profile":
				profile["status"] = "unsupported"
			case "wrong profile":
				profile["id"] = "other-profile"
			case "wrong version":
				profile["version"] = 3
			case "invalid content hash":
				profile["content_hash"] = priceHash
			case "missing prices":
				delete(routing, "prices")
			case "unsupported prices":
				prices["status"] = "unsupported"
			case "invalid price digest":
				prices["price_sha256"] = "invalid"
				actionPrices["price_sha256"] = "invalid"
			case "wrong action price digest":
				actionPrices["price_sha256"] = strings.Repeat("a", 64)
			case "wrong configured digest":
				config.Decisions.PriceSHA256 = strings.Repeat("a", 64)
			case "wrong exclusive":
				prices["exclusive"] = true
				actionPrices["exclusive"] = true
			case "missing action prices":
				delete(identity, "prices")
			}
			decision, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput())
			valid := scenario == "latest" || scenario == "pinned" || scenario == "optional digest pin"
			if !valid {
				if err == nil {
					t.Fatal("unapplied or mismatched profile accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if decision.PriceProfile == nil || decision.PriceProfile.ID != "local-qwen" || decision.PriceProfile.Version != 2 || decision.PriceProfile.ContentHash != profile["content_hash"] {
				t.Fatalf("resolved profile identity lost: %+v", decision.PriceProfile)
			}
		})
	}
}

func TestCurrentProfileConfigRejectsAmbiguityBeforeLookup(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "test-key")
	for _, routing := range []string{
		`{"price_profile_id":""}`, `{"price_profile_id":null}`, `{"price_profile_id":5}`,
		`{"price_profile_id":" local-qwen"}`, `{"price_profile_version":2}`,
		`{"price_profile_id":"local-qwen","price_profile_version":0}`,
		`{"price_profile_id":"local-qwen","price_profile_version":null}`,
		`{"price_profile_id":"local-qwen","price_profile_version":1.5}`,
		`{"price_profile_id":"local-qwen","price_profile_version":9007199254740992}`,
		`{"price_profile_id":"local-qwen","prices":{}}`,
	} {
		t.Run(routing, func(t *testing.T) {
			config := currentTestConfig(t)
			config.Decisions.URL = "http://127.0.0.1:1/v1/routes"
			config.Decisions.Routing = json.RawMessage(routing)
			config.Decisions.PriceSHA256 = strings.Repeat("d", 64)
			if _, err := NewHTTPDecisions(config); err == nil {
				t.Fatal("invalid or ambiguous profile configuration accepted")
			}
		})
	}
}
