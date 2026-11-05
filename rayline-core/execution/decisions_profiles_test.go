package execution

import (
	"encoding/json"
	"strings"
	"testing"
)

// A Routes API V1 price-profile answer for the configured local action: the
// profile prices the action's model, the gateway resolves it from the profile
// and fills a second candidate from the catalog.
func profileRoutingEcho(t *testing.T, model string) (map[string]any, map[string]any, string) {
	t.Helper()
	rows := map[string]map[string]any{
		model:                {"input": 0.2, "cache_read": 0.02, "cache_write": 0.25, "output": 0.8, "source": sources("profile")},
		"anthropic/claude-5": {"input": 3, "cache_read": 0.3, "cache_write": 3.75, "output": 15, "source": sources("catalog")},
	}
	prices := v1PricesEcho(t, rows, []string{})
	profile := map[string]any{"status": "applied", "id": "local-qwen", "version": 2, "content_hash": "sha256:" + strings.Repeat("e", 64), "schema": "price-profile-rules.v1",
		"prices": map[string]any{model: map[string]any{"input": 0.2, "cache_read": 0.02, "cache_write": 0.25, "output": 0.8}}, "unpriced": []string{}, "ineligible": []string{}, "not_offered": []string{}, "card_version": 7}
	return profile, prices, prices["price_sha256"].(string)
}

func TestCurrentProfileRequiresResolvedIdentityAndActualPriceApplication(t *testing.T) {
	for _, scenario := range []string{
		"latest", "pinned", "optional digest pin", "missing profile", "unsupported profile",
		"wrong profile", "wrong version", "invalid content hash", "unknown schema", "missing prices",
		"unsupported prices", "invalid price digest", "wrong action price digest",
		"wrong configured digest", "wrong basis", "missing action prices", "incomplete row",
		"unknown leg source", "missing unused",
	} {
		t.Run(scenario, func(t *testing.T) {
			config := currentTestConfig(t)
			config.Decisions.Routing = json.RawMessage(`{"price_profile":{"id":"local-qwen","version":2}}`)
			action := config.Actions[0]
			profile, prices, priceHash := profileRoutingEcho(t, config.Deployments[0].RegistryModel)
			actionPrices := map[string]any{"basis": "profile", "price_sha256": priceHash}
			identity := map[string]any{"package_sha256": config.PackageSHA256, "thinking_controls_sha256": config.RegistrySHA256, "action_id": action.ID, "policy_action_id": action.ID, "control_id": action.ControlID, "prices": actionPrices}
			routing := map[string]any{"price_profile": profile, "prices": prices, "unapplied_keys": []string{}}
			// The route names the catalog id the table is keyed by, as the gateway does.
			response := map[string]any{"route_id": "route-profile", "model": config.Deployments[0].RegistryModel, "worker": action.Worker, "policy_action_id": action.ID, "identity": identity, "routing": routing}
			row := prices["models"].(map[string]any)[config.Deployments[0].RegistryModel].(map[string]any)
			switch scenario {
			case "latest":
				config.Decisions.Routing = json.RawMessage(`{"price_profile":{"id":"local-qwen"}}`)
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
			case "unknown schema":
				profile["schema"] = "price-profile-rules.v2"
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
			case "wrong basis":
				actionPrices["basis"] = "caller"
			case "missing action prices":
				delete(identity, "prices")
			case "incomplete row":
				delete(row, "cache_write")
			case "unknown leg source":
				row["source"].(map[string]any)["input"] = "discount"
			case "missing unused":
				delete(prices, "unused")
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
	for _, test := range []struct{ routing, priceSHA string }{
		{`{"price_profile":{}}`, ""}, {`{"price_profile":null}`, ""}, {`{"price_profile":5}`, ""},
		{`{"price_profile":{"id":""}}`, ""}, {`{"price_profile":{"id":" local-qwen"}}`, ""},
		{`{"price_profile":{"version":2}}`, ""},
		{`{"price_profile":{"id":"local-qwen","version":0}}`, ""},
		{`{"price_profile":{"id":"local-qwen","version":null}}`, ""},
		{`{"price_profile":{"id":"local-qwen","version":1.5}}`, ""},
		{`{"price_profile":{"id":"local-qwen","version":9007199254740992}}`, ""},
		{`{"price_profile":{"id":"local-qwen","content_hash":"sha256:abc"}}`, ""},
		// The previous contract's members are refused, never translated.
		{`{"price_profile_id":"local-qwen"}`, ""}, {`{"price_profile_version":2}`, ""},
		{`{"exclusive":false}`, ""}, {`{"exclusive":true}`, ""},
		{`{"prices":{}}`, ""}, {`{"prices":{"m/a":{}}}`, ""}, {`{"prices":{"m/a":{"input":-1}}}`, ""},
		{`{"prices":{"m/a":{"input":null}}}`, ""}, {`{"prices":{"m/a":{"input":1,"output":null}}}`, ""},
		{`{"prices":{"m/a":{"input_per_mtok":1}}}`, ""}, {`{"prices":{"m/a":{"input":"1"}}}`, ""},
		{`{"models":"anthropic/*"}`, ""}, {`{"models":[""]}`, ""}, {`{"models":["a/b","a/b"]}`, ""},
		{`{"excluded_models":[1]}`, ""},
		// A digest pin needs a priced world to pin.
		{`{}`, strings.Repeat("d", 64)}, {`{"models":["a/*"]}`, strings.Repeat("d", 64)},
		{`{"prices":{"m/a":{"input":1}}}`, "not-a-digest"},
	} {
		t.Run(test.routing, func(t *testing.T) {
			config := currentTestConfig(t)
			config.Decisions.URL = "http://127.0.0.1:1/v1/routes"
			config.Decisions.Routing = json.RawMessage(test.routing)
			config.Decisions.PriceSHA256 = test.priceSHA
			if _, err := NewHTTPDecisions(config); err == nil {
				t.Fatal("invalid or ambiguous profile configuration accepted")
			}
		})
	}
}
