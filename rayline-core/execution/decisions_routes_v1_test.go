package execution

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// sources gives every leg the one source.
func sources(source string) map[string]any {
	return map[string]any{"input": source, "cache_read": source, "cache_write": source, "output": source}
}

// v1PricesEcho is a Routes API V1 routing.prices echo over rows, with the
// digest the gateway would compute for that table.
func v1PricesEcho(t *testing.T, rows map[string]map[string]any, unused []string) map[string]any {
	t.Helper()
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	var table map[string]resolvedPrice
	if err := json.Unmarshal(raw, &table); err != nil {
		t.Fatal(err)
	}
	digest, err := routePricesSHA256(table)
	if err != nil {
		t.Fatal(err)
	}
	models := make(map[string]any, len(rows))
	for id, row := range rows {
		models[id] = row
	}
	return map[string]any{"status": "applied", "models": models, "unused": unused, "price_sha256": digest}
}

// The public digest is sha256 over the RFC 8785 form of
// {"schema": "arc-route-prices.v1", "prices": table}. The first vector is the
// router's response_full fixture; the rest exercise RFC 8785 number
// formatting and UTF-16 key order with the cell contract's price vectors.
func TestRoutePricesDigestMatchesContractVectors(t *testing.T) {
	for _, vector := range []struct {
		table  string
		digest string
	}{
		{`{"anthropic/claude-sonnet-5":{"input":3,"cache_read":0.3,"cache_write":3.75,"output":15},"deepseek/deepseek-v4.1-flash":{"input":0.1,"cache_read":0,"cache_write":0,"output":0.4}}`, "24f6b46393842bbe5f4a24fc9ecb7f446f5e27999cb81b4bbde091a776741cf5"},
		{`{"m/a":{"input":1.0,"cache_read":0.0,"cache_write":0.0,"output":2.0}}`, "259d880a2a0014f80e9889ed2f1886580272c8f2def114776dc98d0918514dd7"},
		{`{"m/a":{"input":1e-07,"cache_read":1e+21,"cache_write":123.456,"output":1e-06}}`, "078d38bf1ffb98907fcb40151c7f94ef5d157c7156f2d24c939f02d926c3fbba"},
		{`{"z/é":{"input":0.25,"cache_read":0.025,"cache_write":0.3125,"output":1.25},"a/b":{"input":3,"cache_read":0.3,"cache_write":3.75,"output":15}}`, "6daaad32f5faf696dd5ba0268391a142f2859b8d87f161c8d312a63e98e06476"},
	} {
		var rows map[string]map[string]any
		if err := json.Unmarshal([]byte(vector.table), &rows); err != nil {
			t.Fatal(err)
		}
		for _, row := range rows {
			row["source"] = sources("caller")
		}
		if echo := v1PricesEcho(t, rows, nil); echo["price_sha256"] != vector.digest {
			t.Fatalf("digest %s, want %s", echo["price_sha256"], vector.digest)
		}
	}
	for value, want := range map[float64]string{0: "0", 1: "1", 15: "15", 0.3: "0.3", 3.75: "3.75", 0.025: "0.025", 123.456: "123.456", 1e-7: "1e-7", 0.000001: "0.000001", 1e21: "1e+21", 1e20: "100000000000000000000", 1234.5e-9: "0.0000012345", -2.5: "-2.5"} {
		if got := jcsNumber(value); got != want {
			t.Fatalf("jcsNumber(%v) = %s, want %s", value, got, want)
		}
	}
	// The previous contract's digest (cell contract price_sha256 vectors) is
	// recognised for either exclusivity, and only over a fully priced table.
	fourLegs := callerPrices{"anthropic/claude-sonnet-5": {"input": 3, "cache_read": 0.3, "cache_write": 3.75, "output": 15}, "deepseek/deepseek-v4.1-flash": {"input": 0.1, "cache_read": 0, "cache_write": 0, "output": 0.4}}
	for pin, want := range map[string]bool{"d5c16e9479a1aba03cd9e38b4b3878e21440b5a5d156f46e882d3eec835cf57f": true, "3abe1bbdf0242628534a98f82b917de91f29b93320c876891005afb993530b35": true, "24f6b46393842bbe5f4a24fc9ecb7f446f5e27999cb81b4bbde091a776741cf5": false} {
		if got := isCallerPricesDigest(pin, fourLegs); got != want {
			t.Fatalf("isCallerPricesDigest(%s) = %v", pin, got)
		}
	}
	if !isCallerPricesDigest("62fad7dede48638275ddc4eccab435b9fbe333d554573f165fe9f380141d457c", callerPrices{"m/a": {"input": 1, "cache_read": 0, "cache_write": 0, "output": 2}}) || isCallerPricesDigest("62fad7dede48638275ddc4eccab435b9fbe333d554573f165fe9f380141d457c", callerPrices{"m/a": {"input": 1, "output": 2}}) {
		t.Fatal("previous-contract digest recognition")
	}
	for _, row := range []map[string]resolvedPrice{
		{"m/a": {Input: echoPtr(1.0), CacheRead: echoPtr(0.0), CacheWrite: echoPtr(0.0), Source: sourcesOf("caller")}},
		{"m/a": {Input: echoPtr(-1.0), CacheRead: echoPtr(0.0), CacheWrite: echoPtr(0.0), Output: echoPtr(2.0), Source: sourcesOf("caller")}},
		{"m/a": {Input: echoPtr(1.0), CacheRead: echoPtr(0.0), CacheWrite: echoPtr(0.0), Output: echoPtr(2.0), Source: map[string]string{"input": "caller"}}},
	} {
		if _, err := routePricesSHA256(row); err == nil {
			t.Fatal("digested an incomplete resolved price")
		}
	}
}

func sourcesOf(source string) map[string]string {
	return map[string]string{"input": source, "cache_read": source, "cache_write": source, "output": source}
}

// The request carries the V1 members as configured and never the previous
// contract's exclusive flag or profile keys.
func TestRoutesV1RequestCarriesConfiguredMembersAndNoExclusive(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "test-key")
	config := currentTestConfig(t)
	config.Decisions.Result = "custom"
	config.Decisions.Routing = json.RawMessage(`{"v_value":1.5,"prices":{"qwen/qwen3.8-27b":{"input":0.01,"output":0.01}},"price_profile":{"id":"profile-1","version":3},"models":["qwen/*","anthropic/*"],"excluded_models":["anthropic/claude-opus-5"]}`)
	var requests int
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body struct {
			Model   string                     `json:"model"`
			Routing map[string]json.RawMessage `json:"routing"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		for _, removed := range []string{"exclusive", "price_profile_id", "price_profile_version", "execution"} {
			if _, sent := body.Routing[removed]; sent {
				t.Errorf("request carried routing.%s", removed)
			}
		}
		want := map[string]string{"v_value": `1.5`, "prices": `{"qwen/qwen3.8-27b":{"input":0.01,"output":0.01}}`, "price_profile": `{"id":"profile-1","version":3}`, "models": `["qwen/*","anthropic/*"]`, "excluded_models": `["anthropic/claude-opus-5"]`}
		if len(body.Routing) != len(want) || body.Model != "arc-1.0" {
			t.Errorf("request routing %v", body.Routing)
		}
		for key, value := range want {
			if string(body.Routing[key]) != value {
				t.Errorf("routing.%s = %s, want %s", key, body.Routing[key], value)
			}
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"code":"unknown_model","type":"invalid_request_error"}}`))
	}))
	defer endpoint.Close()
	config.Decisions.URL = endpoint.URL
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := provider.Decide(t.Context(), currentTestInput()); err == nil || requests != 1 {
		t.Fatalf("refusal not surfaced after %d requests: %v", requests, err)
	}
}

// The router's response_full fixture (atlasfutures/semantic-router's hosted
// gateway, Routes API V1 at pathfinder#6159), with V under its current name
// `v_value` (#403) and otherwise verbatim: a custom result
// decided at a caller-priced table with one unused caller price. It carries
// no identity, so the current adapter binds it to the configured manifest.
func routesV1FullResponse(t *testing.T) map[string]any {
	t.Helper()
	raw, err := os.ReadFile("testdata/routes_v1_full.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Body map[string]any `json:"body"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture.Body
}

const routesV1FixtureRouting = `{"v_value":1.5,"models":["anthropic/*","deepseek/*"],"excluded_models":["deepseek/deepseek-v4-pro"],"prices":{"anthropic/claude-sonnet-5":{"input":3,"cache_read":0.3,"cache_write":3.75,"output":15},"deepseek/deepseek-v4.1-flash":{"input":0.1,"cache_read":0,"cache_write":0,"output":0.4},"qwen/qwen3.6-35b-a3b":{"input":0.2,"output":0.8}}}`

func TestRoutesV1FixtureIsAcknowledgedWithoutADigestPin(t *testing.T) {
	const fixtureDigest = "24f6b46393842bbe5f4a24fc9ecb7f446f5e27999cb81b4bbde091a776741cf5"
	for _, test := range []struct {
		name     string
		priceSHA string
		valid    bool
	}{{"no pin", "", true}, {"matching pin", fixtureDigest, true}, {"wrong pin", strings.Repeat("a", 64), false}} {
		t.Run(test.name, func(t *testing.T) {
			response := routesV1FullResponse(t)
			config := currentTestConfig(t)
			config.Decisions.Result = "custom"
			config.Decisions.Routing = json.RawMessage(routesV1FixtureRouting)
			config.Decisions.PriceSHA256 = test.priceSHA
			selected := response["custom"].(map[string]any)
			config.Actions[0].ID = selected["policy_action_id"].(string)
			config.Actions[0].Model = selected["model"].(string)
			config.Actions[0].Worker = selected["worker"].(string)
			decision, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput())
			if !test.valid {
				if err == nil {
					t.Fatal("a price world other than the pinned one was accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if decision.Result != "custom" || decision.Model != "anthropic/claude-sonnet-5" || decision.PriceProfile != nil || decision.AppliedValueUSD == nil || *decision.AppliedValueUSD != 1.5 {
				t.Fatalf("fixture decision %+v", decision)
			}
		})
	}
}

// A hybrid (v2) profile prices only its local model and names no
// candidates, as Raydock does: the gateway fills the cloud candidates from
// the catalog and the identity reports the whole table as caller-based.
func TestV2CallerPriceForLocalModelIsAcknowledgedLegByLeg(t *testing.T) {
	for _, scenario := range []string{"applied", "local leg from catalog", "catalog identity", "unapplied prices"} {
		t.Run(scenario, func(t *testing.T) {
			local, cloud := &upstreamFixture{}, &upstreamFixture{}
			config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
			localModel := config.Deployments[0].RegistryModel
			config.Decisions.Routing = json.RawMessage(`{"policy":{"preset":"` + hybridPreset + `"},"prices":{"` + localModel + `":{"input":0.01,"cache_read":0.01,"cache_write":0.01,"output":0.01}}}`)
			fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
			fixture.mutate = func(response map[string]any) {
				rows := map[string]map[string]any{
					localModel:         {"input": 0.01, "cache_read": 0.01, "cache_write": 0.01, "output": 0.01, "source": sources("caller")},
					"moonshot/kimi-k3": {"input": 0.6, "cache_read": 0.15, "cache_write": 0.6, "output": 2.5, "source": sources("catalog")},
				}
				if scenario == "local leg from catalog" {
					rows[localModel]["source"].(map[string]any)["cache_write"] = "catalog"
				}
				echo := v1PricesEcho(t, rows, []string{})
				if scenario == "unapplied prices" {
					echo["status"] = "unsupported"
					response["routing"].(map[string]any)["unapplied_keys"] = []string{"prices"}
				}
				response["routing"].(map[string]any)["prices"] = echo
				prices := map[string]any{"basis": "caller", "price_sha256": echo["price_sha256"]}
				if scenario == "catalog identity" {
					prices = map[string]any{"basis": "catalog", "price_sha256": nil}
				}
				response["identity"].(map[string]any)["prices"] = prices
				response["model"] = localModel // the catalog id the table is keyed by
			}
			_, httpServer := v2Server(t, config, fixture, 300)
			status, body := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "v2-caller-prices")
			if scenario == "applied" {
				if status != 200 || local.count() != 1 {
					t.Fatalf("status %d: %s", status, body)
				}
				return
			}
			if status == 200 || local.count() != 0 || cloud.count() != 0 {
				t.Fatalf("an unacknowledged caller price reached inference: %d %s", status, body)
			}
			if lookups := fixture.snapshot(); len(lookups) != 1 {
				t.Fatalf("%d lookups", len(lookups))
			} else if _, sent := lookups[0].Raw["routing"]; !sent || strings.Contains(string(lookups[0].Raw["routing"]), `"exclusive"`) {
				t.Fatalf("routing sent: %s", lookups[0].Raw["routing"])
			}
		})
	}
}

// A pin that is the previous contract's digest of the configured prices is
// refused before any request, with the migration named: that digest never
// comes back from the Routes API, so it could only fail on every decision.
func TestPreviousContractPriceDigestIsRefusedAtStartup(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "test-key")
	for name, pin := range map[string]string{"exclusive true": "d5c16e9479a1aba03cd9e38b4b3878e21440b5a5d156f46e882d3eec835cf57f", "exclusive false": "3abe1bbdf0242628534a98f82b917de91f29b93320c876891005afb993530b35"} {
		t.Run(name, func(t *testing.T) {
			config := currentTestConfig(t)
			config.Decisions.URL = "http://127.0.0.1:1/v1/routes"
			config.Decisions.Routing = json.RawMessage(`{"prices":{"anthropic/claude-sonnet-5":{"input":3,"cache_read":0.3,"cache_write":3.75,"output":15},"deepseek/deepseek-v4.1-flash":{"input":0.1,"cache_read":0,"cache_write":0,"output":0.4}}}`)
			config.Decisions.PriceSHA256 = pin
			_, err := NewHTTPDecisions(config)
			if err == nil || !strings.Contains(err.Error(), "arc-caller-prices.v1") || !strings.Contains(err.Error(), "arc-route-prices.v1") {
				t.Fatalf("previous-contract pin not refused with a migration message: %v", err)
			}
		})
	}
}

// The previous contract's members are refused at startup in every API mode,
// since every mode sends the routing block as configured.
func TestRemovedRoutingMembersAreRefusedInEveryMode(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "test-key")
	for _, mode := range []string{DecisionsCurrent, DecisionsExecutionContextV1, DecisionsExecutionContextV2} {
		for _, routing := range []string{`{"exclusive":false}`, `{"exclusive":true}`, `{"price_profile_id":"local-qwen"}`, `{"price_profile_id":"local-qwen","price_profile_version":2}`} {
			t.Run(mode+" "+routing, func(t *testing.T) {
				config := testConfig(t, "http://127.0.0.1:1/v1/chat/completions", false)
				config.Decisions.APIMode = mode
				config.Decisions.URL = "http://127.0.0.1:1/v1/routes"
				if mode == DecisionsCurrent {
					config.Decisions.OfferSHA256 = ""
				}
				block := strings.TrimSuffix(routing, "}")
				if mode == DecisionsExecutionContextV2 {
					block += `,"policy":{"preset":"` + hybridPreset + `"}`
				}
				config.Decisions.Routing = json.RawMessage(block + "}")
				_, err := NewHTTPDecisions(config)
				if err == nil || !strings.Contains(err.Error(), "Routes API V1") {
					t.Fatalf("removed member accepted in %s mode: %v", mode, err)
				}
			})
		}
	}
}

// Applied candidates confine the pick even when nothing is priced: the
// catalog receipt alone does not prove the route stayed inside models.
func TestCatalogOnlyCandidatesConfineTheSelectedModel(t *testing.T) {
	const sonnet, flash = "anthropic/claude-sonnet-5", "deepseek/deepseek-v4.1-flash"
	for _, scenario := range []string{"applied", "selected outside resolved", "duplicate resolved", "unapplied"} {
		t.Run(scenario, func(t *testing.T) {
			response := currentContractResponse(t)
			config, _ := currentContractConfig(t, response, "custom") // selects sonnet at the catalog
			config.Decisions.Routing = json.RawMessage(`{"v_value":2,"models":["anthropic/*","deepseek/*"]}`)
			models := map[string]any{"status": "applied", "models": []string{"anthropic/*", "deepseek/*"}, "resolved": []string{sonnet, flash}}
			response["routing"].(map[string]any)["models"] = models
			switch scenario {
			case "selected outside resolved":
				models["resolved"] = []string{flash}
			case "duplicate resolved":
				models["resolved"] = []string{sonnet, sonnet}
			case "unapplied":
				models["status"] = "unsupported"
			}
			decision, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput())
			if scenario != "applied" {
				if err == nil {
					t.Fatal("a pick outside the applied candidates was accepted")
				}
				return
			}
			if err != nil || decision.Model != sonnet || decision.PriceProfile != nil {
				t.Fatalf("catalog-only candidates: %+v %v", decision, err)
			}
		})
	}
}

// A hybrid profile that prices only its local model still lets the route
// pick a catalog-priced cloud candidate, acknowledged in the same table.
func TestV2CallerPriceForLocalModelAllowsACatalogPricedCloudPick(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	localModel, cloudModel := config.Deployments[0].RegistryModel, config.Deployments[1].RegistryModel
	config.Decisions.Routing = json.RawMessage(`{"policy":{"preset":"` + hybridPreset + `"},"prices":{"` + localModel + `":{"input":0.01,"cache_read":0.01,"cache_write":0.01,"output":0.01}}}`)
	fixture := &v2Decisions{choose: func(contextV2) string { return cloudActionID }}
	fixture.mutate = func(response map[string]any) {
		echo := v1PricesEcho(t, map[string]map[string]any{
			localModel: {"input": 0.01, "cache_read": 0.01, "cache_write": 0.01, "output": 0.01, "source": sources("caller")},
			cloudModel: {"input": 0.6, "cache_read": 0.15, "cache_write": 0.6, "output": 2.5, "source": sources("catalog")},
		}, []string{})
		response["routing"].(map[string]any)["prices"] = echo
		response["identity"].(map[string]any)["prices"] = map[string]any{"basis": "caller", "price_sha256": echo["price_sha256"]}
		response["model"] = cloudModel
	}
	_, httpServer := v2Server(t, config, fixture, 300)
	if status, body := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "v2-cloud-pick"); status != 200 || cloud.count() != 1 || local.count() != 0 {
		t.Fatalf("status %d local=%d cloud=%d: %s", status, local.count(), cloud.count(), body)
	}
}

// The deployed top result carries no price receipt but does echo the
// candidate members, so a current-mode top profile may confine the pick
// with models / excluded_models and is acknowledged from those echoes.
// Prices or a profile still need a receipt the top result lacks.
func TestCurrentTopAcknowledgesCandidateEchoesWithoutAReceipt(t *testing.T) {
	for _, scenario := range []string{"applied", "selected outside resolved", "unapplied", "missing echo", "changed exclusions", "prices need a receipt", "profile needs a receipt"} {
		t.Run(scenario, func(t *testing.T) {
			config := currentTestConfig(t)
			action := config.Actions[0]
			model := config.Deployments[0].RegistryModel
			config.Decisions.Routing = json.RawMessage(`{"models":["qwen/*"],"excluded_models":["qwen/qwen3.8-9b"]}`)
			routing := map[string]any{"models": map[string]any{"status": "applied", "models": []string{"qwen/*"}, "resolved": []string{model, "qwen/qwen3.8-72b"}}, "excluded_models": map[string]any{"status": "applied", "excluded_models": []string{"qwen/qwen3.8-9b"}}, "unapplied_keys": []string{}}
			response := map[string]any{"route_id": "route-top", "model": model, "worker": action.Worker, "policy_action_id": action.ID, "routing": routing}
			switch scenario {
			case "selected outside resolved":
				routing["models"].(map[string]any)["resolved"] = []string{"qwen/qwen3.8-72b"}
			case "unapplied":
				routing["models"].(map[string]any)["status"] = "unsupported"
			case "missing echo":
				delete(routing, "excluded_models")
			case "changed exclusions":
				routing["excluded_models"].(map[string]any)["excluded_models"] = []string{}
			case "prices need a receipt":
				config.Decisions.Routing = json.RawMessage(`{"models":["qwen/*"],"prices":{"` + model + `":{"input":0.01}}}`)
				delete(routing, "excluded_models")
			case "profile needs a receipt":
				config.Decisions.Routing = json.RawMessage(`{"models":["qwen/*"],"price_profile":{"id":"local-qwen"}}`)
				delete(routing, "excluded_models")
			}
			decision, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput())
			if scenario != "applied" {
				if err == nil {
					t.Fatal("top result accepted without acknowledging its candidates")
				}
				return
			}
			if err != nil || decision.Result != "top" || decision.IdentitySource != "configured_manifest" || decision.Model != action.Model {
				t.Fatalf("top with candidates: %+v %v", decision, err)
			}
		})
	}
}
