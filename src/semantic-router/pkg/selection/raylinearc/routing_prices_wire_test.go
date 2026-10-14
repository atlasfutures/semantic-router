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

package raylinearc

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// testdata/routing_prices mirrors the live-prices contract's JSON examples
// (feed, request routing_prices, response routing_prices_used, manifest
// capability) on top of the pathfinder v1 fixtures.
const routingPriceFixtures = "testdata/routing_prices"

func readRoutingPriceFixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(routingPriceFixtures, name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func requireSameJSON(t *testing.T, name string, want []byte, got []byte) {
	t.Helper()
	var wantValue, gotValue any
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(wantValue, gotValue) {
		t.Errorf("%s differs:\nwant %s\ngot  %s", name, want, got)
	}
}

func TestRoutingPriceFixturesRoundTripStrictly(t *testing.T) {
	cases := map[string]any{
		"decision_request.v1.json":          &PolicyDecisionRequest{},
		"decision_response_live.v1.json":    &PolicyDecisionResponse{},
		"decision_response_package.v1.json": &PolicyDecisionResponse{},
		// Pathfinder's golden fixtures, copied verbatim from its
		// tests/fixtures/arc_policy_service (branch arc-live-prices).
		"pathfinder_decision_response_routing_prices.v1.json": &PolicyDecisionResponse{},
		"pathfinder_package_manifest.v4.live_prices.json":     &PolicyPackageManifest{},
		"pathfinder_package_manifest.v5.live_prices.json":     &PolicyPackageManifestV5{},
	}
	for name, target := range cases {
		body := readRoutingPriceFixture(t, name)
		if err := decodeStrict(body, target); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		encoded, err := json.Marshal(target)
		if err != nil {
			t.Fatal(err)
		}
		requireSameJSON(t, name, body, encoded)
	}
}

// Pathfinder's golden request's routing_prices round-trips strictly through
// the router's type, so the router writes exactly the members the service
// reads. The rest of that request is a relaxed-mode request (episode_mode),
// which this router does not send, so only routing_prices is compared.
func TestPathfinderGoldenRoutingPricesRoundTripStrictly(t *testing.T) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(readRoutingPriceFixture(t, "pathfinder_decision_request_routing_prices.v1.json"), &request); err != nil {
		t.Fatal(err)
	}
	var prices PolicyRoutingPrices
	if err := decodeStrict(request["routing_prices"], &prices); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(prices)
	if err != nil {
		t.Fatal(err)
	}
	requireSameJSON(t, "golden routing_prices", request["routing_prices"], encoded)
}

// The whole golden request decodes strictly and EncodePolicyDecisionRequest,
// the decide body the client sends, writes it back: routing_prices travels
// in the envelope beside episode_mode.
func TestPathfinderGoldenRequestEncodesAsTheDecideBody(t *testing.T) {
	body := readRoutingPriceFixture(t, "pathfinder_decision_request_routing_prices.v1.json")
	var request PolicyDecisionRequest
	if err := decodeStrict(body, &request); err != nil {
		t.Fatal(err)
	}
	if request.RoutingPrices == nil || request.EpisodeMode != PolicyEpisodeModeRelaxed {
		t.Fatalf("decoded %+v", request)
	}
	encoded, err := EncodePolicyDecisionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	requireSameJSON(t, "golden decide body", body, encoded)
}

// The feed's models become the request's rates as they are: keyed by model
// id, the four legs, validity window and source unchanged, nulls kept.
func TestRoutingPriceFeedBecomesTheRequestRates(t *testing.T) {
	feed, err := DecodeRoutingPriceFeed(readRoutingPriceFixture(t, "feed.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var request map[string]json.RawMessage
	if err = json.Unmarshal(readRoutingPriceFixture(t, "decision_request.v1.json"), &request); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(feed.RequestPrices())
	if err != nil {
		t.Fatal(err)
	}
	requireSameJSON(t, "routing_prices", request["routing_prices"], encoded)
}

// Rates are keyed by the feed's model, the OpenRouter slug the package's arms
// name, even when the entry also carries a billing_model, which is not
// forwarded: keying by the billing-card id would match no arm. store_version
// is opaque, so the feed's integer-as-string passes through unchanged.
func TestRoutingPriceRatesKeyByModelNotBillingModel(t *testing.T) {
	body := `{"schema": "arc-routing-prices.v1", "store_version": "17", "as_of": "2026-10-14T16:00:00.000Z",
		"currency": "USD", "unit": "usd_per_million_tokens", "models": [
		{"model": "anthropic/claude-opus-5", "billing_model": "claude-opus-5", "input": 5, "cache_read": 0.5,
		 "cache_write": 6.25, "output": 25, "valid_from": "2026-10-14T00:00:00.000Z", "valid_until": null,
		 "source": "openrouter:anthropic"}]}`
	feed, err := DecodeRoutingPriceFeed([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	prices := feed.RequestPrices()
	if _, ok := prices.Rates["anthropic/claude-opus-5"]; !ok || len(prices.Rates) != 1 || prices.StoreVersion != "17" {
		t.Fatalf("rates keyed %v, store_version %q", prices.Rates, prices.StoreVersion)
	}
	encoded, err := json.Marshal(prices)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("billing_model")) || bytes.Contains(encoded, []byte(`"claude-opus-5"`)) {
		t.Fatalf("billing model forwarded: %s", encoded)
	}
}

// A request without live prices carries no routing_prices key at all, so it
// stays byte-for-byte what the service accepted before.
func TestPolicyDecisionRequestOmitsAbsentRoutingPrices(t *testing.T) {
	var request PolicyDecisionRequest
	if err := decodeStrict(readPolicyFixture(t, "decision_request.v1.json"), &request); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("routing_prices")) {
		t.Fatalf("request without prices encodes routing_prices: %s", encoded)
	}
	request.RoutingPrices = &PolicyRoutingPrices{
		Schema: RoutingPricesSchema, StoreVersion: "v1", AsOf: "2026-10-14T16:00:00Z",
		Rates: map[string]PolicyRoutingRate{"m": {}},
	}
	encoded, err = json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte(`"routing_prices":{"schema":"arc-routing-prices.v1","store_version":"v1","as_of":"2026-10-14T16:00:00Z","rates":{"m":{"input":null,"cache_read":null,"cache_write":null,"output":null,"valid_from":null,"valid_until":null,"source":null}}}`)) {
		t.Fatalf("request with prices encodes %s", encoded)
	}
}

func TestDecodeRoutingPriceFeedRefusesWhatItCannotForward(t *testing.T) {
	body := string(readRoutingPriceFixture(t, "feed.v1.json"))
	if _, err := DecodeRoutingPriceFeed([]byte(strings.Replace(body, `"currency"`, `"surprise": 1, "currency"`, 1))); err != nil {
		t.Fatalf("a feed with an added field was refused: %v", err)
	}
	cases := map[string][2]string{
		"another schema":      {`"arc-routing-prices.v1"`, `"arc-routing-prices.v2"`},
		"another unit":        {`"usd_per_million_tokens"`, `"usd_per_token"`},
		"another currency":    {`"USD"`, `"EUR"`},
		"no store version":    {`"v000000000042"`, `""`},
		"a malformed as_of":   {`"as_of": "2026-10-14T16:00:00Z"`, `"as_of": "yesterday"`},
		"a duplicate model":   {`"z-ai/glm-4.6"`, `"anthropic/claude-sonnet-4.5"`},
		"an empty model id":   {`"z-ai/glm-4.6"`, `""`},
		"a malformed expiry":  {`"valid_until": "2026-10-14T17:00:00Z"`, `"valid_until": "soon"`},
		"a non-numeric rate":  {`"input": 3.0`, `"input": "3.0"`},
		"a truncated payload": {`]`, ``},
	}
	for name, edit := range cases {
		changed := strings.Replace(body, edit[0], edit[1], 1)
		if changed == body {
			t.Fatalf("%s: fixture no longer contains %s", name, edit[0])
		}
		if _, err := DecodeRoutingPriceFeed([]byte(changed)); err == nil {
			t.Errorf("a feed with %s was accepted", name)
		}
	}
}

func TestDecodePolicyDecisionResponseReadsTheRoutingPriceEcho(t *testing.T) {
	old, err := DecodePolicyDecisionResponse(readPolicyFixture(t, "decision_response.v1.json"))
	if err != nil || old.RoutingPricesUsed != nil {
		t.Fatalf("pre-live-prices response: %+v, %v", old, err)
	}
	live, err := DecodePolicyDecisionResponse(readRoutingPriceFixture(t, "decision_response_live.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if used := live.RoutingPricesUsed; used == nil || used.Mode != RoutingPricesModeLive ||
		*used.StoreVersion != "v000000000042" || len(used.FallbackArms) != 1 {
		t.Fatalf("live echo decoded as %+v", live.RoutingPricesUsed)
	}
	golden, err := DecodePolicyDecisionResponse(readRoutingPriceFixture(t, "pathfinder_decision_response_routing_prices.v1.json"))
	if err != nil || golden.RoutingPricesUsed == nil || *golden.RoutingPricesUsed.StoreVersion != "rps-000000000017" {
		t.Fatalf("pathfinder golden echo: %+v, %v", golden, err)
	}
	pkg, err := DecodePolicyDecisionResponse(readRoutingPriceFixture(t, "decision_response_package.v1.json"))
	if err != nil || pkg.RoutingPricesUsed == nil || pkg.RoutingPricesUsed.Mode != RoutingPricesModePackage {
		t.Fatalf("package echo: %+v, %v", pkg, err)
	}
	pkgBody := string(readRoutingPriceFixture(t, "decision_response_package.v1.json"))
	for _, field := range []string{`"store_version": "v000000000042"`, `"as_of": "2026-10-14T16:00:00Z"`} {
		name := strings.SplitN(field, ":", 2)[0]
		changed := strings.Replace(pkgBody, field, name+": null", 1)
		if changed == pkgBody {
			t.Fatalf("package fixture no longer contains %s", field)
		}
		if _, err := DecodePolicyDecisionResponse([]byte(changed)); err == nil {
			t.Errorf("a package echo with a null %s was accepted", name)
		}
	}

	body := string(readRoutingPriceFixture(t, "decision_response_live.v1.json"))
	cases := map[string][2]string{
		"an unknown mode":          {`"mode": "live"`, `"mode": "snapshot"`},
		"a digest that is no hash": {`"digest": "3b4c`, `"digest": "XYZ`},
		"a live echo, no version":  {`"store_version": "v000000000042"`, `"store_version": null`},
		"a live echo, no as_of":    {`"as_of": "2026-10-14T16:00:00Z"`, `"as_of": null`},
		"an unknown echo field":    {`"fallback_arms"`, `"surprise": 1, "fallback_arms"`},
		"an empty fallback arm":    {`"fallback_arms": [`, `"fallback_arms": ["", `},
	}
	for name, edit := range cases {
		changed := strings.Replace(body, edit[0], edit[1], 1)
		if changed == body {
			t.Fatalf("%s: fixture no longer contains %s", name, edit[0])
		}
		if _, err := DecodePolicyDecisionResponse([]byte(changed)); err == nil {
			t.Errorf("a response with %s was accepted", name)
		}
	}
}

// live_prices_affect_decisions is itself the capability: Pathfinder's v4
// live-prices manifest decodes with it recognized, a package-priced one
// decodes with it off, and only a factored package may set it.
func TestDecodePolicyPackageManifestRecognizesTheLivePricesCapability(t *testing.T) {
	if manifest, err := DecodePolicyPackageManifest(readPolicyFixture(t, "package_manifest.v4.json")); err != nil ||
		manifest.Pricing.LivePricesAffectDecisions {
		t.Fatalf("package-priced manifest: %v", err)
	}
	body := readRoutingPriceFixture(t, "pathfinder_package_manifest.v4.live_prices.json")
	manifest, err := DecodePolicyPackageManifest(body)
	if err != nil || !manifest.Pricing.LivePricesAffectDecisions || manifest.Decision.Mode != "factored" {
		t.Fatalf("live-priced manifest: %v", err)
	}
	learned := bytes.Replace(body, []byte(`"mode": "factored"`), []byte(`"mode": "learned"`), 1)
	if bytes.Equal(learned, body) {
		t.Fatal("fixture no longer names its decision mode")
	}
	if _, err = DecodePolicyPackageManifest(learned); err == nil {
		t.Fatal("a learned package claiming live prices affect its decisions was accepted")
	}
	learnedOff := bytes.Replace(learned, []byte(`"live_prices_affect_decisions": true`), []byte(`"live_prices_affect_decisions": false`), 1)
	if _, err = DecodePolicyPackageManifest(learnedOff); err != nil {
		t.Fatalf("a learned package without live prices was refused: %v", err)
	}
	extra := bytes.Replace(body, []byte(`"live_prices_affect_decisions": true`),
		[]byte(`"live_prices_affect_decisions": true, "external_routing_prices": true`), 1)
	if _, err = DecodePolicyPackageManifest(extra); err == nil {
		t.Fatal("a manifest naming external_routing_prices was accepted")
	}
}

// Every manifest version VSR decodes takes the capability on a factored
// package and refuses it on any other: Pathfinder's v5 live-prices manifest,
// and the v6 fixture with the flag set.
func TestLivePricesCapabilityHoldsAcrossManifestVersions(t *testing.T) {
	v5 := readRoutingPriceFixture(t, "pathfinder_package_manifest.v5.live_prices.json")
	manifest, err := DecodePolicyPackageManifestV5(v5)
	if err != nil || !manifest.Pricing.LivePricesAffectDecisions || manifest.Decision.Mode != "factored" {
		t.Fatalf("v5 live-priced manifest: %v", err)
	}
	if _, err = DecodePolicyPackageManifestV5OrLater(v5); err != nil {
		t.Fatalf("v5-or-later refused the v5 live-priced manifest: %v", err)
	}
	learned := bytes.Replace(v5, []byte(`"mode": "factored"`), []byte(`"mode": "learned"`), 1)
	if bytes.Equal(learned, v5) {
		t.Fatal("v5 fixture no longer names its decision mode")
	}
	if _, err = DecodePolicyPackageManifestV5(learned); err == nil {
		t.Fatal("a learned v5 package claiming live prices was accepted")
	}
	learnedOff := bytes.Replace(learned, []byte(`"live_prices_affect_decisions": true`), []byte(`"live_prices_affect_decisions": false`), 1)
	if _, err = DecodePolicyPackageManifestV5(learnedOff); err != nil {
		t.Fatalf("a learned v5 package without live prices was refused: %v", err)
	}

	v6 := manifestMap(t, policyV6Fixture)
	v6["pricing"].(map[string]any)["live_prices_affect_decisions"] = true
	if _, err = DecodePolicyPackageManifestV6(encodeManifest(t, v6)); err != nil {
		t.Fatalf("a factored v6 package with live prices was refused: %v", err)
	}
	v6["decision"].(map[string]any)["mode"] = "learned"
	if _, err = DecodePolicyPackageManifestV6(encodeManifest(t, v6)); err == nil {
		t.Fatal("a learned v6 package claiming live prices was accepted")
	}
}
