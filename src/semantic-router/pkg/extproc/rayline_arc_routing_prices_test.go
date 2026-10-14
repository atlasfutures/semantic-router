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
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type routingPriceClock struct {
	mu  sync.Mutex
	now time.Time
}

func (clock *routingPriceClock) Now() time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return clock.now
}

func (clock *routingPriceClock) advance(by time.Duration) {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	clock.now = clock.now.Add(by)
}

// newRoutingPriceFeed serves the contract's feed fixture.
func newRoutingPriceFeed(t *testing.T) *httptest.Server {
	t.Helper()
	body, err := os.ReadFile("../selection/raylinearc/testdata/routing_prices/feed.v1.json")
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		_, _ = writer.Write(body)
	}))
	t.Cleanup(server.Close)
	return server
}

// withRoutingPrices arms the fixture's selector with a price cache on the
// fake feed, on a clock the test controls.
func (fixture *policySelectorFixture) withRoutingPrices(t *testing.T, fetch bool) *routingPriceClock {
	t.Helper()
	clock := &routingPriceClock{now: time.Date(2026, 10, 14, 16, 0, 0, 0, time.UTC)}
	cache := raylinearc.NewRoutingPriceCache(raylinearc.RoutingPriceCacheConfig{
		URL: newRoutingPriceFeed(t).URL, TTL: time.Minute, RefreshAhead: 10 * time.Second,
		MaxAge: 10 * time.Minute, FetchTimeout: time.Second, Now: clock.Now,
	})
	if fetch {
		if err := cache.Refresh(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	armed := *fixture.selector.armed.Load()
	armed.prices = cache
	fixture.selector.arm(&armed)
	return clock
}

func liveEcho(digest string, fallback ...string) func(raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
	return func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
		if request.RoutingPrices == nil {
			return nil // the service echoes only when the request carried prices
		}
		version, asOf := request.RoutingPrices.StoreVersion, request.RoutingPrices.AsOf
		return &raylinearc.PolicyRoutingPricesUsed{
			Mode: raylinearc.RoutingPricesModeLive, StoreVersion: &version, AsOf: &asOf,
			Digest: &digest, FallbackArms: append([]string{}, fallback...),
		}
	}
}

// Fresh and stale rates ride on the decision request as the feed served
// them, and the echo becomes the ledger entry's price audit.
func TestPolicySelectorSendsLivePricesAndAuditsTheEcho(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[2].ActionID })
	digest := strings.Repeat("d", 64)
	fixture.fake.echo = liveEcho(digest, "arm-fallback")
	clock := fixture.withRoutingPrices(t, true)
	state, _ := raylinearc.NewEpisodeState(2)

	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0].RoutingPrices
	if sent == nil || sent.Schema != raylinearc.RoutingPricesSchema || sent.StoreVersion != "v000000000042" ||
		sent.AsOf != "2026-10-14T16:00:00Z" || len(sent.Rates) != 2 {
		t.Fatalf("routing_prices sent = %+v", sent)
	}
	glm := sent.Rates["z-ai/glm-4.6"]
	if *glm.Input != 0.6 || glm.CacheRead != nil || glm.ValidUntil != nil {
		t.Fatalf("rates not forwarded as served: %+v", glm)
	}
	trace := result.RaylineARC
	if trace.RoutingPricesStatus != "fresh" || trace.RoutingPricesUsed == nil ||
		trace.PolicyPackageSHA256 != policyTestPackage {
		t.Fatalf("trace = status %q echo %+v package %q", trace.RoutingPricesStatus, trace.RoutingPricesUsed, trace.PolicyPackageSHA256)
	}
	ledger := trace.PolicyNextState.Ledger
	audit := ledger[len(ledger)-1].Prices
	if audit == nil || audit.Mode != raylinearc.RoutingPricesModeLive || audit.PackageSHA256 != policyTestPackage ||
		audit.StoreVersion != "v000000000042" || audit.AsOf != "2026-10-14T16:00:00Z" || audit.Digest != digest ||
		len(audit.FallbackArms) != 1 || audit.FallbackArms[0] != "arm-fallback" ||
		ledger[len(ledger)-1].ActionID != bindings[2].ActionID {
		t.Fatalf("ledger audit = %+v", audit)
	}
	fields := map[string]interface{}{}
	addPolicyPriceAuditFields(fields, trace)
	if fields["routing_prices_mode"] != "live" || fields["routing_prices_digest"] != digest ||
		fields["policy_package_sha256"] != policyTestPackage || fields["routing_prices_status"] != "fresh" {
		t.Fatalf("selection log audit = %v", fields)
	}

	// Past the TTL the same rates are still sent, as stale.
	clock.advance(2 * time.Minute)
	result, err = fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil || result.RaylineARC.RoutingPricesStatus != "stale" || fixture.fake.received()[1].RoutingPrices == nil {
		t.Fatalf("stale select: %v, status %q", err, result.RaylineARC.RoutingPricesStatus)
	}
}

// A feed that covers few of a large package's models leaves most arms on
// package prices, and the service lists every one: 1300 fallback arms are
// served, not refused, and the ledger and log keep 8 and the count.
func TestPolicySelectorServesAnEchoWithManyFallbackArms(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	arms := make([]string, 1300)
	for index := range arms {
		arms[index] = fmt.Sprintf("%064x", index)
	}
	fixture.fake.echo = liveEcho(strings.Repeat("d", 64), arms...)
	fixture.withRoutingPrices(t, true)
	state, _ := raylinearc.NewEpisodeState(2)

	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select with 1300 fallback arms: %v", err)
	}
	audit := result.RaylineARC.PolicyNextState.Ledger[0].Prices
	if audit == nil || audit.FallbackArmCount != 1300 || len(audit.FallbackArms) != 8 || audit.FallbackArms[0] != arms[0] {
		t.Fatalf("ledger audit = %+v", audit)
	}
	fields := map[string]interface{}{}
	addPolicyPriceAuditFields(fields, result.RaylineARC)
	if logged, _ := fields["routing_prices_fallback_arms"].([]string); len(logged) != 8 ||
		fields["routing_prices_fallback_arm_count"] != 1300 {
		t.Fatalf("selection log fallback arms = %v, count %v", fields["routing_prices_fallback_arms"], fields["routing_prices_fallback_arm_count"])
	}
}

// Past max age, or before the first fetch, the request carries no rates:
// the service scores with package prices and the turn is still served.
func TestPolicySelectorFallsBackToPackagePricesWithoutUsableRates(t *testing.T) {
	for name, setup := range map[string]func(*testing.T, *policySelectorFixture) string{
		"never fetched": func(t *testing.T, f *policySelectorFixture) string {
			f.withRoutingPrices(t, false)
			return "absent"
		},
		"past max age": func(t *testing.T, f *policySelectorFixture) string {
			f.withRoutingPrices(t, true).advance(10 * time.Minute)
			return "expired"
		},
	} {
		t.Run(name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, "")
			bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
			fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
			fixture.fake.echo = liveEcho(strings.Repeat("d", 64))
			wantStatus := setup(t, fixture)
			state, _ := raylinearc.NewEpisodeState(2)
			result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
			if err != nil {
				t.Fatalf("select: %v", err)
			}
			if fixture.fake.received()[0].RoutingPrices != nil {
				t.Fatal("unusable rates were sent")
			}
			// No prices sent, so no echo and no audit to record.
			audit := result.RaylineARC.PolicyNextState.Ledger[0].Prices
			if result.RaylineARC.RoutingPricesStatus != wantStatus || audit != nil ||
				result.RaylineARC.RoutingPricesUsed != nil {
				t.Fatalf("status %q audit %+v", result.RaylineARC.RoutingPricesStatus, audit)
			}
		})
	}
}

// Without routing_prices configured, and a service that echoes nothing, the
// request and the ledger are exactly what they were before live prices.
func TestPolicySelectorWithoutLivePricesIsUnchanged(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	if fixture.fake.received()[0].RoutingPrices != nil || result.RaylineARC.RoutingPricesStatus != "" ||
		result.RaylineARC.RoutingPricesUsed != nil || result.RaylineARC.PolicyNextState.Ledger[0].Prices != nil {
		t.Fatalf("live-price state without live prices: %+v", result.RaylineARC)
	}
	if newRoutingPriceCache(nil) != nil {
		t.Fatal("an absent routing_prices config built a cache")
	}
	if newRoutingPriceCache(&config.RaylineARCRoutingPricesConfig{CredentialEnv: "RAYLINE_ARC_TEST_UNSET_PRICE_TOKEN"}) != nil {
		t.Fatal("an unset credential built a cache")
	}
}

// An echo that is not of the prices this request sent is refused, since it
// would be recorded as the decision's audit.
func TestPolicySelectorRefusesAnUntrueRoutingPriceEcho(t *testing.T) {
	digest := strings.Repeat("d", 64)
	other := "v000000000041"
	for name, test := range map[string]struct {
		fetch     bool
		echo      func(raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed
		wantClass string
	}{
		"live without prices sent": {false, func(raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			asOf := "2026-10-14T16:00:00Z"
			return &raylinearc.PolicyRoutingPricesUsed{Mode: "live", StoreVersion: &other, AsOf: &asOf, Digest: &digest}
		}, arcFailurePolicyPricesUnsent},
		"package echo without prices sent": {false, func(raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			asOf := "2026-10-14T16:00:00Z"
			return &raylinearc.PolicyRoutingPricesUsed{Mode: "package", StoreVersion: &other, AsOf: &asOf, FallbackArms: []string{}}
		}, arcFailurePolicyPricesUnsent},
		"package echo of another store version": {true, func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			used := liveEcho(digest)(request)
			used.Mode, used.StoreVersion = raylinearc.RoutingPricesModePackage, &other
			return used
		}, arcFailurePolicyPricesMismatch},
		"another store version": {true, func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			used := liveEcho(digest)(request)
			used.StoreVersion = &other
			return used
		}, arcFailurePolicyPricesMismatch},
		"another as_of": {true, func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			used := liveEcho(digest)(request)
			later := "2026-10-14T16:05:00Z"
			used.AsOf = &later
			return used
		}, arcFailurePolicyPricesMismatch},
		// Prices sent but no echo: the decision would have no price audit.
		"prices sent, no echo": {true, func(raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			return nil
		}, arcFailurePolicyPricesUnechoed},
	} {
		t.Run(name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, "")
			bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
			fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
			fixture.fake.echo = test.echo
			fixture.withRoutingPrices(t, test.fetch)
			state, _ := raylinearc.NewEpisodeState(2)
			_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
			var failure *raylineARCSelectionFailure
			if !errors.As(err, &failure) || failure.class != test.wantClass {
				t.Fatalf("error = %v, want class %s", err, test.wantClass)
			}
		})
	}
	// The same instant in another offset form is the same as_of.
	if !sameInstant("2026-10-14T16:00:00Z", "2026-10-14T16:00:00+00:00") || sameInstant("2026-10-14T16:00:00Z", "later") {
		t.Fatal("as_of instants compared wrongly")
	}
}

// auth google_id_token sends the metadata server's identity token for the
// configured audience, and needs no credential in the environment.
func TestRoutingPriceCacheConfiguredForGoogleIDToken(t *testing.T) {
	const audience = "https://routerproxy-abc123-uc.a.run.app"
	payload := base64.RawURLEncoding.EncodeToString([]byte(fmt.Sprintf(`{"exp":%d}`, time.Now().Add(time.Hour).Unix())))
	token := "e30." + payload + ".c2ln"
	metadata := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Metadata-Flavor") != "Google" || request.URL.Query().Get("audience") != audience ||
			request.URL.Query().Get("format") != "full" {
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		_, _ = writer.Write([]byte(token))
	}))
	t.Cleanup(metadata.Close)
	body, err := os.ReadFile("../selection/raylinearc/testdata/routing_prices/feed.v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var sent atomic.Value
	feed := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		sent.Store(request.Header.Get("Authorization"))
		_, _ = writer.Write(body)
	}))
	t.Cleanup(feed.Close)

	cache := newRoutingPriceCacheWithMetadata(&config.RaylineARCRoutingPricesConfig{
		URL: feed.URL + "/arc/routing-prices", Auth: config.RaylineARCRoutingPricesAuthGoogleIDToken, Audience: audience,
		TTLSeconds: 300, MaxAgeSeconds: 3600, FetchTimeoutMillis: 2000,
	}, metadata.URL)
	if cache == nil {
		t.Fatal("google_id_token built no cache")
	}
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got, _ := sent.Load().(string); got != "Bearer "+token {
		t.Fatalf("Authorization = %q", got)
	}
}
