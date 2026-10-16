package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func validRoutingPricesConfig() *RaylineARCRoutingPricesConfig {
	return &RaylineARCRoutingPricesConfig{
		URL: "https://router-proxy.internal/arc/routing-prices", CredentialEnv: "RAYLINE_ARC_ROUTING_PRICES_TOKEN",
		TTLSeconds: 300, RefreshAheadSeconds: 60, MaxAgeSeconds: 3600, FetchTimeoutMillis: 2000,
	}
}

func TestRaylineARCRoutingPricesConfigParsesAndValidates(t *testing.T) {
	var policy RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte(`
routing_prices:
  url: https://router-proxy.internal/arc/routing-prices
  credential_env: RAYLINE_ARC_ROUTING_PRICES_TOKEN
  ttl_seconds: 300
  refresh_ahead_seconds: 60
  max_age_seconds: 3600
  fetch_timeout_ms: 2000
`), &policy); err != nil {
		t.Fatal(err)
	}
	if *policy.RoutingPrices != *validRoutingPricesConfig() {
		t.Fatalf("parsed %+v", policy.RoutingPrices)
	}

	// Absent is off, and valid.
	cfg, decision := policyDispatchFixture()
	if decision.Algorithm.RaylineARC.PolicyService.RoutingPrices != nil {
		t.Fatal("routing prices are on by default")
	}
	decision.Algorithm.RaylineARC.PolicyService.RoutingPrices = validRoutingPricesConfig()
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid routing_prices refused: %v", err)
	}

	// google_id_token names an audience and no static credential.
	var idToken RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte(`
routing_prices:
  url: https://routerproxy-abc123-uc.a.run.app/arc/routing-prices
  auth: google_id_token
  audience: https://routerproxy-abc123-uc.a.run.app
  ttl_seconds: 300
  max_age_seconds: 3600
  fetch_timeout_ms: 2000
`), &idToken); err != nil {
		t.Fatal(err)
	}
	if idToken.RoutingPrices.Auth != RaylineARCRoutingPricesAuthGoogleIDToken ||
		idToken.RoutingPrices.Audience != "https://routerproxy-abc123-uc.a.run.app" {
		t.Fatalf("parsed %+v", idToken.RoutingPrices)
	}
	decision.Algorithm.RaylineARC.PolicyService.RoutingPrices = idToken.RoutingPrices
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid google_id_token routing_prices refused: %v", err)
	}
}

func TestRaylineARCRoutingPricesConfigRefusesBadBounds(t *testing.T) {
	cases := map[string]struct {
		mutate  func(*RaylineARCRoutingPricesConfig)
		wantErr string
	}{
		"a relative url":            {func(c *RaylineARCRoutingPricesConfig) { c.URL = "/arc/routing-prices" }, "routing_prices url"},
		"credentials in the url":    {func(c *RaylineARCRoutingPricesConfig) { c.URL = "https://u:p@host/x" }, "routing_prices url"},
		"a malformed credential":    {func(c *RaylineARCRoutingPricesConfig) { c.CredentialEnv = "not a name" }, "credential_env"},
		"no ttl":                    {func(c *RaylineARCRoutingPricesConfig) { c.TTLSeconds = 0 }, "ttl_seconds"},
		"refresh ahead past ttl":    {func(c *RaylineARCRoutingPricesConfig) { c.RefreshAheadSeconds = 300 }, "refresh_ahead_seconds"},
		"a max age below the ttl":   {func(c *RaylineARCRoutingPricesConfig) { c.MaxAgeSeconds = 299 }, "max_age_seconds"},
		"no fetch timeout":          {func(c *RaylineARCRoutingPricesConfig) { c.FetchTimeoutMillis = 0 }, "fetch_timeout_ms"},
		"a fetch timeout too large": {func(c *RaylineARCRoutingPricesConfig) { c.FetchTimeoutMillis = 30_001 }, "fetch_timeout_ms"},
		"an unknown auth":           {func(c *RaylineARCRoutingPricesConfig) { c.Auth = "oauth" }, "routing_prices auth"},
		"an audience on bearer":     {func(c *RaylineARCRoutingPricesConfig) { c.Audience = "https://feed.example" }, "audience applies only"},
		"an id token without audience": {func(c *RaylineARCRoutingPricesConfig) {
			c.Auth, c.CredentialEnv = RaylineARCRoutingPricesAuthGoogleIDToken, ""
		}, "routing_prices audience"},
		"an id token with a static credential": {func(c *RaylineARCRoutingPricesConfig) {
			c.Auth, c.Audience = RaylineARCRoutingPricesAuthGoogleIDToken, "https://feed.example"
		}, "credential_env does not apply"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			cfg, decision := policyDispatchFixture()
			prices := validRoutingPricesConfig()
			test.mutate(prices)
			decision.Algorithm.RaylineARC.PolicyService.RoutingPrices = prices
			err := validatePolicyDispatch(cfg, decision)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}
