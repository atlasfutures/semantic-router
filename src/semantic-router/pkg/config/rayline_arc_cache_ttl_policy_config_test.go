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

package config

import (
	"strings"
	"testing"
)

// user_turn_cache names one of the service's modes; a mode that reads
// cache_context, and any cache_ttl_policy row, needs cache_context on.
func TestRaylineARCUserTurnCacheAndTTLPolicyValidate(t *testing.T) {
	moonshot := RaylineARCCacheTTLPolicyRow{
		Provider: "Moonshot AI", TTLSeconds: 300,
		Source: "platform.kimi.ai/docs/guide/use-context-caching-feature-of-kimi-api 5m idle, read 2026-10-21",
	}
	cases := []struct {
		name   string
		mutate func(*RaylineARCPolicyServiceConfig)
		want   string
	}{
		{"absent", func(*RaylineARCPolicyServiceConfig) {}, ""},
		{"switch cost without cache_context", func(p *RaylineARCPolicyServiceConfig) {
			p.UserTurnCache = RaylineARCUserTurnCacheSwitchCost
		}, ""},
		{"ttl mode with cache_context", func(p *RaylineARCPolicyServiceConfig) {
			p.UserTurnCache, p.CacheContext = RaylineARCUserTurnCacheRepartitionTTL, true
		}, ""},
		{"ttl mode without cache_context", func(p *RaylineARCPolicyServiceConfig) {
			p.UserTurnCache = RaylineARCUserTurnCacheRepartitionTTL
		}, "reads cache_context"},
		{"multiwarm without cache_context", func(p *RaylineARCPolicyServiceConfig) {
			p.UserTurnCache = RaylineARCUserTurnCacheRepartitionMultiwarm
		}, "reads cache_context"},
		{"unknown mode", func(p *RaylineARCPolicyServiceConfig) { p.UserTurnCache = "always_warm" }, "user_turn_cache must be"},
		{"policy rows", func(p *RaylineARCPolicyServiceConfig) {
			p.CacheContext, p.CacheTTLPolicy = true, []RaylineARCCacheTTLPolicyRow{moonshot}
		}, ""},
		{"policy rows without cache_context", func(p *RaylineARCPolicyServiceConfig) {
			p.CacheTTLPolicy = []RaylineARCCacheTTLPolicyRow{moonshot}
		}, "feeds cache_context"},
		{"row without a source", func(p *RaylineARCPolicyServiceConfig) {
			row := moonshot
			row.Source = ""
			p.CacheContext, p.CacheTTLPolicy = true, []RaylineARCCacheTTLPolicyRow{row}
		}, "source is required"},
		{"row without a ttl", func(p *RaylineARCPolicyServiceConfig) {
			row := moonshot
			row.TTLSeconds = 0
			p.CacheContext, p.CacheTTLPolicy = true, []RaylineARCCacheTTLPolicyRow{row}
		}, "ttl_seconds must be"},
		{"repeated provider", func(p *RaylineARCPolicyServiceConfig) {
			p.CacheContext, p.CacheTTLPolicy = true, []RaylineARCCacheTTLPolicyRow{moonshot, moonshot}
		}, "repeats provider"},
	}
	for _, tc := range cases {
		_, decision := policyDispatchFixture()
		tc.mutate(decision.Algorithm.RaylineARC.PolicyService)
		err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC)
		if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
			t.Fatalf("%s: error = %v, want %q", tc.name, err, tc.want)
		}
	}
}

// A row naming the worker wins over the provider's general row; no row is no
// TTL.
func TestRaylineARCCacheTTLPolicyForPrefersTheWorkersRow(t *testing.T) {
	policy := &RaylineARCPolicyServiceConfig{CacheTTLPolicy: []RaylineARCCacheTTLPolicyRow{
		{Provider: "Moonshot AI", TTLSeconds: 300, Source: "general"},
		{Provider: "Moonshot AI", Workers: []string{"kimi-k3"}, TTLSeconds: 600, Source: "kimi"},
	}}
	if row, ok := policy.CacheTTLPolicyFor("Moonshot AI", "kimi-k3"); !ok || row.Source != "kimi" {
		t.Fatalf("worker row = %+v, %v", row, ok)
	}
	if row, ok := policy.CacheTTLPolicyFor("Moonshot AI", "kimi-k2.7"); !ok || row.Source != "general" {
		t.Fatalf("general row = %+v, %v", row, ok)
	}
	for _, provider := range []string{"Modal", ""} {
		if _, ok := policy.CacheTTLPolicyFor(provider, "kimi-k3"); ok {
			t.Fatalf("provider %q found a row", provider)
		}
	}
	var none *RaylineARCPolicyServiceConfig
	if _, ok := none.CacheTTLPolicyFor("Moonshot AI", "kimi-k3"); ok {
		t.Fatal("a nil config found a row")
	}
}
