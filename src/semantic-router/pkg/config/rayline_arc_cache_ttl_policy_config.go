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

import "fmt"

// The decide request's selection.user_turn_cache modes (pathfinder
// docs/arc_serving_contract.md, "Stage one"): how a typed user turn prices
// the warm prefix a model switch forfeits.
const (
	RaylineARCUserTurnCacheSwitchCost           = "switch_cost"
	RaylineARCUserTurnCacheRepartition          = "repartition"
	RaylineARCUserTurnCacheRepartitionTTL       = "repartition_ttl"
	RaylineARCUserTurnCacheRepartitionMultiwarm = "repartition_multiwarm"
)

// RaylineARCCacheTTLPolicyRow is one provider's documented prompt-cache
// lifetime: the TTL a call that provider served is reported with in
// cache_context when its dispatched body states none the Router reads
// (only an Anthropic model's cache_control is read from the wire). It is a
// documented assumption, never a cache-verified fact: Source names the
// provider's own documentation and when it was read, and travels as the
// entry's ttl_source.
type RaylineARCCacheTTLPolicyRow struct {
	// Provider is the provider that served the call, exactly as the upstream
	// reports it (llm_usage upstream_provider, e.g. "Moonshot AI").
	Provider string `yaml:"provider"`
	// Workers, when set, limits the row to these workers; empty is every
	// worker the provider serves.
	Workers []string `yaml:"workers,omitempty"`
	// TTLSeconds is the documented idle expiry: a cache entry unused this
	// long is gone, and each hit refreshes it. Only an idle-expiry rule
	// belongs here; a documented minimum retention (an entry may live
	// longer) is not an expiry, and a provider whose lifetime is
	// undocumented has no row.
	TTLSeconds int `yaml:"ttl_seconds"`
	// Source is the documentation the TTL comes from, with its read date,
	// or the measurement it was observed in (route, date, sample size).
	Source string `yaml:"source"`
	// Basis is documented (the provider's own documentation; the default)
	// or observed (a cache-read probe on the route at known gaps). It names
	// the entry's ttl_source prefix, so the two are never confused.
	Basis string `yaml:"basis,omitempty"`
}

// The cache_ttl_policy row bases.
const (
	RaylineARCCacheTTLBasisDocumented = "documented"
	RaylineARCCacheTTLBasisObserved   = "observed"
)

// raylineARCMaxCacheTTLPolicyRows bounds cache_ttl_policy.
const raylineARCMaxCacheTTLPolicyRows = 64

// raylineARCMaxCacheTTLSeconds is the longest TTL a row may state (a day).
const raylineARCMaxCacheTTLSeconds = 86400

// RaylineARCUserTurnCacheReadsCacheContext reports whether a mode reads the
// decide request's cache_context.
func RaylineARCUserTurnCacheReadsCacheContext(mode string) bool {
	return mode == RaylineARCUserTurnCacheRepartitionTTL || mode == RaylineARCUserTurnCacheRepartitionMultiwarm
}

func validateRaylineARCUserTurnCache(cfg *RaylineARCPolicyServiceConfig) error {
	switch cfg.UserTurnCache {
	case "", RaylineARCUserTurnCacheSwitchCost, RaylineARCUserTurnCacheRepartition,
		RaylineARCUserTurnCacheRepartitionTTL, RaylineARCUserTurnCacheRepartitionMultiwarm:
	default:
		return fmt.Errorf("user_turn_cache must be empty or one of %s, %s, %s, %s",
			RaylineARCUserTurnCacheSwitchCost, RaylineARCUserTurnCacheRepartition,
			RaylineARCUserTurnCacheRepartitionTTL, RaylineARCUserTurnCacheRepartitionMultiwarm)
	}
	if RaylineARCUserTurnCacheReadsCacheContext(cfg.UserTurnCache) && !cfg.CacheContext {
		return fmt.Errorf("user_turn_cache %s reads cache_context: set cache_context: true", cfg.UserTurnCache)
	}
	if len(cfg.CacheTTLPolicy) > 0 && !cfg.CacheContext {
		return fmt.Errorf("cache_ttl_policy feeds cache_context: set cache_context: true")
	}
	if len(cfg.CacheTTLPolicy) > raylineARCMaxCacheTTLPolicyRows {
		return fmt.Errorf("cache_ttl_policy holds at most %d rows", raylineARCMaxCacheTTLPolicyRows)
	}
	seen := make(map[string]bool, len(cfg.CacheTTLPolicy))
	for index, row := range cfg.CacheTTLPolicy {
		if row.Provider == "" || len(row.Provider) > maxRaylineARCConfigStringLength {
			return fmt.Errorf("cache_ttl_policy[%d].provider is required", index)
		}
		if row.TTLSeconds <= 0 || row.TTLSeconds > raylineARCMaxCacheTTLSeconds {
			return fmt.Errorf("cache_ttl_policy[%d].ttl_seconds must be between 1 and %d", index,
				raylineARCMaxCacheTTLSeconds)
		}
		switch row.Basis {
		case "", RaylineARCCacheTTLBasisDocumented, RaylineARCCacheTTLBasisObserved:
		default:
			return fmt.Errorf("cache_ttl_policy[%d].basis must be empty, %s or %s", index,
				RaylineARCCacheTTLBasisDocumented, RaylineARCCacheTTLBasisObserved)
		}
		if row.Source == "" || len(row.Source) > maxRaylineARCConfigStringLength {
			return fmt.Errorf("cache_ttl_policy[%d].source is required: the documentation and its read date", index)
		}
		workers := row.Workers
		if len(workers) == 0 {
			workers = []string{""}
		}
		for _, worker := range workers {
			key := row.Provider + "\x00" + worker
			if seen[key] {
				return fmt.Errorf("cache_ttl_policy[%d] repeats provider %q for worker %q", index, row.Provider, worker)
			}
			seen[key] = true
		}
	}
	return nil
}

// CacheTTLPolicyFor is the row for a call the provider served on the worker:
// a row naming the worker before one for every worker. ok is false when no
// row applies, and the call's TTL stays unknown.
func (cfg *RaylineARCPolicyServiceConfig) CacheTTLPolicyFor(
	provider string,
	worker string,
) (RaylineARCCacheTTLPolicyRow, bool) {
	if cfg == nil || provider == "" {
		return RaylineARCCacheTTLPolicyRow{}, false
	}
	var general *RaylineARCCacheTTLPolicyRow
	for index := range cfg.CacheTTLPolicy {
		row := &cfg.CacheTTLPolicy[index]
		if row.Provider != provider {
			continue
		}
		if len(row.Workers) == 0 {
			general = row
			continue
		}
		for _, named := range row.Workers {
			if named == worker {
				return *row, true
			}
		}
	}
	if general != nil {
		return *general, true
	}
	return RaylineARCCacheTTLPolicyRow{}, false
}
