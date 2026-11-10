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
	"fmt"
	"time"
)

// RaylineARCRoutesCallerBudgetConfig bounds one caller's route lookups.
//
// The lane (routes_api.admission) caps lookups cell-wide, but it caps
// concurrency, not rate, and a refusal frees its slot at once: one client
// retrying a refused lookup in a tight loop still reaches the backend at
// whatever rate it retries. The budget holds each caller to a number of
// lookups a minute and a number at once; past either it answers 429 with
// Retry-After and error.code "caller_budget", before any backend work.
//
// The caller is the conversation the lookup names (x-rayline-session and
// x-rayline-branch, the identity the cell keys episode leases on). A lookup
// that names none is not counted against any caller; the lane still bounds
// it. The counts are kept in the episode Redis when the episode backend is
// redis, so every router instance shares them, and per process otherwise.
// A Redis failure falls back to the per-process counts (fail open), as the
// lane does.
type RaylineARCRoutesCallerBudgetConfig struct {
	// Disabled turns the budget off. Off by default.
	Disabled bool `yaml:"disabled,omitempty"`
	// LookupsPerMinute is how many lookups one caller may send in a fixed
	// one-minute window. Zero selects the default.
	LookupsPerMinute int `yaml:"lookups_per_minute,omitempty"`
	// MaxInflight is how many of one caller's lookups may hold the backend
	// at once. A lookup holds it until the backend answers, which can be
	// after its caller was answered 504. Zero selects the default.
	MaxInflight int `yaml:"max_inflight,omitempty"`
}

// RaylineARCRoutesRefusalCacheConfig answers a refused request's repeats
// locally.
//
// A lookup the decision refused with a deterministic 4xx (an undecidable
// turn, a priced model the package does not offer, a body the service cannot
// represent) is refused identically every time it is sent, so a caller that
// retries it only spends the backend's session slots. For TTL after such a
// refusal, the same caller sending the same request (body and routing
// headers) is answered the same refusal from this process's memory. A 429,
// 408, 409 or any 5xx is never cached, nor a refusal the cell made before
// calling the backend (it costs nothing to repeat), nor the refusal of a
// lookup that joins an episode (it can depend on the episode's state).
type RaylineARCRoutesRefusalCacheConfig struct {
	// Disabled turns the cache off. Off by default.
	Disabled bool `yaml:"disabled,omitempty"`
	// TTLMS is how long a refusal is replayed. Zero selects the default.
	TTLMS int `yaml:"ttl_ms,omitempty"`
	// MaxEntries bounds the refusals one process remembers; the oldest is
	// dropped past it. Zero selects the default.
	MaxEntries int `yaml:"max_entries,omitempty"`
}

const (
	// DefaultRaylineARCRoutesCallerLookupsPerMinute is 4x the gateway's own
	// 30-lookups-a-minute per-user limit, and a conversation is narrower
	// than a user: a caller the gateway admits never reaches it. It trips
	// on a client retrying a lookup in a loop, or on traffic that bypassed
	// (or failed open through) the gateway's limit.
	DefaultRaylineARCRoutesCallerLookupsPerMinute = 120
	// DefaultRaylineARCRoutesCallerMaxInflight: one conversation lane asks
	// one lookup per turn, so four at once is a client that stopped waiting
	// for its answers. It is half the default lane, so one caller can never
	// hold the whole lane.
	DefaultRaylineARCRoutesCallerMaxInflight = 4
	// DefaultRaylineARCRoutesRefusalCacheTTLMS covers a client's whole retry
	// ladder for one turn (about ten retries with backoff).
	DefaultRaylineARCRoutesRefusalCacheTTLMS = 60_000
	// DefaultRaylineARCRoutesRefusalCacheEntries: an entry is a digest key
	// and a refusal envelope of well under a kilobyte, so a full cache is a
	// few MB.
	DefaultRaylineARCRoutesRefusalCacheEntries = 4096

	maxRaylineARCRoutesCallerLookupsPerMinute = 100_000
	minRaylineARCRoutesRefusalCacheTTLMS      = 1000
	maxRaylineARCRoutesRefusalCacheTTLMS      = 600_000
	maxRaylineARCRoutesRefusalCacheEntries    = 1_000_000
)

// EffectiveLookupsPerMinute is the per-caller rate, or zero when the budget is
// disabled.
func (cfg RaylineARCRoutesCallerBudgetConfig) EffectiveLookupsPerMinute() int {
	if cfg.Disabled {
		return 0
	}
	if cfg.LookupsPerMinute > 0 {
		return cfg.LookupsPerMinute
	}
	return DefaultRaylineARCRoutesCallerLookupsPerMinute
}

// EffectiveMaxInflight is the per-caller concurrency, or zero when the budget
// is disabled.
func (cfg RaylineARCRoutesCallerBudgetConfig) EffectiveMaxInflight() int {
	if cfg.Disabled {
		return 0
	}
	if cfg.MaxInflight > 0 {
		return cfg.MaxInflight
	}
	return DefaultRaylineARCRoutesCallerMaxInflight
}

// Enabled reports whether the budget applies.
func (cfg RaylineARCRoutesCallerBudgetConfig) Enabled() bool { return !cfg.Disabled }

// EffectiveTTL is how long a refusal is replayed, or zero when the cache is
// disabled.
func (cfg RaylineARCRoutesRefusalCacheConfig) EffectiveTTL() time.Duration {
	if cfg.Disabled {
		return 0
	}
	if cfg.TTLMS > 0 {
		return time.Duration(cfg.TTLMS) * time.Millisecond
	}
	return DefaultRaylineARCRoutesRefusalCacheTTLMS * time.Millisecond
}

// EffectiveMaxEntries is the cache bound, or zero when it is disabled.
func (cfg RaylineARCRoutesRefusalCacheConfig) EffectiveMaxEntries() int {
	if cfg.Disabled {
		return 0
	}
	if cfg.MaxEntries > 0 {
		return cfg.MaxEntries
	}
	return DefaultRaylineARCRoutesRefusalCacheEntries
}

func validateRaylineARCRoutesCallerBudgetConfig(cfg RaylineARCRoutesAPIConfig) error {
	budget := cfg.CallerBudget
	if budget.LookupsPerMinute < 0 || budget.LookupsPerMinute > maxRaylineARCRoutesCallerLookupsPerMinute {
		return fmt.Errorf("lookups_per_minute must be between 0 and %d", maxRaylineARCRoutesCallerLookupsPerMinute)
	}
	if budget.MaxInflight < 0 || budget.MaxInflight > maxRaylineARCRoutesAdmissionInflight {
		return fmt.Errorf("max_inflight must be between 0 and %d", maxRaylineARCRoutesAdmissionInflight)
	}
	explicit := budget.LookupsPerMinute != 0 || budget.MaxInflight != 0
	if budget.Disabled && explicit {
		return fmt.Errorf("lookups_per_minute and max_inflight need the budget: drop disabled")
	}
	if explicit && !cfg.Enabled {
		return fmt.Errorf("caller_budget needs the endpoint enabled")
	}
	return nil
}

func validateRaylineARCRoutesRefusalCacheConfig(cfg RaylineARCRoutesAPIConfig) error {
	cache := cfg.RefusalCache
	if cache.TTLMS != 0 && (cache.TTLMS < minRaylineARCRoutesRefusalCacheTTLMS || cache.TTLMS > maxRaylineARCRoutesRefusalCacheTTLMS) {
		return fmt.Errorf("ttl_ms must be 0 or between %d and %d", minRaylineARCRoutesRefusalCacheTTLMS, maxRaylineARCRoutesRefusalCacheTTLMS)
	}
	if cache.MaxEntries < 0 || cache.MaxEntries > maxRaylineARCRoutesRefusalCacheEntries {
		return fmt.Errorf("max_entries must be between 0 and %d", maxRaylineARCRoutesRefusalCacheEntries)
	}
	explicit := cache.TTLMS != 0 || cache.MaxEntries != 0
	if cache.Disabled && explicit {
		return fmt.Errorf("ttl_ms and max_entries need the cache: drop disabled")
	}
	if explicit && !cfg.Enabled {
		return fmt.Errorf("refusal_cache needs the endpoint enabled")
	}
	return nil
}
