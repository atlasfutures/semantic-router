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
	"math"
	"sort"
	"time"
)

const (
	// CacheContextWindow is how long after its last call an arm is still
	// reported in a decide request's cache_context: Anthropic's longest
	// prompt-cache TTL.
	CacheContextWindow = time.Hour
	// MaxCacheContextActions bounds cache_context.actions, most recent first.
	MaxCacheContextActions = 16

	wireTTLEphemeral5mSeconds = 300
	wireTTLEphemeral1hSeconds = 3600
	wireTTLSourceEphemeral5m  = "anthropic cache_control ephemeral 5m"
	wireTTLSourceEphemeral1h  = "anthropic cache_control ephemeral 1h"
)

// CallCache is what an arm's last call sent and was told about its prompt
// cache. It is replaced whole, never changed in place, so a shallow copy of
// WorkerWarmth may share it. Nil and zero members are unknown.
type CallCache struct {
	// ActionID is the policy-service action the call served.
	ActionID string
	// TTLSeconds and TTLSource are the cache_control TTL the call sent on the
	// wire; 0 and "" when it stated none.
	TTLSeconds int
	TTLSource  string
	// PromptTokens is the provider's count of the whole prompt, cache read
	// and write included; the cache legs are its provider-reported parts.
	PromptTokens     *int
	CacheReadTokens  *int
	CacheWriteTokens *int
}

// PolicyCacheContext is the decide request's cache_context
// (pathfinder docs/arc_serving_contract.md).
type PolicyCacheContext struct {
	Actions []PolicyActionCache `json:"actions"`
}

// PolicyActionCache is one action's last call. Every member but ActionID is
// null when unknown, never 0.
type PolicyActionCache struct {
	ActionID         string   `json:"action_id"`
	GapS             *float64 `json:"gap_s"`
	TTLS             *float64 `json:"ttl_s"`
	TTLSource        *string  `json:"ttl_source"`
	PromptTokens     *int     `json:"prompt_tokens"`
	CacheReadTokens  *int     `json:"cache_read_tokens"`
	CacheWriteTokens *int     `json:"cache_write_tokens"`
}

// BuildPolicyCacheContext is the cache_context for a decide at now: one
// entry per arm whose last call completed within CacheContextWindow and
// recorded the action it served, most recent first, at most
// MaxCacheContextActions. armOf maps an action id to the arm the package
// binds it to; an entry whose action the package does not bind to that arm
// is left out. Every value the service would refuse is sent null or left
// out, so the field is always valid.
func BuildPolicyCacheContext(
	warmth []*WorkerWarmth,
	now time.Time,
	armOf func(actionID string) (int, bool),
) *PolicyCacheContext {
	type candidate struct {
		arm    int
		warmth *WorkerWarmth
	}
	candidates := make([]candidate, 0, len(warmth))
	for arm, value := range warmth {
		if value == nil || value.Call == nil || value.LastUsed.IsZero() || !lowerHex64(value.Call.ActionID) {
			continue
		}
		if armOf == nil {
			continue
		}
		if bound, ok := armOf(value.Call.ActionID); !ok || bound != arm {
			continue
		}
		if now.Sub(value.LastUsed) > CacheContextWindow {
			continue
		}
		candidates = append(candidates, candidate{arm: arm, warmth: value})
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].warmth.LastUsed.After(candidates[j].warmth.LastUsed)
	})
	context := &PolicyCacheContext{Actions: make([]PolicyActionCache, 0, min(len(candidates), MaxCacheContextActions))}
	seen := make(map[string]bool, len(candidates))
	for _, candidate := range candidates {
		if len(context.Actions) == MaxCacheContextActions {
			break
		}
		call := candidate.warmth.Call
		if seen[call.ActionID] {
			continue
		}
		seen[call.ActionID] = true
		context.Actions = append(context.Actions, actionCacheEntry(call, now.Sub(candidate.warmth.LastUsed)))
	}
	return context
}

func actionCacheEntry(call *CallCache, gap time.Duration) PolicyActionCache {
	entry := PolicyActionCache{ActionID: call.ActionID}
	// A call dated after now (another replica's clock) has no known gap.
	if seconds := gap.Seconds(); gap >= 0 && !math.IsInf(seconds, 0) && !math.IsNaN(seconds) {
		entry.GapS = &seconds
	}
	if call.TTLSeconds > 0 && call.TTLSource != "" {
		ttl, source := float64(call.TTLSeconds), call.TTLSource
		entry.TTLS, entry.TTLSource = &ttl, &source
	}
	entry.PromptTokens = knownCount(call.PromptTokens)
	entry.CacheReadTokens = knownCount(call.CacheReadTokens)
	entry.CacheWriteTokens = knownCount(call.CacheWriteTokens)
	// Known legs that exceed the prompt are not the prompt's parts; the
	// prompt is kept and the legs are unknown.
	if entry.PromptTokens != nil {
		legs := 0
		for _, leg := range []*int{entry.CacheReadTokens, entry.CacheWriteTokens} {
			if leg != nil {
				legs += *leg
			}
		}
		if legs > *entry.PromptTokens {
			entry.CacheReadTokens, entry.CacheWriteTokens = nil, nil
		}
	}
	return entry
}

func knownCount(count *int) *int {
	if count == nil || *count < 0 {
		return nil
	}
	value := *count
	return &value
}

func lowerHex64(value string) bool {
	return len(value) == 64 && lowerHex(value)
}

// WireCacheTTL reads the prompt-cache TTL a dispatched body states: the
// shortest cache_control TTL on it (Anthropic orders a longer TTL before a
// shorter one, so the last breakpoint's prefix expires first), with its
// source. A body with no cache_control, or one whose TTL is not Anthropic's
// 5m or 1h, states none: 0 and "".
func WireCacheTTL(body []byte) (int, string) {
	if !bytes.Contains(body, []byte(`"cache_control"`)) {
		return 0, ""
	}
	var root interface{}
	if json.Unmarshal(body, &root) != nil {
		return 0, ""
	}
	shortest, unknown := 0, false
	var walk func(value interface{})
	walk = func(value interface{}) {
		switch typed := value.(type) {
		case map[string]interface{}:
			for key, member := range typed {
				if key == "cache_control" {
					if directive, ok := member.(map[string]interface{}); ok && directive["type"] == "ephemeral" {
						seconds := 0
						switch ttl, stated := directive["ttl"]; {
						case !stated || ttl == nil || ttl == "5m":
							seconds = wireTTLEphemeral5mSeconds
						case ttl == "1h":
							seconds = wireTTLEphemeral1hSeconds
						default:
							unknown = true
						}
						if seconds > 0 && (shortest == 0 || seconds < shortest) {
							shortest = seconds
						}
					}
					continue
				}
				// A tool call's input is the model's own JSON, not the request's.
				if _, object := member.(map[string]interface{}); object && key == "input" {
					continue
				}
				walk(member)
			}
		case []interface{}:
			for _, member := range typed {
				walk(member)
			}
		}
	}
	walk(root)
	switch {
	case unknown || shortest == 0:
		return 0, ""
	case shortest == wireTTLEphemeral1hSeconds:
		return shortest, wireTTLSourceEphemeral1h
	default:
		return shortest, wireTTLSourceEphemeral5m
	}
}
