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
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// policyRetainedRequest is the request a strict decide last left in a policy
// service's retained session for one episode: its message count and digest
// (what the episode commits as its prefix) and digests of its system prompt
// and tools, which the service's session also covers.
type policyRetainedRequest struct {
	len    int
	digest string
	system uint64
	tools  uint64
	// turn is the episode turn index the decide's commit leaves. A lookup
	// whose episode is at another index read a state some other decide
	// committed (another router instance, a retry), which may have left
	// another shell in the session.
	turn uint64
}

// policyRetainedSessionCache remembers, per policy backend and episode, the
// request this process last had a policy service retain. A route lookup's
// admission reads it to predict an append (predictRaylineRoutesEncode): the
// episode's committed prefix is in the service's session only if it is the
// request last retained there.
//
// It is in memory, bounded and never persisted, so it knows only the decides
// this process made; a lookup whose previous turn another router instance
// decided finds nothing and is predicted a full encode. That is the safe
// direction: the cost of a missed exemption is a queued lookup, the cost of
// a wrong one is a full encode past the gate.
type policyRetainedSessionCache struct {
	mu       sync.Mutex
	requests map[string]policyRetainedRequest
	order    []string
}

// policyRetainedSessionEntries bounds the episodes remembered; the oldest
// inserted is forgotten first.
const policyRetainedSessionEntries = 4096

var policyRetainedSessions = &policyRetainedSessionCache{requests: map[string]policyRetainedRequest{}}

// policyRetainedScope names the session store a decide writes: the policy
// backend (as admission pools name it) and the package it decides with.
func policyRetainedScope(arcConfig *config.RaylineARCAlgorithmConfig) string {
	if arcConfig == nil || arcConfig.PolicyService == nil {
		return ""
	}
	return raylineRoutesAdmissionPool(arcConfig) + "\x00" + arcConfig.PolicyService.PackageSHA256
}

func policyRetainedKey(scope, episode string) string { return scope + "\x00" + episode }

// remember records shape as the request the episode's session now holds,
// decided on the episode at turnIndex.
func (cache *policyRetainedSessionCache) remember(scope, episode string, turnIndex uint64, shape *raylinearc.PolicyPrefixShape) {
	if shape == nil {
		cache.forget(scope, episode)
		return
	}
	key := policyRetainedKey(scope, episode)
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if _, known := cache.requests[key]; !known {
		if len(cache.order) >= policyRetainedSessionEntries {
			delete(cache.requests, cache.order[0])
			cache.order = cache.order[1:]
		}
		cache.order = append(cache.order, key)
	}
	cache.requests[key] = policyRetainedRequest{
		len: shape.Len, digest: shape.Digest, system: shape.System, tools: shape.Tools, turn: turnIndex + 1,
	}
}

// forget drops what is known of the episode's session: a decide is about to
// change it, or changed it to something no shape describes.
func (cache *policyRetainedSessionCache) forget(scope, episode string) {
	key := policyRetainedKey(scope, episode)
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if _, known := cache.requests[key]; !known {
		return
	}
	delete(cache.requests, key)
	for index, entry := range cache.order {
		if entry == key {
			cache.order = append(cache.order[:index], cache.order[index+1:]...)
			break
		}
	}
}

// holds reports whether the episode's session is known to hold the prefix
// state committed, under the same system prompt and tools as current: the
// decide this process recorded is the one whose commit state is. shellChanged
// is true when it holds the prefix under a different shell, which the
// service rebuilds.
func (cache *policyRetainedSessionCache) holds(
	scope, episode string,
	state *raylinearc.EpisodeState,
	current *raylinearc.PolicyPrefixShape,
) (held, shellChanged bool) {
	if state == nil || state.Policy == nil || current == nil {
		return false, false
	}
	cache.mu.Lock()
	retained, known := cache.requests[policyRetainedKey(scope, episode)]
	cache.mu.Unlock()
	if !known || retained.turn != state.TurnIndex ||
		retained.len != state.Policy.PrefixLen || retained.digest != state.Policy.PrefixDigest {
		return false, false
	}
	if retained.system != current.System || retained.tools != current.Tools {
		return false, true
	}
	return true, false
}
