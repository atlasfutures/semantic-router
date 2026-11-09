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

// policyRetainedEntry is one session's record, the writers in flight on it
// and the generation of the last write that started or was dropped. A
// record is kept only by a writer that started last and finished with no
// other writer in flight, so overlapping calls (a side call holds no lease;
// a busy retry can finish after a later call) never leave a record standing
// for a session another of them changed.
type policyRetainedEntry struct {
	generation uint64
	writers    int
	// owner is the scope of the scorer whose decide the record is.
	owner   string
	request *policyRetainedRequest
}

// policyRetainedSessionCache remembers, per service session, the request this
// process last had a policy service retain there. A route lookup's admission
// reads it to predict an append (predictRaylineRoutesEncode): the episode's
// committed prefix is in the service's session only if it is the request
// last retained there.
//
// An entry is keyed by the package and the episode hash a decide names
// (policyRetainedBackend), not by endpoint: two recipes whose endpoint sets
// differ can still reach one replica, so every scorer deciding that package
// for that conversation invalidates the entry. Across distinct backends that
// only costs an exemption. A record names the scorer whose decide it is (its
// owner scope, unique to one recipe's scorer in one router generation), and
// only that scorer's lookups read it, since only that scorer's episode
// committed the prefix it describes; a reload starts every scorer with no
// records.
//
// It is in memory, bounded and never persisted, so it knows only the decides
// this process made; a lookup whose previous turn another router instance
// decided finds nothing and is predicted a full encode. That is the safe
// direction: the cost of a missed exemption is a queued lookup, the cost of
// a wrong one is a full encode past the gate.
type policyRetainedSessionCache struct {
	mu         sync.Mutex
	entries    map[string]*policyRetainedEntry
	order      []string
	generation uint64
}

// policyRetainedSessionEntries bounds the sessions remembered; the oldest
// inserted is forgotten first.
const policyRetainedSessionEntries = 4096

var policyRetainedSessions = &policyRetainedSessionCache{entries: map[string]*policyRetainedEntry{}}

func policyRetainedKey(backend, episode string) string { return backend + "\x00" + episode }

// policyRetainedBackend names the session store a decide writes: the package
// it decides with.
func policyRetainedBackend(arcConfig *config.RaylineARCAlgorithmConfig) string {
	if arcConfig == nil || arcConfig.PolicyService == nil {
		return ""
	}
	return "package:" + arcConfig.PolicyService.PackageSHA256
}

// policyRetainedScope is where a scorer's decides record retained sessions:
// the backend they write and the scorer's own owner scope.
type policyRetainedScope struct {
	backend string
	owner   string
}

func (scope policyRetainedScope) valid() bool { return scope.backend != "" && scope.owner != "" }

// entry is the session's entry, created if absent. The caller holds mu.
func (cache *policyRetainedSessionCache) entry(scope policyRetainedScope, episode string) *policyRetainedEntry {
	key := policyRetainedKey(scope.backend, episode)
	entry, known := cache.entries[key]
	if !known {
		if len(cache.order) >= policyRetainedSessionEntries {
			delete(cache.entries, cache.order[0])
			cache.order = cache.order[1:]
		}
		cache.order = append(cache.order, key)
		entry = &policyRetainedEntry{}
		cache.entries[key] = entry
	}
	return entry
}

// begin starts a write to the session: a call is about to change it. The
// record is dropped, and the returned fence is what the call's finish
// presents; zero when the scope records nothing.
func (cache *policyRetainedSessionCache) begin(scope policyRetainedScope, episode string) uint64 {
	if !scope.valid() {
		return 0
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	entry := cache.entry(scope, episode)
	cache.generation++
	entry.generation, entry.owner, entry.request = cache.generation, "", nil
	entry.writers++
	return entry.generation
}

// finish ends the write begin fenced. shape is the request the session now
// holds, decided on the episode at turnIndex, or nil when the call left
// nothing a turn commits (it failed, it was a side call, it was served
// relaxed). The record is kept only when this was the last write to start
// and no other is still in flight; otherwise the session is left unknown.
func (cache *policyRetainedSessionCache) finish(
	scope policyRetainedScope,
	episode string,
	fence uint64,
	turnIndex uint64,
	shape *raylinearc.PolicyPrefixShape,
) {
	if !scope.valid() || fence == 0 {
		return
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	entry, known := cache.entries[policyRetainedKey(scope.backend, episode)]
	if !known {
		return
	}
	if entry.writers > 0 {
		entry.writers--
	}
	if entry.generation != fence || entry.writers > 0 || shape == nil {
		// A later write started, another is in flight, or this one left
		// nothing: whatever stood is no longer known to be in the session.
		entry.owner, entry.request = "", nil
		return
	}
	entry.owner = scope.owner
	entry.request = &policyRetainedRequest{
		len: shape.Len, digest: shape.Digest, system: shape.System, tools: shape.Tools, turn: turnIndex + 1,
	}
}

// drop forgets the session's record without a write of its own in flight: a
// call changed the session after the fact (an append, a call served strict
// that was sent relaxed).
func (cache *policyRetainedSessionCache) drop(scope policyRetainedScope, episode string) {
	if !scope.valid() {
		return
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	entry := cache.entry(scope, episode)
	cache.generation++
	entry.generation, entry.owner, entry.request = cache.generation, "", nil
}

// holds reports whether the episode's session is known to hold the prefix
// state committed, under the same system prompt and tools as current: the
// decide this process recorded is the one whose commit state is. shellChanged
// is true when it holds the prefix under a different shell, which the
// service rebuilds.
func (cache *policyRetainedSessionCache) holds(
	scope policyRetainedScope,
	episode string,
	state *raylinearc.EpisodeState,
	current *raylinearc.PolicyPrefixShape,
) (held, shellChanged bool) {
	if !scope.valid() || state == nil || state.Policy == nil || current == nil {
		return false, false
	}
	cache.mu.Lock()
	var retained *policyRetainedRequest
	if entry, known := cache.entries[policyRetainedKey(scope.backend, episode)]; known && entry.request != nil &&
		entry.owner == scope.owner {
		copied := *entry.request
		retained = &copied
	}
	cache.mu.Unlock()
	if retained == nil || retained.turn != state.TurnIndex ||
		retained.len != state.Policy.PrefixLen || retained.digest != state.Policy.PrefixDigest {
		return false, false
	}
	if retained.system != current.System || retained.tools != current.Tools {
		return false, true
	}
	return true, false
}

// raylineARCRetainedScoper is a selector that can name the scope its policy
// decides record retained sessions under; the zero scope when it records
// none (an artifact-mode selector, or one not yet armed).
type raylineARCRetainedScoper interface {
	policyRetainedScope() policyRetainedScope
}

func (selector *raylineARCSelector) policyRetainedScope() policyRetainedScope {
	armed := selector.armedComponents()
	if armed == nil {
		return policyRetainedScope{}
	}
	if scorer, ok := armed.scorer.(*policyServiceScorer); ok {
		return scorer.retainedScope
	}
	return policyRetainedScope{}
}
