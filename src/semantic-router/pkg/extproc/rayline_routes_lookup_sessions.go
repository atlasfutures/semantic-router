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
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// raylineRoutesLookupSessionPruneEvery bounds how often a record sweeps the
// expired entries; a scrape always sweeps.
const raylineRoutesLookupSessionPruneEvery = 10 * time.Second

// raylineRoutesLookupSessionTracker counts, per process, the retained encoder
// sessions tracked lookups hold: the distinct episodes (per policy package)
// whose last lookup decision left a session on the encoder (session_action
// created, rebuilt, appended or reused) within the episode idle TTL.
//
// Each such session occupies a slot of the encoder container's session
// budget until the encoder evicts it, whether or not the conversation ever
// sends another lookup, so a dev check counts this against that budget. It is
// an estimate, not the encoder's own count: an entry lives for the episode
// idle TTL from the episode's last lookup, so it overcounts a session the
// encoder evicted sooner (its own idle TTL, a close, capacity), and it does
// not see sessions this process's routed turns or other replicas hold.
type raylineRoutesLookupSessionTracker struct {
	mu        sync.Mutex
	expiry    map[string]time.Time
	lastPrune time.Time
}

var raylineRoutesLookupSessions = func() *raylineRoutesLookupSessionTracker {
	tracker := &raylineRoutesLookupSessionTracker{expiry: map[string]time.Time{}}
	metrics.SetRaylineARCRoutesLookupSessionsSource(func() int { return tracker.count(time.Now()) })
	return tracker
}()

// record notes that a tracked lookup left episode's session on the encoder,
// live until idle from now, and returns the sessions held.
func (tracker *raylineRoutesLookupSessionTracker) record(backend, episode string, idle time.Duration, now time.Time) int {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	if now.Sub(tracker.lastPrune) >= raylineRoutesLookupSessionPruneEvery {
		tracker.pruneLocked(now)
	}
	tracker.expiry[backend+"\x00"+episode] = now.Add(idle)
	return len(tracker.expiry)
}

// count is the sessions held as of now.
func (tracker *raylineRoutesLookupSessionTracker) count(now time.Time) int {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	tracker.pruneLocked(now)
	return len(tracker.expiry)
}

func (tracker *raylineRoutesLookupSessionTracker) pruneLocked(now time.Time) {
	for key, expiry := range tracker.expiry {
		if !now.Before(expiry) {
			delete(tracker.expiry, key)
		}
	}
	tracker.lastPrune = now
}

// recordRaylineRoutesLookupSession records a tracked lookup decision that
// left a retained encoder session, and logs the process's count with it.
func recordRaylineRoutesLookupSession(decisionID, backend, episode, sessionAction string, idle time.Duration) {
	if episode == "" || sessionAction == "" || idle <= 0 {
		return
	}
	held := raylineRoutesLookupSessions.record(backend, episode, idle, time.Now())
	logging.ComponentEvent("extproc", "rayline_routes_lookup_session", map[string]interface{}{
		"routing_decision_id": decisionID,
		"session_action":      sessionAction,
		"lookup_sessions":     held,
	})
}

// raylineRoutesLookupSessionBackend names the encoder sessions an episode's
// lookups share: the policy package's, which every recipe and replica set
// serving that package keys its sessions under. Artifact mode has one.
func raylineRoutesLookupSessionBackend(arcConfig *config.RaylineARCAlgorithmConfig) string {
	if arcConfig == nil || arcConfig.PolicyService == nil {
		return ""
	}
	return "package:" + arcConfig.PolicyService.PackageSHA256
}
