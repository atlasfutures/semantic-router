package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineARCInflightRegistry coalesces identical in-flight turns on one
// episode.
//
// A client that resends a turn while the first copy is still being decided
// (Claude Code retries a slow turn on its own) would otherwise contend with
// that copy for the episode lease, wait out acquire_timeout and fail with
// session_busy, and with a cold decision service one slow turn becomes a
// cascade of them. A resend is the same turn, not a new one, so it joins the
// pending decision instead: it waits for the first copy to decide, dispatches
// with that decision against a read-only copy of the prepared state, and
// commits nothing.
//
// Only the first copy records the turn. If its own upstream call fails while
// a resend's succeeds, that turn goes unrecorded: the next turn scores against
// a trajectory one turn behind, and the reply carries no attribution. That is
// deliberate. Letting any copy commit makes the leased transaction shared
// mutable state across concurrent requests, and the coalescing a resend needs
// does not justify it. Eval traffic serializes per trajectory upstream and
// does not depend on this path.
//
// Identity is the episode store, the deciding ARC decision, the episode, the
// exact request body and every request input outside the body that changes
// how ARC treats the turn: the wire format, the episode close request, the
// thinking-lever eligibility header, an injected fault and the caller's
// provider credentials, which decide who issues encrypted reasoning. Any other request
// on the same episode, including one that differs by a byte, keeps today's
// behaviour. The registry is process-local: two router instances do not
// coalesce across each other.
type raylineARCInflightRegistry struct {
	mu      sync.Mutex
	entries map[string]*raylineARCInflightEntry
}

// raylineARCInflightEntry is one pending decision that identical requests may
// join. It is published at most once and finished exactly once.
type raylineARCInflightEntry struct {
	key  string
	done chan struct{}
	// finished closes once the entry has left the registry, so a request whose
	// leader gave up waits for that rather than finding the entry again.
	finished chan struct{}

	// Set before done closes, read only after.
	result *selection.SelectionResult
	state  *raylinearc.EpisodeState
	ok     bool

	once       sync.Once
	finishOnce sync.Once
}

func raylineARCInflightKey(store raylinearc.EpisodeStore, episodeIDHash string, body []byte, inputs string) string {
	digest := sha256.Sum256(body)
	return fmt.Sprintf("%p:%s:%s:%s", store, episodeIDHash, hex.EncodeToString(digest[:]), inputs)
}

// raylineARCTurnInputs names every request input outside the body that changes
// how ARC treats a turn, so two requests that differ in one never coalesce.
// That includes the explicit turn-signal headers: a side call and a main turn
// with the same body are different turns.
//
// The decision is named by its own ARC config: several decisions may share one
// selector and episode store, and each routes with its own plugins and system
// prompts.
//
// Caller-supplied credentials (the credential resolver's injected headers)
// enter only as a digest, so the key never holds a credential.
func raylineARCTurnInputs(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	credentialHeaders []string,
) string {
	eligibility := ""
	if lever := arcConfig.ThinkingLever; lever != nil && lever.EligibilityHeader != "" {
		eligibility = strings.TrimSpace(reqCtx.Headers[lever.EligibilityHeader])
	}
	return fmt.Sprintf(
		"decision=%p|format=%s|close=%t|eligible=%q|fault=%s|credentials=%s|call_kind=%q|compaction=%q|agent_key_source=%q",
		arcConfig,
		reqCtx.SourceFormat,
		reqCtx.RaylineARCCloseRequested,
		eligibility,
		requestedFault(arcConfig, reqCtx),
		raylineARCCredentialDigest(reqCtx.Headers, credentialHeaders),
		strings.TrimSpace(reqCtx.Headers[raylineARCCallKindHeader]),
		strings.TrimSpace(reqCtx.Headers[raylineARCCompactionHeader]),
		// The key source decides whether a subagent is a main turn of its
		// own episode or a side call, so two requests that differ in it are
		// different turns.
		strings.TrimSpace(reqCtx.Headers[raylineARCAgentKeySourceHeader]),
	)
}

// raylineARCCredentialDigest fingerprints the values of the credential headers
// a request carries, in a fixed order.
func raylineARCCredentialDigest(headers map[string]string, names []string) string {
	sorted := slices.Sorted(slices.Values(names))
	digest := sha256.New()
	for _, name := range sorted {
		digest.Write([]byte(name))
		digest.Write([]byte{0})
		digest.Write([]byte(headers[name]))
		digest.Write([]byte{0})
	}
	return hex.EncodeToString(digest.Sum(nil)[:8])
}

// join returns the pending entry for key and whether the caller leads it. A
// leader must finish its entry on every path; a follower waits on it.
func (registry *raylineARCInflightRegistry) join(key string) (*raylineARCInflightEntry, bool) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if entry, found := registry.entries[key]; found {
		return entry, false
	}
	if registry.entries == nil {
		registry.entries = make(map[string]*raylineARCInflightEntry)
	}
	entry := &raylineARCInflightEntry{key: key, done: make(chan struct{}), finished: make(chan struct{})}
	registry.entries[key] = entry
	return entry, true
}

// publish hands the leader's decision to every follower, present and future,
// for as long as the leader's turn is in flight.
func (registry *raylineARCInflightRegistry) publish(
	entry *raylineARCInflightEntry,
	result *selection.SelectionResult,
	state *raylinearc.EpisodeState,
) {
	if entry == nil || result == nil {
		return
	}
	entry.once.Do(func() {
		entry.result = cloneRaylineARCSelectionResult(result)
		entry.state = cloneARCState(state)
		entry.ok = true
		close(entry.done)
	})
}

// finish ends the leader's turn. An entry that never published releases its
// followers to decide for themselves; either way the next identical request
// is a new turn.
func (registry *raylineARCInflightRegistry) finish(entry *raylineARCInflightEntry) {
	if entry == nil {
		return
	}
	entry.once.Do(func() { close(entry.done) })
	registry.mu.Lock()
	if registry.entries[entry.key] == entry {
		delete(registry.entries, entry.key)
	}
	registry.mu.Unlock()
	entry.finishOnce.Do(func() { close(entry.finished) })
}

// awaitFinished blocks until entry has left the registry, or ctx ends.
func (entry *raylineARCInflightEntry) awaitFinished(ctx context.Context) error {
	select {
	case <-entry.finished:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// wait blocks until the leader publishes or gives up, or ctx ends. It
// reports the leader's decision and prepared state, and whether there is one.
func (entry *raylineARCInflightEntry) wait(
	ctx context.Context,
) (*selection.SelectionResult, *raylinearc.EpisodeState, bool, error) {
	select {
	case <-entry.done:
	case <-ctx.Done():
		return nil, nil, false, ctx.Err()
	}
	if !entry.ok {
		return nil, nil, false, nil
	}
	return cloneRaylineARCSelectionResult(entry.result), cloneARCState(entry.state), true, nil
}

// cloneRaylineARCSelectionResult copies everything a follower might touch, so
// no two requests share a mutable map, slice or ledger.
func cloneRaylineARCSelectionResult(result *selection.SelectionResult) *selection.SelectionResult {
	if result == nil {
		return nil
	}
	cloned := *result
	if result.AllScores != nil {
		cloned.AllScores = maps.Clone(result.AllScores)
	}
	if result.RaylineARC != nil {
		trace := *result.RaylineARC
		trace.RawScores = slices.Clone(trace.RawScores)
		trace.AdjustedScores = slices.Clone(trace.AdjustedScores)
		trace.SwitchCostUSD = slices.Clone(trace.SwitchCostUSD)
		trace.CacheMissTokens = slices.Clone(trace.CacheMissTokens)
		trace.UpgradeExemptions = slices.Clone(trace.UpgradeExemptions)
		trace.ExcludedArms = slices.Clone(trace.ExcludedArms)
		trace.EncoderVisitedReplicaIDs = slices.Clone(trace.EncoderVisitedReplicaIDs)
		if result.RaylineARC.PolicyNextState != nil {
			trace.PolicyNextState = result.RaylineARC.PolicyNextState.Clone()
		}
		if result.RaylineARC.PolicyBoundary != nil {
			boundary := *result.RaylineARC.PolicyBoundary
			trace.PolicyBoundary = &boundary
		}
		cloned.RaylineARC = &trace
	}
	return &cloned
}

// publishRaylineARCDecision hands a leader's decision to any identical resend
// waiting on it, with the state it was decided against. A request that leads
// nothing publishes nothing.
func (r *OpenAIRouter) publishRaylineARCDecision(ctx *RequestContext, result *selection.SelectionResult) {
	if r == nil || ctx == nil || ctx.RaylineARCInflight == nil || ctx.RaylineARCTransaction == nil {
		return
	}
	r.raylineARCInflight.publish(ctx.RaylineARCInflight, result, ctx.RaylineARCTransaction.state)
}
