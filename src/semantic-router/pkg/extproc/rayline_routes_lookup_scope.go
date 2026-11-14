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
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Lookup sessions.
//
// A throwaway lookup -- one that carries routing.execution, caller prices,
// exclusive or exploration, or any lookup on a cell with episode_writes off
// -- decides under a random episode id, so its encode would start from
// token 0: about 2 s at 100-200k tokens, for a conversation whose previous
// lookup encoded the same prefix moments before. When the caller named a
// conversation, the lookup instead decides on that conversation's lookup
// scope: a stable episode id the policy service keeps one warm encoder
// session for, appended to by each lookup. The service records nothing on
// it (session_scope "lookup": no turn, no held arm, no staging), so the
// lookup still commits nothing anywhere, and the scope's busy lock is its
// own, never the conversation's routed episode's.
//
// There is no one-shot fallback. A cell whose routes API is on requires the
// service to advertise lookup_sessions before it is ready; a concurrent
// lookup on the same scope (session_busy) is waited out within the lookup's
// deadline; a service that refuses the scope fails the lookup loudly. Only
// lookups the scope does not apply to are decided one-shot: one that names
// no conversation, one on a replica-pool cell, a side call and a private
// (zdr) turn.

// raylineRoutesLookupScopePrefix namespaces a lookup scope's episode
// identity, so it can never hash to a routed conversation's episode.
const raylineRoutesLookupScopePrefix = "lookup:"

// arcFailureLookupSessionsUnadvertised fails a lookup-scoped decision when
// the service does not advertise lookup_sessions: an older service would
// run it as a strict episode on the scope, holding its arm for the next
// lookup. Readiness requires the capability, so this is a service that lost
// it after the cell armed.
const arcFailureLookupSessionsUnadvertised = "policy_lookup_sessions_unadvertised"

// arcFailureLookupScopeRefused fails a lookup the service refused
// session_scope on (invalid_request naming the field): a cell and a service
// that disagree on the contract, never the caller's request.
const arcFailureLookupScopeRefused = "policy_lookup_scope_refused"

// raylineRoutesLookupScope is the lookup scope a lookup may decide on, or ""
// when the caller named no conversation. The authenticated user is folded
// in although the gateway's conversation identity already carries it, so
// two users can never share a scope even if a caller's identity does not.
func raylineRoutesLookupScope(authzUserID, episode string) string {
	if episode == "" {
		return ""
	}
	return raylineRoutesLookupScopePrefix + authzUserID + ":" + episode
}

// raylineRoutesLookupSessionsServed reports whether a cell decides its
// throwaway lookups on lookup scopes: a policy-service cell with the routes
// API on and a single service. A replica pool's capability list is the
// cell's, not each replica's, and a lookup may fail over to an older
// replica that would run it as a strict episode on its scope, so a pool
// keeps one-shot lookups.
func raylineRoutesLookupSessionsServed(arc *config.RaylineARCAlgorithmConfig) bool {
	return arc != nil && arc.RoutesAPI.Enabled && arc.PolicyService != nil && len(arc.PolicyService.Replicas) == 0
}

// requireRaylineRoutesLookupSessions keeps a cell that decides lookups on
// lookup scopes from being ready on a service that does not keep them:
// there is no one-shot fallback, and an older service would run a scoped
// lookup as a strict episode on its scope. The policy service is deployed
// with lookup_sessions before such a cell rolls.
func requireRaylineRoutesLookupSessions(arc *config.RaylineARCAlgorithmConfig, scorer *policyServiceScorer) error {
	if !raylineRoutesLookupSessionsServed(arc) || scorer.advertises(raylinearc.PolicyCapabilityLookupSessions) {
		return nil
	}
	logging.ComponentErrorEvent("extproc", "rayline_arc_policy_service_unsupported", map[string]interface{}{
		"package_alias": arc.PolicyService.PackageAlias, "class": raylinearc.PolicyCapabilityUnsupportedClass,
		"capability": raylinearc.PolicyCapabilityLookupSessions,
	})
	return &raylinearc.PolicyServiceError{
		Class:  raylinearc.PolicyCapabilityUnsupportedClass,
		Detail: map[string]any{"capability": raylinearc.PolicyCapabilityLookupSessions},
	}
}

// raylineRoutesLookupSessionEligible reports whether a decision is placed on
// its lookup scope, from what the request and the cell say before the
// selection runs. A call the selection reads as a side call leaves the scope
// there (leaveRaylineRoutesLookupScope).
func raylineRoutesLookupSessionEligible(
	request routerruntime.RouteDecisionRequest,
	algorithm *config.AlgorithmConfig,
) bool {
	return request.LookupScope != "" && request.Ephemeral &&
		request.Surface == routerruntime.RouteDecisionSurfaceRoutes &&
		algorithm != nil && raylineRoutesLookupSessionsServed(algorithm.RaylineARC)
}

// applyRaylineRoutesLookupScope places a prepared throwaway lookup on its
// lookup scope. It reports false, leaving the context as built, for a call
// classified as a side call or a private (zdr) turn before the selection: a
// side call is decided relaxed and must not touch the scope, and a zdr turn
// retains nothing anywhere.
func applyRaylineRoutesLookupScope(arcContext *selection.RaylineARCSelectionContext, scope string) bool {
	if arcContext == nil || arcContext.PreparationFailure != "" ||
		arcContext.PolicyCallKind == raylinearc.PolicyCallSide || arcContext.EphemeralTurn {
		return false
	}
	arcContext.RoutesLookupMintedEpisode = arcContext.EpisodeIDHash
	arcContext.EpisodeIDHash = raylinearc.HashEpisodeID(scope)
	arcContext.RoutesLookupSession = true
	return true
}

// leaveRaylineRoutesLookupScope returns a lookup the selection read as a
// side call to the minted episode it was prepared on. The service refuses
// session_scope on a side call, and a side call is decided relaxed, one-shot,
// as every throwaway side call is; the decision service then closes the
// minted episode's session as it closes any one-shot lookup's.
func leaveRaylineRoutesLookupScope(arcContext *selection.RaylineARCSelectionContext) {
	arcContext.EpisodeIDHash = arcContext.RoutesLookupMintedEpisode
	arcContext.RoutesLookupSession = false
	arcContext.PolicyLookupSessions, arcContext.PolicyLookupSessionKey = nil, ""
	logging.ComponentEvent("extproc", "rayline_routes_lookup_scope_skipped", map[string]interface{}{
		"request_id": arcContext.RequestID,
		"reason":     "side_call",
	})
}

// failRaylineRoutesLookupScope logs why a lookup-scoped decision failed on
// its scope and returns its selection failure (503 on the lookup).
func failRaylineRoutesLookupScope(arcContext *selection.RaylineARCSelectionContext, class string) error {
	logging.ComponentErrorEvent("extproc", "rayline_routes_lookup_session_failed", map[string]interface{}{
		"request_id": arcContext.RequestID,
		"class":      class,
	})
	return arcSelectionFailure(class)
}

// policyLookupScopeRefused reports whether a lookup-scoped decide's failure
// is the service refusing session_scope.
func policyLookupScopeRefused(err error) bool {
	var failure *raylinearc.PolicyServiceError
	return errors.As(err, &failure) && failure.Class == "invalid_request" &&
		failure.Detail["field"] == raylinearc.PolicySessionScopeField
}

// policyLookupSessionDeadline is how long a lookup-scoped decide waits out
// session_busy: until the lookup's own deadline. Its only competitor is a
// concurrent lookup on the same conversation, whose append it then extends;
// a lookup still busy at its deadline answers the deadline error any lookup
// that ran out of time does.
func policyLookupSessionDeadline(arcContext *selection.RaylineARCSelectionContext) (time.Time, bool) {
	if !arcContext.RoutesLookupSession || arcContext.RoutesLookupDeadline.IsZero() {
		return time.Time{}, false
	}
	return arcContext.RoutesLookupDeadline, true
}

// raylineRoutesLookupSessionPinsLocal pins lookup scopes' sessions for a
// cell whose episode store shares nothing: each router instance pins its own.
var raylineRoutesLookupSessionPinsLocal = raylinearc.NewMemoryPolicyLookupSessionPins()

// raylineRoutesLookupSessionPinsFor is where a lookup scope's Modal session
// is pinned: beside the cell's episodes when its store is shared (Redis), so
// every router instance routes a conversation's lookups to one container;
// in this process otherwise.
func raylineRoutesLookupSessionPinsFor(store raylinearc.EpisodeStore) raylinearc.PolicyLookupSessionPins {
	if shared, ok := store.(raylinearc.PolicyLookupSessionPins); ok {
		return shared
	}
	return raylineRoutesLookupSessionPinsLocal
}

// raylineRoutesLookupSessionPinKey names a scope's pin on one policy
// backend (its URL and package), so recipes on different Modal apps, which
// may share a pin table or a Redis prefix, never send each other's tokens.
func raylineRoutesLookupSessionPinKey(policy *config.RaylineARCPolicyServiceConfig, scopeHash string) string {
	backend := ""
	if policy != nil {
		backend = policy.BaseURL + "\x00" + policy.PackageSHA256
	}
	return raylinearc.HashEpisodeID("lookup-session:" + backend + "\x00" + scopeHash)
}

// raylineRoutesLookupSessionPinTimeout bounds a pin's read and its write.
const raylineRoutesLookupSessionPinTimeout = time.Second

// bindPolicyLookupSession binds a lookup-scoped decide to its scope's pinned
// Modal session, or to one it starts when there is none (or the pin cannot
// be read: the lookup then starts a session, and pins it).
func bindPolicyLookupSession(
	ctx context.Context,
	backend *raylinearc.PolicySessionBackend,
	arcContext *selection.RaylineARCSelectionContext,
) *raylinearc.PolicySessionBinding {
	var pinned *raylinearc.PolicySession
	if pins := arcContext.PolicyLookupSessions; pins != nil {
		readCtx, cancel := context.WithTimeout(ctx, raylineRoutesLookupSessionPinTimeout)
		session, err := pins.LoadLookupSession(readCtx, arcContext.PolicyLookupSessionKey)
		cancel()
		if err != nil {
			logging.ComponentWarnEvent("extproc", "rayline_routes_lookup_session_pin_failed", map[string]interface{}{
				"request_id": arcContext.RequestID, "stage": "read",
			})
		}
		pinned = session
	}
	return backend.BindLookup(pinned)
}

// pinPolicyLookupSession pins the session a decided lookup ended on, for the
// scope's idle timeout from now, and ends a pinned session it moved off once
// the new one is pinned. The scope's session is never episode state.
//
// The pin changes only from the session this lookup read: a concurrent
// lookup that already pinned another session (it moved first, or started
// the scope's first session) keeps its pin, and a lookup still finishing on
// a retired session cannot restore it. A session this lookup started that
// it did not pin is ended, so it does not hold a container's slot until its
// idle timeout; the session it read is left to whoever pinned past it.
//
// It runs off the answer's path: the episode store's Redis client does not
// bound a call by its context, so a stalled write would otherwise hold an
// already-decided answer past the lookup deadline.
func pinPolicyLookupSession(
	ctx context.Context,
	arcContext *selection.RaylineARCSelectionContext,
	binding *raylinearc.PolicySessionBinding,
	previous *raylinearc.PolicySession,
	idle time.Duration,
) {
	session, _ := binding.Session()
	pins := arcContext.PolicyLookupSessions
	if session == nil || pins == nil {
		return
	}
	key, requestID := arcContext.PolicyLookupSessionKey, arcContext.RequestID
	detached := context.WithoutCancel(ctx)
	go func() {
		writeCtx, cancel := context.WithTimeout(detached, raylineRoutesLookupSessionPinTimeout)
		pinned, err := pins.StoreLookupSession(writeCtx, key, previous, *session, idle)
		cancel()
		if err != nil {
			// A failed write may still have landed (its reply was lost): the
			// pin says which, so a session other lookups may already be
			// routed to is never ended.
			readCtx, cancel := context.WithTimeout(detached, raylineRoutesLookupSessionPinTimeout)
			current, readErr := pins.LoadLookupSession(readCtx, key)
			cancel()
			pinned = readErr == nil && current != nil && *current == *session
		}
		if !pinned {
			stage := "superseded"
			if err != nil {
				stage = "write"
			}
			logging.ComponentWarnEvent("extproc", "rayline_routes_lookup_session_pin_failed", map[string]interface{}{
				"request_id": requestID, "stage": stage,
			})
			binding.EndUnrecorded(detached)
			return
		}
		binding.TerminateReplaced(detached)
	}()
}

// addRaylineRoutesEncodeFields records how a lookup's encode ran on its
// routing_decision event, so a cell's lookup encode cost reads the same way
// before and after lookup sessions: lookup_session, the
// encoder's session_action and the encode's milliseconds.
func addRaylineRoutesEncodeFields(fields map[string]interface{}, encode routerruntime.RouteEncode) {
	fields["lookup_session"] = encode.LookupSession
	if encode.SessionAction != "" {
		fields["session_action"] = encode.SessionAction
	}
	if encode.Latency > 0 {
		fields["encode_ms"] = encode.Latency.Milliseconds()
	}
}
