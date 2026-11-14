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

// Lookup sessions (routes_api.lookup_sessions).
//
// A throwaway lookup -- one that carries routing.execution, caller prices,
// exclusive or exploration, or any lookup on a cell with episode_writes off
// -- decides under a random episode id, so its encode always starts from
// token 0: about 2 s at 100-200k tokens, for a conversation whose previous
// lookup encoded the same prefix moments before. When the caller named a
// conversation, the lookup instead decides on that conversation's lookup
// scope: a stable episode id the policy service keeps one warm encoder
// session for, appended to by each lookup. The service records nothing on
// it (session_scope "lookup": no turn, no held arm, no staging), so the
// lookup still commits nothing anywhere, and the scope's busy lock is its
// own, never the conversation's routed episode's.
//
// Every way the scope cannot be served falls back to the one-shot lookup
// within the same deadline: a service that does not advertise it, a call the
// service classifies as a side call, a private (zdr) turn, a concurrent
// lookup on the same scope (session_busy) and the service refusing the
// scope. A lookup is never refused because of this path.

// raylineRoutesLookupScopePrefix namespaces a lookup scope's episode
// identity, so it can never hash to a routed conversation's episode.
const raylineRoutesLookupScopePrefix = "lookup:"

// arcFailureRoutesLookupSessionFallback is the selection's answer when a
// lookup-scoped decision cannot be served on the scope. It is never a
// lookup's outcome: the decision service decides the lookup again, one-shot.
const arcFailureRoutesLookupSessionFallback = "routes_lookup_session_fallback"

// raylineRoutesLookupScope is the lookup scope a lookup may decide on, or ""
// when it may not: the cell has lookup_sessions off, or the caller named no
// conversation. The authenticated user is folded in although the gateway's
// conversation identity already carries it, so two users can never share a
// scope even if a caller's identity does not.
func raylineRoutesLookupScope(settings *config.RaylineARCRoutesAPIConfig, authzUserID, episode string) string {
	if settings == nil || !settings.LookupSessions || episode == "" {
		return ""
	}
	return raylineRoutesLookupScopePrefix + authzUserID + ":" + episode
}

// raylineRoutesLookupSessionEligible reports whether a decision may be tried
// on its lookup scope, from what the request and the cell say before the
// selection runs. The service's capability and the call's materialized kind
// are checked by the selection itself (policyLookupSessionUnservable).
func raylineRoutesLookupSessionEligible(
	request routerruntime.RouteDecisionRequest,
	algorithm *config.AlgorithmConfig,
) bool {
	return request.LookupScope != "" && request.Ephemeral &&
		request.Surface == routerruntime.RouteDecisionSurfaceRoutes &&
		algorithm != nil && algorithm.RaylineARC != nil && algorithm.RaylineARC.PolicyService != nil
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
	arcContext.EpisodeIDHash = raylinearc.HashEpisodeID(scope)
	arcContext.RoutesLookupSession = true
	return true
}

// policyLookupSessionUnservable names why a lookup-scoped decision cannot be
// sent on its scope, or "" when it can.
func (scorer *policyServiceScorer) policyLookupSessionUnservable(
	arcContext *selection.RaylineARCSelectionContext,
	sideCall bool,
) string {
	switch {
	case !scorer.lookupSessionsServable:
		return "replica_pool"
	case !scorer.advertises(raylinearc.PolicyCapabilityLookupSessions):
		// An older service ignores session_scope and would run the call as a
		// strict episode on the scope, holding its arm for the next lookup.
		return "capability_unadvertised"
	case sideCall:
		return "side_call"
	case arcContext.EphemeralTurn:
		return "ephemeral_turn"
	}
	return ""
}

// policyLookupSessionBusyWait is how long a decision retries session_busy:
// not at all on a lookup scope, whose only competitor is a concurrent lookup
// on the same conversation, and which falls back to a one-shot lookup at
// once rather than spend its deadline waiting.
func policyLookupSessionBusyWait(scorer *policyServiceScorer, arcContext *selection.RaylineARCSelectionContext) time.Duration {
	if arcContext.RoutesLookupSession {
		return 0
	}
	return scorer.busyWait
}

// policyLookupSessionRefused reports whether a lookup-scoped decide's failure
// is the scope's own: a concurrent lookup holds it, or the service refused
// session_scope. Any other failure is the lookup's, as on the one-shot path.
func policyLookupSessionRefused(err error) (string, bool) {
	var failure *raylinearc.PolicyServiceError
	if !errors.As(err, &failure) {
		return "", false
	}
	switch {
	case failure.Class == "session_busy":
		return "session_busy", true
	case failure.Class == "invalid_request" && failure.Detail["field"] == raylinearc.PolicySessionScopeField:
		return "scope_refused", true
	}
	return "", false
}

// raylineRoutesLookupSessionFallback logs why a lookup left its scope and
// returns the failure the decision service decides it again on.
func raylineRoutesLookupSessionFallback(
	arcContext *selection.RaylineARCSelectionContext,
	reason string,
) error {
	logging.ComponentEvent("extproc", "rayline_routes_lookup_session_fallback", map[string]interface{}{
		"request_id": arcContext.RequestID,
		"reason":     reason,
	})
	return &raylineARCSelectionFailure{class: arcFailureRoutesLookupSessionFallback}
}

// raylineRoutesCallerWaits reports whether the lookup's caller still waits:
// neither the decision's context nor the caller's own deadline has passed.
// An ephemeral lookup that holds an admission slot decides on a context
// detached from its caller, so the caller's deadline travels on the request.
func raylineRoutesCallerWaits(ctx context.Context, request routerruntime.RouteDecisionRequest) bool {
	if ctx.Err() != nil {
		return false
	}
	return request.CallerDeadline.IsZero() || time.Now().Before(request.CallerDeadline)
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
// it could not pin -- it lost that race, or the write failed -- is ended, so
// it does not hold a container's slot until its idle timeout; the session
// it read is left to whoever pinned past it. The write is bounded by the
// lookup's own context, so it never holds the answer past the deadline.
func pinPolicyLookupSession(
	ctx context.Context,
	arcContext *selection.RaylineARCSelectionContext,
	binding *raylinearc.PolicySessionBinding,
	previous *raylinearc.PolicySession,
	idle time.Duration,
) {
	session, _ := binding.Session()
	if session == nil || arcContext.PolicyLookupSessions == nil {
		return
	}
	writeCtx, cancel := context.WithTimeout(ctx, raylineRoutesLookupSessionPinTimeout)
	pinned, err := arcContext.PolicyLookupSessions.StoreLookupSession(
		writeCtx, arcContext.PolicyLookupSessionKey, previous, *session, idle)
	cancel()
	if err != nil || !pinned {
		stage := "superseded"
		if err != nil {
			stage = "write"
		}
		logging.ComponentWarnEvent("extproc", "rayline_routes_lookup_session_pin_failed", map[string]interface{}{
			"request_id": arcContext.RequestID, "stage": stage,
		})
		go binding.EndUnrecorded(context.WithoutCancel(ctx))
		return
	}
	binding.TerminateReplaced(ctx)
}

// isRaylineRoutesLookupSessionFallback reports whether a decision failed only
// because its lookup scope could not be served.
func isRaylineRoutesLookupSessionFallback(err error) bool {
	var failure *raylineARCSelectionFailure
	return errors.As(err, &failure) && failure.class == arcFailureRoutesLookupSessionFallback
}

// addRaylineRoutesEncodeFields records how a lookup's encode ran on its
// routing_decision event, so a cell's lookup encode cost reads the same way
// before and after lookup_sessions is turned on: lookup_session, the
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
