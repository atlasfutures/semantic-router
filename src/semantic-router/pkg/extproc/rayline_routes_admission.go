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
	"math"
	"sort"
	"strings"
	"sync"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineRoutesAdmissionCode is error.code on a lookup the admission gate
// shed, so a caller can tell it from a contended lease ("contended") and a
// lookup that ran out of time ("deadline_exceeded").
const raylineRoutesAdmissionCode = "lookup_capacity"

// buildRaylineRoutesAdmission builds one lookup admission gate per recipe
// whose routes-enabled ARC decision configures routes_api.admission.
//
// Only the lookup handler reads these gates. A dispatched turn never passes
// through them: the cap exists to keep lookups from occupying the encoders
// live turns need, not to slow those turns down.
func buildRaylineRoutesAdmission(router *OpenAIRouter) map[config.RecipeName]*raylinearc.RouteLookupGate {
	if router == nil || router.Config == nil {
		return nil
	}
	var gates map[config.RecipeName]*raylinearc.RouteLookupGate
	poolLimits := map[string]int{}
	add := func(name config.RecipeName, recipe *config.RoutingRecipe, store raylinearc.EpisodeStore) {
		decision := router.raylineRoutesDecision(recipe)
		if decision == nil || !decision.Algorithm.RaylineARC.RoutesAPI.Admission.Enabled() {
			return
		}
		arcConfig := decision.Algorithm.RaylineARC
		settings := arcConfig.RoutesAPI
		pool := raylineRoutesAdmissionPool(arcConfig)
		gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
			MaxInflight:    settings.Admission.MaxInflight,
			QueueWait:      settings.EffectiveAdmissionQueueWait(),
			LookupDeadline: settings.EffectiveDeadline(),
			Pool:           pool,
			Store:          store,
		})
		recipeLabel := raylineARCRecipeLabel(name)
		// Recipes on one backend share its slots only at one limit; at two
		// they each count against their own local cap.
		if previous, seen := poolLimits[pool]; seen && previous != gate.Limit() {
			logging.ComponentWarnEvent("extproc", "rayline_routes_admission_limit_conflict", map[string]interface{}{
				"recipe": recipeLabel, "max_inflight": gate.Limit(), "other_max_inflight": previous,
			})
		}
		poolLimits[pool] = gate.Limit()
		gate.SetFailOpenObserver(func(err error) {
			logging.ComponentWarnEvent("extproc", "rayline_routes_admission_fail_open", map[string]interface{}{
				"recipe": recipeLabel,
				"error":  err.Error(),
			})
		})
		fields := map[string]interface{}{
			"recipe":        recipeLabel,
			"max_inflight":  gate.Limit(),
			"queue_wait_ms": gate.QueueWait().Milliseconds(),
			"scope":         "cell",
			"backend":       raylinearc.RouteLookupBackendRedis,
		}
		if !gate.Distributed() {
			// Said at startup, because the difference matters: with no
			// shared store each router instance admits max_inflight on its
			// own, so the cell admits that many per instance.
			fields["scope"] = "process"
			fields["backend"] = raylinearc.RouteLookupBackendLocal
			fields["reason"] = "episode_backend_not_redis"
		}
		logging.ComponentEvent("extproc", "rayline_routes_admission", fields)
		if gates == nil {
			gates = map[config.RecipeName]*raylinearc.RouteLookupGate{}
		}
		gates[name] = gate
	}
	add(config.DefaultRecipeName, nil, router.RaylineARCEpisodeStore)
	for index := range router.Config.Recipes {
		recipe := &router.Config.Recipes[index]
		if recipe.Name == config.DefaultRecipeName {
			continue
		}
		add(recipe.Name, recipe, router.RaylineARCRecipeEpisodeStores[recipe.Name])
	}
	return gates
}

// raylineRoutesAdmissionPool names the backend a decision's lookups occupy:
// the policy service, or the encoder in the artifact mode. Two recipes that
// call the same backend compute the same pool and so share its slots.
//
// The identity is the backend kind and its endpoint set, normalized, sorted
// and de-duplicated, so base_url: A and replicas: [A] are one pool. An
// encoder fleet with dynamic membership has no static endpoints; its
// identity is the membership document it reads, which lives under the
// episode key prefix.
func raylineRoutesAdmissionPool(arcConfig *config.RaylineARCAlgorithmConfig) string {
	kind, base, replicas := "encoder", arcConfig.Encoder.BaseURL, arcConfig.Encoder.Replicas
	if policy := arcConfig.PolicyService; policy != nil {
		kind, base, replicas = "policy_service", policy.BaseURL, policy.Replicas
	}
	endpoints := map[string]bool{}
	add := func(url string) {
		if url = strings.TrimRight(strings.TrimSpace(url), "/"); url != "" {
			endpoints[url] = true
		}
	}
	add(base)
	for _, replica := range replicas {
		add(replica.BaseURL)
	}
	identity := []string{kind}
	if len(endpoints) == 0 {
		identity = append(identity, "membership", arcConfig.Episode.KeyPrefix)
	}
	urls := make([]string, 0, len(endpoints))
	for url := range endpoints {
		urls = append(urls, url)
	}
	sort.Strings(urls)
	return raylinearc.RouteLookupPoolID(append(identity, urls...)...)
}

// raylineRoutesAdmissionGate is the gate for a lookup's recipe, or nil when
// that recipe admits lookups without one.
func (r *OpenAIRouter) raylineRoutesAdmissionGate(recipe *config.RoutingRecipe) *raylinearc.RouteLookupGate {
	if r == nil || len(r.raylineRoutesAdmission) == 0 {
		return nil
	}
	name := config.DefaultRecipeName
	if recipe != nil && recipe.Name != "" {
		name = recipe.Name
	}
	return r.raylineRoutesAdmission[name]
}

// admitRaylineRoutesLookup takes the lookup's admission slot. It returns the
// ticket to release when the lookup answers, or the response to answer
// instead: 429 lookup_capacity when no slot freed within the queue wait.
// A wait the lookup's own context ended is left to raylineRoutesFailure,
// which answers a deadline as 504 like any other.
func (r *OpenAIRouter) admitRaylineRoutesLookup(
	ctx *RequestContext,
	lookupContext context.Context,
	recipe *config.RoutingRecipe,
	routeID string,
) (raylinearc.RouteLookupTicket, *ext_proc.ProcessingResponse) {
	gate := r.raylineRoutesAdmissionGate(recipe)
	if gate == nil {
		return raylinearc.RouteLookupTicket{}, nil
	}
	ticket, err := acquireRaylineRoutesLookup(lookupContext, gate, routeID)
	if err != nil {
		return raylinearc.RouteLookupTicket{}, r.raylineRoutesAdmissionRefusal(ctx, lookupContext, routeID, gate, err)
	}
	return ticket, nil
}

// acquireRaylineRoutesLookup takes a slot on gate and records the outcome.
func acquireRaylineRoutesLookup(
	lookupContext context.Context,
	gate *raylinearc.RouteLookupGate,
	routeID string,
) (raylinearc.RouteLookupTicket, error) {
	ticket, err := gate.Acquire(lookupContext)
	if err == nil {
		metrics.RecordRaylineARCRoutesAdmission(ticket.Outcome, ticket.Backend, ticket.Wait.Seconds())
		if ticket.Outcome != raylinearc.RouteLookupAdmitted {
			logging.ComponentEvent("extproc", "rayline_routes_admission_wait", map[string]interface{}{
				"routing_decision_id": routeID,
				"outcome":             ticket.Outcome,
				"backend":             ticket.Backend,
				"wait_ms":             ticket.Wait.Milliseconds(),
			})
		}
		return ticket, nil
	}
	var refusal *raylinearc.RouteLookupRefusal
	if errors.As(err, &refusal) {
		metrics.RecordRaylineARCRoutesAdmission(refusal.Outcome, refusal.Backend, refusal.Wait.Seconds())
		logging.ComponentEvent("extproc", "rayline_routes_admission_wait", map[string]interface{}{
			"routing_decision_id": routeID,
			"outcome":             refusal.Outcome,
			"backend":             refusal.Backend,
			"wait_ms":             refusal.Wait.Milliseconds(),
		})
	}
	return raylinearc.RouteLookupTicket{}, err
}

// raylineRoutesAdmissionRefusal answers a lookup the gate did not admit: 429
// lookup_capacity for a shed, raylineRoutesFailure for anything else (a wait
// the lookup's deadline ended answers 504).
func (r *OpenAIRouter) raylineRoutesAdmissionRefusal(
	ctx *RequestContext,
	lookupContext context.Context,
	routeID string,
	gate *raylinearc.RouteLookupGate,
	err error,
) *ext_proc.ProcessingResponse {
	if !errors.Is(err, raylinearc.ErrRouteLookupShed) {
		return r.raylineRoutesFailure(ctx, lookupContext, routeID, err)
	}
	return raylineRoutesRetryAfter(
		r.createRaylineRoutesCodedError(ctx, &routesContractError{
			status:  429,
			errType: "rate_limit_error",
			code:    raylineRoutesAdmissionCode,
			message: "route lookup capacity is in use: retry after the Retry-After interval",
		}),
		raylineRoutesAdmissionRetrySeconds(gate),
	)
}

// raylineRoutesSlot is a gated lookup's admission slot and the hold its
// decision keeps on it.
//
// The slot is the backend's, and the backend does not stop encoding when the
// caller stops waiting. So once a decision may encode on the slot it is
// pinned: from then on the decision runs on its own context, bounded by
// RouteLookupDetachBound, and the slot is released when the backend answers,
// not at the caller's 504 (raylineRoutesDecide). An ephemeral lookup is
// pinned from the start, as it took its slot up front. A tracked lookup is
// pinned in admit, once its episode is read and before any encoder call; up
// to then its decision is cancelled with the caller, so an abandoned lookup
// starts no encode, and its commit stays on the caller's own context
// (RouteDecisionRequest.Caller), so an abandoned one never advances its
// episode.
//
// With appendExempt (routes_api.admission.append_exempt) a tracked lookup
// takes no slot up front: admit takes one, inside the caller's deadline,
// unless the decision is predicted to append at most maxSuffixTokens to the
// episode's retained session, in which case it holds none and is not pinned.
type raylineRoutesSlot struct {
	gate            *raylinearc.RouteLookupGate
	routeID         string
	lookup          context.Context
	appendExempt    bool
	maxSuffixTokens int

	mu     sync.Mutex
	ticket raylinearc.RouteLookupTicket
	// refused is the gate's error when admit did not admit the lookup, so
	// the handler can tell the decision's admission failure from every other.
	refused   error
	pinned    bool
	abandoned bool
	cancel    context.CancelFunc
}

// admit is the lookup's RouteDecisionRequest.Admit.
func (slot *raylineRoutesSlot) admit(
	_ context.Context,
	predict func() routerruntime.RouteEncodePrediction,
) (bool, error) {
	slot.mu.Lock()
	held := slot.ticket.Held()
	slot.mu.Unlock()
	if !held {
		if prediction := predict(); slot.appendExempt && prediction.Append && prediction.SuffixTokens <= slot.maxSuffixTokens {
			metrics.RecordRaylineARCRoutesAdmission(raylinearc.RouteLookupExempt, raylinearc.RouteLookupBackendNone, 0)
			return true, nil
		}
		// The wait is the caller's: it is spent inside its deadline.
		ticket, err := acquireRaylineRoutesLookup(slot.lookup, slot.gate, slot.routeID)
		slot.mu.Lock()
		defer slot.mu.Unlock()
		if err != nil {
			slot.refused = err
			return false, err
		}
		slot.ticket = ticket
		return false, slot.pinLocked()
	}
	slot.mu.Lock()
	defer slot.mu.Unlock()
	return false, slot.pinLocked()
}

// pinLocked pins the decision to the slot, unless the caller has already
// left, in which case the decision is cancelled and starts no encode. The
// caller holds mu.
func (slot *raylineRoutesSlot) pinLocked() error {
	if slot.abandoned {
		return slot.lookup.Err()
	}
	slot.pinned = true
	return nil
}

// start arms the hold for a decision about to run on its own context:
// cancel ends that context; pinned is whether it is pinned from the start.
func (slot *raylineRoutesSlot) start(cancel context.CancelFunc, pinned bool) {
	slot.mu.Lock()
	defer slot.mu.Unlock()
	slot.cancel, slot.pinned = cancel, pinned
}

// abandon is the caller leaving: an unpinned decision is cancelled with it.
func (slot *raylineRoutesSlot) abandon() {
	slot.mu.Lock()
	defer slot.mu.Unlock()
	if slot.pinned {
		return
	}
	slot.abandoned = true
	if slot.cancel != nil {
		slot.cancel()
	}
}

// release frees the slot, if one is held. Safe to call more than once.
func (slot *raylineRoutesSlot) release() {
	if slot == nil {
		return
	}
	slot.mu.Lock()
	ticket := slot.ticket
	slot.ticket = raylinearc.RouteLookupTicket{}
	slot.mu.Unlock()
	ticket.Release()
}

// refusedBy reports whether err is the gate's refusal of this lookup.
func (slot *raylineRoutesSlot) refusedBy(err error) bool {
	if slot == nil || err == nil {
		return false
	}
	slot.mu.Lock()
	refused := slot.refused
	slot.mu.Unlock()
	return refused != nil && errors.Is(err, refused)
}

// raylineRoutesAdmissionFailureClass is the episode abort class of a tracked
// lookup the gate refused after the episode lease was taken.
const raylineRoutesAdmissionFailureClass = "routes_admission"

// raylineRoutesAbandonedFailureClass is the episode abort class of a tracked
// lookup whose decision ran on after its caller left (it held a slot), and
// which is therefore released rather than committed.
const raylineRoutesAbandonedFailureClass = "routes_abandoned"

// raylineRoutesAdmissionRecord is a predicted lookup's admission, logged with
// the session action its decision reported once it decided.
type raylineRoutesAdmissionRecord struct {
	decisionID string
	prediction routerruntime.RouteEncodePrediction
	exempt     bool
}

// log records whether the lookup was gated or exempt, what was predicted and
// what the service actually did, so an exempt lookup that turned out to be a
// full encode (session_action created or rebuilt) is counted.
func (record *raylineRoutesAdmissionRecord) log(result *selection.SelectionResult, err error) {
	if record == nil {
		return
	}
	sessionAction := ""
	if err == nil && result != nil && result.RaylineARC != nil {
		sessionAction = boundedPolicySessionAction(result.RaylineARC.SessionAction)
	}
	if sessionAction == "" {
		sessionAction = "none"
	}
	admission := "gated"
	if record.exempt {
		admission = raylinearc.RouteLookupExempt
	}
	metrics.RecordRaylineARCRoutesAdmissionPrediction(admission, record.prediction.Reason, sessionAction)
	logging.ComponentEvent("extproc", "rayline_routes_admission_prediction", map[string]interface{}{
		"routing_decision_id":     record.decisionID,
		"admission":               admission,
		"prediction":              record.prediction.Reason,
		"predicted_append":        record.prediction.Append,
		"predicted_suffix_tokens": record.prediction.SuffixTokens,
		"session_action":          sessionAction,
		"decided":                 err == nil,
	})
}

// raylineRoutesAdmissionRetrySeconds is the Retry-After on a shed lookup:
// the queue wait it already spent, rounded up, and at least a second. The
// slot it missed is held until the backend answers a lookup that may encode
// for its whole deadline or longer, so coming back sooner than the wait that
// just failed mostly fails again.
func raylineRoutesAdmissionRetrySeconds(gate *raylinearc.RouteLookupGate) int {
	seconds := int(math.Ceil(gate.QueueWait().Seconds()))
	return max(seconds, raylineRoutesContendedRetrySeconds)
}
