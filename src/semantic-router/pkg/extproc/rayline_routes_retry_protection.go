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
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/proto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Retry-storm protection for /v1/routes, in front of the lane
// (rayline_routes_admission.go):
//
//   - the refusal cache answers a caller's repeat of a request the decision
//     refused with a deterministic 4xx, from memory;
//   - the caller budget holds each caller to lookups per minute and at once;
//   - the lane caps lookups cell-wide, so they cannot take the session slots
//     live turns need.
//
// And on the way out, every non-2xx this cell answers says whether and when to
// retry (stampRaylineRetryHints).

// raylineRoutesCallerBudgetCode is error.code on a lookup its caller had no
// budget left for.
const raylineRoutesCallerBudgetCode = "caller_budget"

const (
	// raylineRetryAfterHeader and raylineShouldRetryHeader are the retry
	// hints. x-should-retry is the header the Anthropic and OpenAI SDKs read
	// before their own status-based retry decision.
	raylineRetryAfterHeader  = "retry-after"
	raylineShouldRetryHeader = "x-should-retry"
	// raylineRefusalCacheHeader marks an answer replayed from the refusal
	// cache, so a fault test (and a support read) can tell it from a fresh
	// decision.
	raylineRefusalCacheHeader = "x-rayline-refusal-cache"
	// raylineCellUnavailableRetrySeconds is the Retry-After on a cell 503 or
	// other 5xx that set none. An unavailable decision backend is either
	// starting (a cold container takes seconds) or replacing one that
	// wedged; a client told to come back at once lands on the same state.
	raylineCellUnavailableRetrySeconds = 5
)

// raylineRoutesCallerIdentity is the caller a lookup's budget and refusal
// cache are keyed on: the conversation it names (x-rayline-session, with
// x-rayline-branch), the identity the cell already keys episode leases on.
// The gateway composes it from the user and the conversation, so a client
// retrying one turn sends the same caller every time. Empty when the lookup
// names no conversation.
func raylineRoutesCallerIdentity(ctx *RequestContext) string {
	return raylineRoutesEpisodeIdentity(ctx)
}

// buildRaylineRoutesCallerBudgets builds one caller budget per recipe that
// serves lookups, unless it sets routes_api.caller_budget.disabled.
func buildRaylineRoutesCallerBudgets(router *OpenAIRouter) map[config.RecipeName]*raylinearc.RouteCallerBudget {
	var budgets map[config.RecipeName]*raylinearc.RouteCallerBudget
	forEachRaylineRoutesRecipe(router, func(name config.RecipeName, decision *config.Decision, store raylinearc.EpisodeStore) {
		arcConfig := decision.Algorithm.RaylineARC
		settings := arcConfig.RoutesAPI
		budget := raylinearc.NewRouteCallerBudget(raylinearc.RouteCallerBudgetConfig{
			LookupsPerMinute: settings.CallerBudget.EffectiveLookupsPerMinute(),
			MaxInflight:      settings.CallerBudget.EffectiveMaxInflight(),
			LookupDeadline:   settings.EffectiveDeadline(),
			Pool:             raylineRoutesAdmissionPool(arcConfig),
			Store:            store,
		})
		if budget == nil {
			return
		}
		recipeLabel := raylineARCRecipeLabel(name)
		budget.SetFailOpenObserver(func(err error) {
			logging.ComponentWarnEvent("extproc", "rayline_routes_caller_budget_fail_open", map[string]interface{}{
				"recipe": recipeLabel,
				"error":  err.Error(),
			})
		})
		scope, backend := "cell", raylinearc.RouteLookupBackendRedis
		if !budget.Distributed() {
			scope, backend = "process", raylinearc.RouteLookupBackendLocal
		}
		logging.ComponentEvent("extproc", "rayline_routes_caller_budget", map[string]interface{}{
			"recipe":             recipeLabel,
			"lookups_per_minute": budget.LookupsPerMinute(),
			"max_inflight":       budget.MaxInflight(),
			"scope":              scope,
			"backend":            backend,
		})
		if budgets == nil {
			budgets = map[config.RecipeName]*raylinearc.RouteCallerBudget{}
		}
		budgets[name] = budget
	})
	return budgets
}

// admitRaylineRoutesCaller counts the lookup against its caller. It returns
// the ticket to release when the backend answers, or the response to answer
// instead: 429 caller_budget with Retry-After.
func (r *OpenAIRouter) admitRaylineRoutesCaller(
	ctx *RequestContext,
	lookupContext context.Context,
	recipe *config.RoutingRecipe,
	routeID string,
) (raylinearc.RouteCallerTicket, *ext_proc.ProcessingResponse) {
	budget := r.raylineRoutesCallerBudgets[raylineRoutesRecipeName(recipe)]
	if budget == nil {
		return raylinearc.RouteCallerTicket{}, nil
	}
	ticket, err := budget.Acquire(lookupContext, raylineRoutesCallerIdentity(ctx))
	if err == nil {
		metrics.RecordRaylineARCRoutesCallerBudget(ticket.Outcome, ticket.Backend)
		return ticket, nil
	}
	var refusal *raylinearc.RouteCallerRefusal
	if !errors.As(err, &refusal) {
		// Acquire refuses only with a refusal; anything else admits.
		return raylinearc.RouteCallerTicket{}, nil
	}
	metrics.RecordRaylineARCRoutesCallerBudget(refusal.Outcome, refusal.Backend)
	retrySeconds := max(int(math.Ceil(refusal.RetryAfter.Seconds())), 1)
	logging.ComponentEvent("extproc", "rayline_routes_caller_budget_refused", map[string]interface{}{
		"routing_decision_id": routeID,
		"outcome":             refusal.Outcome,
		"backend":             refusal.Backend,
		"retry_after_s":       retrySeconds,
	})
	message := "route lookup rate for this caller is over budget: retry after the Retry-After interval"
	if refusal.Outcome == raylinearc.RouteCallerInflightLimited {
		message = "this caller already has the most route lookups in flight: retry after the Retry-After interval"
	}
	return raylinearc.RouteCallerTicket{}, raylineRoutesRetryAfter(
		r.createRaylineRoutesCodedError(ctx, &routesContractError{
			status: 429, errType: "rate_limit_error", code: raylineRoutesCallerBudgetCode, message: message,
		}),
		retrySeconds,
	)
}

func raylineRoutesRecipeName(recipe *config.RoutingRecipe) config.RecipeName {
	if recipe != nil && recipe.Name != "" {
		return recipe.Name
	}
	return config.DefaultRecipeName
}

// raylineRoutesRefusalCache remembers, per process, the deterministic
// refusals the decision gave, keyed by caller and request digest. Bounded by
// count (oldest dropped first) and by each entry's TTL.
type raylineRoutesRefusalCache struct {
	mu      sync.Mutex
	max     int
	order   *list.List
	entries map[[sha256.Size]byte]*list.Element
	now     func() time.Time
}

type raylineRoutesRefusalEntry struct {
	key      [sha256.Size]byte
	expires  time.Time
	response *ext_proc.ProcessingResponse
}

func newRaylineRoutesRefusalCache(maxEntries int) *raylineRoutesRefusalCache {
	if maxEntries <= 0 {
		return nil
	}
	return &raylineRoutesRefusalCache{
		max:     maxEntries,
		order:   list.New(),
		entries: map[[sha256.Size]byte]*list.Element{},
		now:     time.Now,
	}
}

// buildRaylineRoutesRefusalCache sizes one process cache for every recipe
// that keeps one, at the largest bound any of them asks for.
func buildRaylineRoutesRefusalCache(router *OpenAIRouter) *raylineRoutesRefusalCache {
	maxEntries := 0
	forEachRaylineRoutesRecipe(router, func(_ config.RecipeName, decision *config.Decision, _ raylinearc.EpisodeStore) {
		maxEntries = max(maxEntries, decision.Algorithm.RaylineARC.RoutesAPI.RefusalCache.EffectiveMaxEntries())
	})
	return newRaylineRoutesRefusalCache(maxEntries)
}

// lookup returns a copy of the remembered refusal for key, or nil.
func (cache *raylineRoutesRefusalCache) lookup(key [sha256.Size]byte) *ext_proc.ProcessingResponse {
	if cache == nil {
		return nil
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	element, ok := cache.entries[key]
	if !ok {
		return nil
	}
	entry := element.Value.(*raylineRoutesRefusalEntry)
	if !cache.now().Before(entry.expires) {
		cache.order.Remove(element)
		delete(cache.entries, key)
		return nil
	}
	return proto.Clone(entry.response).(*ext_proc.ProcessingResponse)
}

// store remembers response for key for ttl.
func (cache *raylineRoutesRefusalCache) store(key [sha256.Size]byte, response *ext_proc.ProcessingResponse, ttl time.Duration) {
	if cache == nil || ttl <= 0 {
		return
	}
	entry := &raylineRoutesRefusalEntry{
		key:      key,
		expires:  cache.now().Add(ttl),
		response: proto.Clone(response).(*ext_proc.ProcessingResponse),
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if element, ok := cache.entries[key]; ok {
		cache.order.Remove(element)
	}
	cache.entries[key] = cache.order.PushFront(entry)
	for cache.order.Len() > cache.max {
		oldest := cache.order.Back()
		cache.order.Remove(oldest)
		delete(cache.entries, oldest.Value.(*raylineRoutesRefusalEntry).key)
		metrics.RecordRaylineARCRoutesRefusalCache("evicted")
	}
}

// raylineRoutesRefusalCacheable reports whether a decision refusal is the
// same every time the same request is sent: a 4xx, except the statuses that
// mean "not now" (408 timeout, 409 conflict, 425 too early, 429 capacity).
// A 5xx never is.
func raylineRoutesRefusalCacheable(status int) bool {
	if status < 400 || status >= 500 {
		return false
	}
	switch status {
	case 408, 409, 425, 429:
		return false
	}
	return true
}

// raylineRoutesRefusalKey digests what decides a lookup's answer: the
// caller, the request path, the body as sent, and the routing headers (every
// x-rayline-* header but the per-request route id, and the fault header). A
// header like x-request-id that differs on each retry is left out, so a
// retry of one request lands on one key; any header that can change the
// decision is in, so two requests that may be decided differently never
// share one.
func raylineRoutesRefusalKey(ctx *RequestContext, caller string, body []byte) [sha256.Size]byte {
	digest := sha256.New()
	write := func(value string) {
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(value)))
		digest.Write(length[:])
		digest.Write([]byte(value))
	}
	write(caller)
	write(normalizeRequestPath(requestPathFromContext(ctx)))
	if ctx != nil {
		names := make([]string, 0, len(ctx.Headers))
		for name := range ctx.Headers {
			lower := strings.ToLower(name)
			if lower == raylineRoutesRouteIDHeader {
				continue
			}
			if strings.HasPrefix(lower, "x-rayline-") || lower == headers.VSRFault {
				names = append(names, name)
			}
		}
		sort.Strings(names)
		for _, name := range names {
			write(strings.ToLower(name))
			write(strings.TrimSpace(ctx.Headers[name]))
		}
	}
	write(string(body))
	var key [sha256.Size]byte
	copy(key[:], digest.Sum(nil))
	return key
}

// replayRaylineRoutesRefusal answers a lookup from the refusal cache, or
// returns nil to decide it.
func (r *OpenAIRouter) replayRaylineRoutesRefusal(
	ctx *RequestContext,
	settings *config.RaylineARCRoutesAPIConfig,
	key [sha256.Size]byte,
	routeID string,
) *ext_proc.ProcessingResponse {
	if settings.RefusalCache.EffectiveTTL() <= 0 {
		return nil
	}
	response := r.raylineRoutesRefusals.lookup(key)
	if response == nil {
		return nil
	}
	metrics.RecordRaylineARCRoutesRefusalCache("hit")
	logging.ComponentEvent("extproc", "rayline_routes_refusal_replayed", map[string]interface{}{
		"routing_decision_id": routeID,
		"status":              response.GetImmediateResponse().GetStatus().GetCode(),
	})
	appendImmediateResponseHeader(response, raylineRefusalCacheHeader, "hit")
	if ctx != nil {
		ctx.ImmediateResponseEncoded = true
	}
	return response
}

// rememberRaylineRoutesRefusal stores a decision refusal that the same
// request would get again, and returns the response unchanged.
func (r *OpenAIRouter) rememberRaylineRoutesRefusal(
	settings *config.RaylineARCRoutesAPIConfig,
	key [sha256.Size]byte,
	response *ext_proc.ProcessingResponse,
) *ext_proc.ProcessingResponse {
	ttl := settings.RefusalCache.EffectiveTTL()
	status := int(response.GetImmediateResponse().GetStatus().GetCode())
	if ttl <= 0 || r.raylineRoutesRefusals == nil || !raylineRoutesRefusalCacheable(status) {
		return response
	}
	r.raylineRoutesRefusals.store(key, response, ttl)
	metrics.RecordRaylineARCRoutesRefusalCache("stored")
	return response
}

// stampRaylineRetryHints tells a client whether and when to retry the cell's
// own non-2xx answers, on the two surfaces a retrying client storms:
//
//   - /v1/routes: a 429, 503 or other 5xx carries Retry-After (integer
//     seconds); 429 and 503 also carry x-should-retry: true; any other 4xx
//     is refused the same way every time and carries x-should-retry: false.
//   - a live ARC turn: a 503 carries Retry-After.
//
// A header a producer already set is kept as it is.
func stampRaylineRetryHints(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	immediate := response.GetImmediateResponse()
	if immediate == nil || ctx == nil {
		return
	}
	status := int(immediate.GetStatus().GetCode())
	if status < 400 {
		return
	}
	surface := "turn"
	if isRaylineRoutesRequest(ctx) {
		surface = "routes"
	} else if ctx.RaylineARCIdentity == nil {
		return
	}
	retryAfter := 0
	shouldRetry := ""
	switch {
	case surface == "turn":
		if status == 503 {
			retryAfter = raylineCellUnavailableRetrySeconds
		}
	case status == 429:
		retryAfter, shouldRetry = raylineRoutesContendedRetrySeconds, "true"
	case status == 503:
		retryAfter, shouldRetry = raylineCellUnavailableRetrySeconds, "true"
	case status >= 500:
		retryAfter = raylineCellUnavailableRetrySeconds
	case !raylineRoutesRefusalCacheable(status):
		// 408, 409, 425: transient, but no wait to suggest.
	default:
		shouldRetry = "false"
	}
	if retryAfter > 0 && !immediateHasHeader(immediate, raylineRetryAfterHeader) {
		appendImmediateResponseHeader(response, raylineRetryAfterHeader, strconv.Itoa(retryAfter))
	}
	if shouldRetry != "" && !immediateHasHeader(immediate, raylineShouldRetryHeader) {
		appendImmediateResponseHeader(response, raylineShouldRetryHeader, shouldRetry)
	}
	metrics.RecordRaylineARCRetryHint(surface, strconv.Itoa(status),
		raylineRetryHintHeader(immediate, raylineShouldRetryHeader), immediateHasHeader(immediate, raylineRetryAfterHeader))
}

func immediateHasHeader(immediate *ext_proc.ImmediateResponse, name string) bool {
	return raylineRetryHintHeader(immediate, name) != ""
}

func raylineRetryHintHeader(immediate *ext_proc.ImmediateResponse, name string) string {
	for _, option := range immediate.GetHeaders().GetSetHeaders() {
		header := option.GetHeader()
		if strings.EqualFold(header.GetKey(), name) {
			if value := string(header.GetRawValue()); value != "" {
				return value
			}
			if header.GetValue() != "" {
				return header.GetValue()
			}
		}
	}
	return ""
}
