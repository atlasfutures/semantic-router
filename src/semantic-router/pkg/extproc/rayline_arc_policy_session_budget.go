package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A sessioned policy Server's sessions are shared by every caller, and one
// caller fanning out parallel requests (each a fork of a busy parent episode,
// each starting its own session) can exhaust the pool for everyone. The cell
// counts the new sessions each caller opens (sessions.max_new_sessions_per_caller
// per raylinearc.PolicySessionBudgetWindow, keyed on the gateway's
// x-rayline-caller) in the episode store, so every router instance shares the
// count. A turn past the budget is not refused: it is decided one-shot on the
// backend's shared overflow session (raylinearc.BindOverflow) and opens none.
//
// Only a turn whose episode holds no session is counted, so an existing
// session's turns are never affected; so is a turn that moves off a lost
// session (it replaces one, it does not add one). A private (zdr) turn and a
// lookup-scope lookup keep their own session rules and are never counted.

// Budget outcomes, as counted (llm_rayline_arc_policy_session_budget_total).
const (
	policySessionBudgetAdmitted    = "admitted"
	policySessionBudgetSessionless = "over_budget_sessionless"
	policySessionBudgetRedisError  = "redis_error"
)

// policySessionBudgetTimeout bounds the budget's store call: a slow store
// fails open rather than delay the turn.
const policySessionBudgetTimeout = 500 * time.Millisecond

// policySessionBudget is a cell's per-caller new-session budget.
type policySessionBudget struct {
	counter raylinearc.PolicySessionBudget
	limit   int
	window  time.Duration
}

// newPolicySessionBudget is the cell's budget: nil without sessions or with
// max_new_sessions_per_caller unset. It counts in the episode store when that
// is shared (Redis), and per process otherwise.
func newPolicySessionBudget(
	policy *config.RaylineARCPolicyServiceConfig,
	sessions *raylinearc.PolicySessionBackend,
	store raylinearc.EpisodeStore,
) *policySessionBudget {
	if policy == nil || sessions == nil || policy.Sessions.MaxNewSessionsPerCaller <= 0 {
		return nil
	}
	return &policySessionBudget{
		counter: policySessionBudgetCounterFor(store),
		limit:   policy.Sessions.MaxNewSessionsPerCaller,
		window:  raylinearc.PolicySessionBudgetWindow,
	}
}

// policySessionBudgetCounterFor finds the shared counter behind the store,
// through any wrapper, as the lookup pins do.
func policySessionBudgetCounterFor(store raylinearc.EpisodeStore) raylinearc.PolicySessionBudget {
	for store != nil {
		if shared, ok := store.(raylinearc.PolicySessionBudget); ok {
			return shared
		}
		wrapped, ok := store.(interface {
			UnwrapEpisodeStore() raylinearc.EpisodeStore
		})
		if !ok {
			break
		}
		store = wrapped.UnwrapEpisodeStore()
	}
	return raylinearc.NewMemoryPolicySessionBudget()
}

// overBudget counts a turn that would open a new session against its caller's
// budget and reports whether it is past it. A turn that opens none (its
// episode holds a session, a private turn, a lookup scope), a turn without a
// caller and a cell without a budget are never counted. A store failure fails
// open: the turn opens its session as it would without a budget.
func (budget *policySessionBudget) overBudget(
	ctx context.Context,
	arcContext *selection.RaylineARCSelectionContext,
	state *raylinearc.EpisodeState,
	binding *raylinearc.PolicySessionBinding,
	ephemeral bool,
) bool {
	if budget == nil || binding == nil || ephemeral || arcContext == nil ||
		arcContext.CallerKey == "" || arcContext.RoutesLookupSession ||
		state != nil && state.PolicySession != nil {
		return false
	}
	countCtx, cancel := context.WithTimeout(ctx, policySessionBudgetTimeout)
	admitted, err := budget.counter.AdmitPolicySessionStart(countCtx, arcContext.CallerKey, budget.limit, budget.window)
	cancel()
	switch {
	case err != nil:
		metrics.RecordRaylineARCPolicySessionBudget(policySessionBudgetRedisError)
		logging.ComponentWarnEvent("extproc", "rayline_arc_policy_session_budget_unavailable", map[string]interface{}{
			"request_id": arcContext.RequestID, "effect": "admitted",
		})
		return false
	case admitted:
		metrics.RecordRaylineARCPolicySessionBudget(policySessionBudgetAdmitted)
		return false
	}
	metrics.RecordRaylineARCPolicySessionBudget(policySessionBudgetSessionless)
	// Bounded fields only: never the caller id or its key.
	logging.ComponentWarnEvent("extproc", "rayline_arc_policy_session_over_budget", map[string]interface{}{
		"request_id":      arcContext.RequestID,
		"episode_id_hash": arcContext.EpisodeIDHash,
		"call_kind":       string(arcContext.PolicyCallKind),
		"limit":           budget.limit,
		"window_seconds":  int(budget.window / time.Second),
		"effect":          "sessionless",
	})
	return true
}
