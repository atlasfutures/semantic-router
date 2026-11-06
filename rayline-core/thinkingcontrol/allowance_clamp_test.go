package thinkingcontrol

import (
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func requestWithAllowance(allowance *int64) *llmprotocol.Request {
	request := &llmprotocol.Request{}
	request.Sampling.MaxOutputTokens = allowance
	return request
}

// A harness that states one session-wide allowance is lowered to the limit of
// the arm the turn landed on, and the number it stated is kept so a
// downstream bound is not derived from the limit.
func TestClampMessagesAllowanceLowersAStatedAllowance(t *testing.T) {
	request := requestWithAllowance(llmprotocol.Int64(128000))
	clamped, ok := ClampMessagesAllowance(request, nil, 24576)
	if !ok {
		t.Fatalf("clamp = false, want a clamp of 128000 to 24576")
	}
	if clamped.From != 128000 || clamped.To != 24576 {
		t.Fatalf("clamp = %d -> %d, want 128000 -> 24576", clamped.From, clamped.To)
	}
	if got := *request.Sampling.MaxOutputTokens; got != 24576 {
		t.Fatalf("max_output_tokens = %d, want the limit 24576", got)
	}
	if request.ClientMaxOutputTokens == nil || *request.ClientMaxOutputTokens != 128000 {
		t.Fatalf("client_max_output_tokens = %v, want the stated 128000", request.ClientMaxOutputTokens)
	}
	if !request.RouterSetMaxOutputTokens {
		t.Fatalf("router_set_max_output_tokens = false, want the substitution marked")
	}
}

// An allowance at or under the limit is the caller's and is not touched, so a
// cloud turn under its own ceiling is dispatched exactly as it was sent.
func TestClampMessagesAllowanceLeavesAnAllowanceWithinTheLimit(t *testing.T) {
	for _, allowance := range []int64{24576, 1000} {
		request := requestWithAllowance(llmprotocol.Int64(allowance))
		if _, ok := ClampMessagesAllowance(request, nil, 24576); ok {
			t.Fatalf("allowance %d: clamp = true, want untouched", allowance)
		}
		if got := *request.Sampling.MaxOutputTokens; got != allowance {
			t.Fatalf("allowance %d: max_output_tokens = %d, want it unchanged", allowance, got)
		}
		if request.RouterSetMaxOutputTokens || request.ClientMaxOutputTokens != nil {
			t.Fatalf("allowance %d: an untouched allowance must not be marked as the router's", allowance)
		}
	}
}

// A request that states no allowance is the deployment fill-in's to make, not
// the clamp's.
func TestClampMessagesAllowanceIgnoresARequestThatStatesNone(t *testing.T) {
	request := requestWithAllowance(nil)
	if _, ok := ClampMessagesAllowance(request, nil, 24576); ok {
		t.Fatalf("clamp = true, want no clamp on a request that states no allowance")
	}
	if request.Sampling.MaxOutputTokens != nil {
		t.Fatalf("max_output_tokens = %v, want it left unset", request.Sampling.MaxOutputTokens)
	}
}

// Anthropic refuses a thinking budget that is not strictly below max_tokens,
// and RaiseMessagesAllowance adds the budget on top of the allowance: a limit
// that cannot hold the admitted budget beside any output is not clamped into,
// so the turn keeps the deployment-limit refusal instead of being rewritten
// into a request the codec would refuse.
func TestClampMessagesAllowanceWillNotClampIntoTheThinkingBudget(t *testing.T) {
	for _, testCase := range []struct {
		name   string
		budget int64
		limit  int64
	}{
		{name: "budget equals the limit", budget: 24576, limit: 24576},
		{name: "budget over the limit", budget: 32000, limit: 24576},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			request := requestWithAllowance(llmprotocol.Int64(128000))
			if _, ok := ClampMessagesAllowance(request, &testCase.budget, testCase.limit); ok {
				t.Fatalf("clamp = true, want the refusal kept")
			}
			if got := *request.Sampling.MaxOutputTokens; got != 128000 {
				t.Fatalf("max_output_tokens = %d, want the stated 128000 left for the refusal", got)
			}
		})
	}
}

// A budget below the limit is clamped into, and the budget itself is never
// lowered: it is the admitted control's, fixed per episode by PlacementKey.
// A budget one token below the limit is the tightest valid case, since
// Anthropic requires only that the budget be strictly below max_tokens, and
// RaiseMessagesAllowance leaves the clamped allowance alone because it then
// exceeds the budget.
func TestClampMessagesAllowanceKeepsTheAdmittedBudget(t *testing.T) {
	for _, budget := range []int64{2048, 24575} {
		request := requestWithAllowance(llmprotocol.Int64(128000))
		admitted := budget
		if _, ok := ClampMessagesAllowance(request, &admitted, 24576); !ok {
			t.Fatalf("budget %d: clamp = false, want a clamp below the 24576 limit", budget)
		}
		if admitted != budget {
			t.Fatalf("budget %d: became %d, want the admitted budget untouched", budget, admitted)
		}
		if got := *request.Sampling.MaxOutputTokens; got != 24576 {
			t.Fatalf("budget %d: max_output_tokens = %d, want the limit 24576", budget, got)
		}
		// Anthropic's rule, and the state the codec will check.
		if admitted >= *request.Sampling.MaxOutputTokens {
			t.Fatalf("budget %d is not strictly below max_tokens %d", admitted, *request.Sampling.MaxOutputTokens)
		}
	}
}
