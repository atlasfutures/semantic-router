package protocolcodec

import "testing"

// A bound the Router derives has to travel beside an effort level, because
// measured on the dev cell 2026-09-04 neither thinking arm obeys the bound on
// its own: with reasoning.max_tokens 1024 travelling alone,
// xiaomi/mimo-v2.5-pro@thinking-on spent 21,674 reasoning tokens and
// deepseek/deepseek-v4-flash@thinking-on spent 20,974. The same shape with
// reasoning_effort medium beside the bound spent 16,030, so the effort dial is
// the only control the providers act on and the bound is the cap for the ones
// that honour it.
//
// The level is not chosen freely. OpenRouter documents the conversion in the
// other direction, and this reverses it: "budget_tokens = max(min(max_tokens *
// {effort_ratio}, 128000), 1024)", with the ratio "0.8 for high effort, 0.5
// for medium effort, 0.2 for low effort". The level that travels is the
// cheapest one whose documented budget reaches the bound, so the dial never
// buys less reasoning than the bound allows and never buys a step more than it
// has to. Read 2026-09-04:
//
//	https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
func TestReasoningEffortForBound(t *testing.T) {
	tests := []struct {
		name      string
		bound     int64
		allowance int64
		want      string
	}{
		{
			// The floor is what makes both of these low: 20% of either
			// allowance is under 1024, so low already buys the whole bound.
			name:  "the floored bound is bought by the cheapest dial",
			bound: 1024, allowance: 512, want: "low",
		},
		{
			name:  "a bound at the floor stays low against a larger allowance",
			bound: 1024, allowance: 2048, want: "low",
		},
		{
			// 20% of 10,000 is 2,000, short of the bound; 50% is 5,000, which
			// reaches it exactly.
			name:  "half the allowance is medium",
			bound: 5000, allowance: 10000, want: "medium",
		},
		{
			// 50% is 5,000 and short; 80% is 8,000 and reaches it.
			name:  "most of the allowance is high",
			bound: 8000, allowance: 10000, want: "high",
		},
		{
			// A bound that no documented dial reaches gets the largest dial
			// there is. Nothing buys more reasoning than high.
			name:  "a bound above every dial gets the largest one",
			bound: 32000, allowance: 32000, want: "high",
		},
		{
			// The conversion caps at 128,000 before the bound is compared, so
			// an allowance past that point cannot make a dial reach further.
			name:  "the documented cap holds above 128000",
			bound: 200000, allowance: 400000, want: "high",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := ReasoningEffortForBound(test.bound, test.allowance); got != test.want {
				t.Fatalf("ReasoningEffortForBound(%d, %d) = %q, want %q",
					test.bound, test.allowance, got, test.want)
			}
		})
	}
}
