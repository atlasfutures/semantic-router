package protocolcodec

import "testing"

// The per-turn fields leave the line and every other field stays, in order,
// so two turns of one conversation carry the same line. Text that is not the
// line is never rewritten.
func TestStableBillingAttributionLine(t *testing.T) {
	cases := []struct {
		name, text, want string
		isLine           bool
	}{
		{"claude code 2.1.319",
			"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli; cch=8dc28; cc_prompt_id=e69f0f9f-03e0;",
			"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli;", true},
		{"subagent flag kept",
			"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli; cch=0cd8e; cc_is_subagent=1;",
			"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli; cc_is_subagent=1;", true},
		{"no per-turn fields",
			"x-anthropic-billing-header: cc_version=2.1.282.772; cc_entrypoint=sdk-cli;",
			"x-anthropic-billing-header: cc_version=2.1.282.772; cc_entrypoint=sdk-cli;", true},
		{"prose after the prefix",
			"x-anthropic-billing-header: cc_version=2.1.260.ada;\nYou are a Claude agent.",
			"x-anthropic-billing-header: cc_version=2.1.260.ada;\nYou are a Claude agent.", false},
		{"no version field",
			"x-anthropic-billing-header: cch=8dc28;", "x-anthropic-billing-header: cch=8dc28;", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, isLine := StableBillingAttributionLine(c.text)
			if got != c.want || isLine != c.isLine {
				t.Fatalf("StableBillingAttributionLine(%q) = %q, %v; want %q, %v", c.text, got, isLine, c.want, c.isLine)
			}
		})
	}
	first, _ := StableBillingAttributionLine(
		"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli; cch=8dc28; cc_prompt_id=a;")
	next, _ := StableBillingAttributionLine(
		"x-anthropic-billing-header: cc_version=2.1.319.20d; cc_entrypoint=cli; cch=0cd8e; cc_prompt_id=b;")
	if first != next {
		t.Fatalf("two turns differ after the per-turn fields left: %q, %q", first, next)
	}
}
