package protocolcodec

import (
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// billingAttributionTurnFields are the fields of Claude Code's billing
// attribution line that change within one conversation: cch on every turn,
// cc_prompt_id with every new prompt. The others (cc_version, cc_entrypoint,
// cc_is_subagent) hold for the whole session.
var billingAttributionTurnFields = []string{"cch", "cc_prompt_id"}

// StableBillingAttributionLine returns text without the line's per-turn
// fields, and whether text is the line at all. A Messages body bound for a
// model Anthropic does not serve carries the line as prompt text, so those
// fields defeat the provider's prefix cache at the top of the system prompt;
// without them the line is the same on every turn. Text that is not the line
// is returned unchanged.
func StableBillingAttributionLine(text string) (string, bool) {
	if !billingAttributionLine(llmprotocol.Content{Kind: llmprotocol.ContentText, Text: text}) {
		return text, false
	}
	fields := strings.SplitAfter(strings.TrimPrefix(text, billingAttributionPrefix), ";")
	var stable strings.Builder
	stable.WriteString(billingAttributionPrefix)
	for _, field := range fields {
		key, _, _ := strings.Cut(strings.TrimPrefix(field, " "), "=")
		if field == "" || isBillingAttributionTurnField(key) {
			continue
		}
		stable.WriteString(field)
	}
	return stable.String(), true
}

func isBillingAttributionTurnField(key string) bool {
	for _, field := range billingAttributionTurnFields {
		if key == field {
			return true
		}
	}
	return false
}
