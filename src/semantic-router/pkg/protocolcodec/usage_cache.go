package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// Cache detail fields are optional provider evidence. In particular, vLLM can
// omit them when usage details are disabled; absence must not invent a zero.
//
// Usage evidence never fails a completion (US-003c): the provider has already
// billed the turn, and a decode failure leaves nothing to bill from. Aliases
// that disagree, or a breakdown that does not fit its total, leave the parts
// unknown, keep the stated totals, and are reported once per process. The
// error return is kept for upstream's call shape and is always nil.
func decodeInputCacheUsage(usage *llmprotocol.Usage, cached, written *int64, aliases ...*int64) error {
	usage.InputCacheRead = optionalAuthoritative(cached)
	usage.InputUncached = unknownCount()
	for _, alias := range aliases {
		if alias == nil {
			continue
		}
		if written != nil && *written != *alias {
			reportUnreconciledUsage("usage.cache_write_tokens")
			usage.InputCacheWrite = unknownCount()
			return nil
		}
		written = alias
	}
	usage.InputCacheWrite = optionalAuthoritative(written)
	if usage.InputTotal.Value == nil {
		return nil
	}
	total := *usage.InputTotal.Value
	read, write := int64(0), int64(0)
	if cached != nil {
		read = *cached
	}
	if written != nil {
		write = *written
	}
	if read < 0 || write < 0 || total < read || write > total-read {
		reportUnreconciledUsage("usage.input_cache_details")
		usage.InputCacheRead = unknownCount()
		usage.InputCacheWrite = unknownCount()
		return nil
	}
	if cached == nil || written == nil {
		return nil
	}
	usage.InputUncached = llmprotocol.TokenCount{Value: llmprotocol.Int64(total - *cached - *written), Provenance: llmprotocol.UsageDerived}
	return nil
}

func optionalAuthoritative(value *int64) llmprotocol.TokenCount {
	if value == nil {
		return unknownCount()
	}
	return authoritative(*value)
}

func appendAnthropicPartialCacheOmission(diagnostics *llmprotocol.Diagnostics, policy llmprotocol.Policy, source llmprotocol.WireFormat, usage llmprotocol.Usage) {
	readKnown, writeKnown := usage.InputCacheRead.Value != nil, usage.InputCacheWrite.Value != nil
	if readKnown != writeKnown {
		appendAccountingOmission(diagnostics, policy, source, llmprotocol.AnthropicMessagesV1,
			"usage.cache", "Messages requires numeric cache buckets; the unreported bucket is zero-filled for representation, while settlement retains unknown usage")
	}
}
