package cachecontext

import "time"

// Usage contains only provider-reported prompt/cache counts. Nil means the
// provider did not report that count. Estimated or invalid usage is never
// evidence of a provider cache hit.
type Usage struct {
	PromptTokens     *int
	CacheReadTokens  *int
	CacheWriteTokens *int
	Invalid          bool
	Estimated        bool
}

// RecordCompletion stages evidence after the complete provider response has
// been decoded. It must not be called for an incomplete or failed stream.
// Staging alone does not update an episode; its owner commits or discards it.
func (call *CallCache) RecordCompletion(now time.Time, usage Usage) {
	call.CompletedAt = now
	call.PromptTokens, call.CacheReadTokens, call.CacheWriteTokens = nil, nil, nil
	if usage.Invalid || usage.Estimated {
		return
	}
	call.PromptTokens = copyCount(usage.PromptTokens)
	call.CacheReadTokens = copyCount(usage.CacheReadTokens)
	call.CacheWriteTokens = copyCount(usage.CacheWriteTokens)
}

func copyCount(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// CommitCall returns the last completed call for a successfully committed
// turn. A decision-only consult leaves the previous evidence and its clock
// untouched. Failed turns must discard their staged call without committing.
func CommitCall(previous *CallCache, staged CallCache, completed bool, now time.Time) *CallCache {
	if staged.ActionID == "" || !completed {
		return previous
	}
	if staged.CompletedAt.IsZero() || staged.CompletedAt.After(now) {
		staged.CompletedAt = now
	}
	return &staged
}
