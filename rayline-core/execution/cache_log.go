package execution

import (
	"sort"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

type cacheCallLog struct {
	CompletedAt      time.Time                   `json:"completed_at"`
	PromptTokens     *int                        `json:"prompt_tokens"`
	CacheReadTokens  *int                        `json:"cache_read_tokens"`
	CacheWriteTokens *int                        `json:"cache_write_tokens"`
	UsageProvenance  *string                     `json:"usage_provenance"`
	TTLSeconds       *int                        `json:"ttl_s"`
	TTLSource        *string                     `json:"ttl_source"`
	CurrentPrefix    *cachecontext.CurrentPrefix `json:"current_prefix"`
	Residency        *Residency                  `json:"residency"`
}

type cacheWorkerLog struct {
	Deployment string `json:"deployment"`
	ActionID   string `json:"action_id"`
	cacheCallLog
}

type cacheCompletionLog struct {
	CompletedDeploymentRecords int              `json:"completed_deployment_records"`
	CompletedTurns             uint64           `json:"completed_turns"`
	CurrentCall                cacheCallLog     `json:"current_call"`
	Workers                    []cacheWorkerLog `json:"workers"`
}

// Log committed counts and already measured prefix evidence only. Records and
// matching token prefixes do not establish live KV residency or a cache hit.
// Prompt bodies, token IDs and hidden instructions never enter this projection.
func completionCacheEvidence(state *branchState, deployment string, measured ExecutionContext, currentPrefix *cachecontext.CurrentPrefix) cacheCompletionLog {
	prefixes := make(map[string]*cachecontext.CurrentPrefix, len(measured.Workers)+1)
	for _, worker := range measured.Workers {
		if worker.PrefixTokens != nil {
			prefixes[worker.DeploymentID] = &cachecontext.CurrentPrefix{Tokens: *worker.PrefixTokens, OverlapTokens: worker.OverlapTokens, Units: worker.TokenUnits, Provenance: worker.PrefixProvenance}
		}
	}
	prefixes[deployment] = currentPrefix
	result := cacheCompletionLog{CompletedDeploymentRecords: len(state.workers), CompletedTurns: state.completed, Workers: make([]cacheWorkerLog, 0, len(state.workers))}
	for id, worker := range state.workers {
		call := worker.Call
		entry := cacheCallLog{CompletedAt: call.CompletedAt, PromptTokens: call.PromptTokens, CacheReadTokens: call.CacheReadTokens, CacheWriteTokens: call.CacheWriteTokens, CurrentPrefix: prefixes[id], Residency: worker.Residency}
		if call.PromptTokens != nil || call.CacheReadTokens != nil || call.CacheWriteTokens != nil {
			provenance := "provider_reported"
			entry.UsageProvenance = &provenance
		}
		if call.TTLSeconds > 0 && call.TTLSource != "" {
			entry.TTLSeconds, entry.TTLSource = &call.TTLSeconds, &call.TTLSource
		}
		result.Workers = append(result.Workers, cacheWorkerLog{Deployment: id, ActionID: call.ActionID, cacheCallLog: entry})
		if id == deployment {
			result.CurrentCall = entry
		}
	}
	sort.Slice(result.Workers, func(i, j int) bool { return result.Workers[i].Deployment < result.Workers[j].Deployment })
	return result
}
