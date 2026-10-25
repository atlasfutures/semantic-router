package cachecontext

import (
	"testing"
	"time"
)

func TestCompletionKeepsExecutionEvidenceAcrossConsultsAndFailures(t *testing.T) {
	now := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)
	previous := &CallCache{ActionID: actionIDFor(0), CompletedAt: now.Add(-time.Minute)}
	staged := CallCache{ActionID: actionIDFor(1), ContextEpoch: "2", TTLSeconds: 300, TTLSource: "anthropic cache_control ephemeral 5m"}
	if got := CommitCall(previous, staged, false, now); got != previous {
		t.Fatal("a consult or incomplete call replaced the completed evidence")
	}
	prompt, read := 100, 0
	staged.RecordCompletion(now.Add(-time.Second), Usage{PromptTokens: &prompt, CacheReadTokens: &read})
	prompt = 200
	if previous.CompletedAt != now.Add(-time.Minute) {
		t.Fatal("staging changed the committed clock")
	}
	committed := CommitCall(previous, staged, true, now)
	if committed == previous || committed.CompletedAt != now.Add(-time.Second) || *committed.PromptTokens != 100 || *committed.CacheReadTokens != 0 || committed.CacheWriteTokens != nil {
		t.Fatalf("completed provider evidence changed: %+v", committed)
	}
	if got := CommitCall(committed, CallCache{}, false, now.Add(time.Minute)); got != committed {
		t.Fatal("a later decision-only consult advanced completed-call evidence")
	}
	staged.RecordCompletion(now.Add(time.Minute), Usage{PromptTokens: &prompt, Estimated: true})
	committed = CommitCall(committed, staged, true, now)
	if committed.CompletedAt != now || committed.PromptTokens != nil || committed.CacheReadTokens != nil {
		t.Fatalf("estimated usage or future clock leaked into evidence: %+v", committed)
	}
}
