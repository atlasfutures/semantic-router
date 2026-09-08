package sessiontelemetry

import (
	"errors"
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

// erroringRouterSessionStateStore fails every shared-store operation. It stands
// in for a shared store that is down, unreachable, or past its timeout.
type erroringRouterSessionStateStore struct{ loads, saves int }

func (s *erroringRouterSessionStateStore) Load(string) (RouterSessionSnapshot, bool, error) {
	s.loads++
	return RouterSessionSnapshot{}, false, errors.New("shared store unavailable")
}

func (s *erroringRouterSessionStateStore) Save(RouterSessionSnapshot, time.Duration) error {
	s.saves++
	return errors.New("shared store unavailable")
}

func (s *erroringRouterSessionStateStore) Close() error { return nil }

// TestRouterSessionSnapshot_StaleLocalEntryShadowsSharedStore shows that a
// replica holding a local entry never re-reads the shared store, so it keeps
// serving its own copy after another replica advances the same session.
func TestRouterSessionSnapshot_StaleLocalEntryShadowsSharedStore(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := &fakeRouterSessionStateStore{}
	SetRouterSessionStateStore(store)
	t.Cleanup(func() {
		SetRouterSessionStateStore(nil)
		ResetRouterSessionMemoryForTesting()
	})

	firstTurn := time.Now()
	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "session-1",
		SelectedModel: "model-a",
		Timestamp:     firstTurn,
	})

	// A second replica advances the same session and writes it to the shared store.
	secondTurn := firstTurn.Add(time.Minute)
	store.snapshot = RouterSessionSnapshot{
		SessionID:    "session-1",
		CurrentModel: "model-b",
		LastSeen:     secondTurn,
		TurnCount:    2,
	}
	store.found = true

	snapshot, ok := GetRouterSessionSnapshot("session-1", secondTurn)
	if !ok {
		t.Fatal("expected a snapshot for session-1")
	}
	if snapshot.CurrentModel != "model-b" {
		t.Fatalf("this replica served its stale local entry: current model is %q, but the shared store holds %q",
			snapshot.CurrentModel, "model-b")
	}
}

// TestRouterSessionStore_ErrorsAreObservable shows that shared-store failures on
// the write path and the read path are both discarded without a trace.
func TestRouterSessionStore_ErrorsAreObservable(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := &erroringRouterSessionStateStore{}
	SetRouterSessionStateStore(store)
	t.Cleanup(func() {
		SetRouterSessionStateStore(nil)
		ResetRouterSessionMemoryForTesting()
	})

	core, logs := observer.New(zapcore.ErrorLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))

	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "session-2",
		SelectedModel: "model-a",
		Timestamp:     time.Now(),
	})
	ResetRouterSessionMemoryForTesting()
	GetRouterSessionSnapshot("session-2", time.Now())

	if store.saves == 0 || store.loads == 0 {
		t.Fatalf("expected the shared store to be exercised: saves=%d loads=%d", store.saves, store.loads)
	}
	if logs.Len() == 0 {
		t.Fatalf("the shared store failed %d save(s) and %d load(s) with nothing logged at error level",
			store.saves, store.loads)
	}
}
