package extproc

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A new episode whose session start hangs (a scale-up placed it on a
// container still booting) holds no admission slot while it waits: with the
// cell's gate at one slot, an established episode's turn is decided meanwhile
// instead of being shed. The hanging turn is given up at the start timeout,
// starts a new session once and is decided there, within two start timeouts.
func TestPolicySessionStartHangHoldsNoSlotAndRetriesOnce(t *testing.T) {
	const startTimeout = 300 * time.Millisecond
	gate := raylinearc.NewAdmissionGate(1)
	fixture, front := newPolicySessionFixtureWith(t, startTimeout, gate)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})

	established, _ := raylinearc.NewEpisodeState(2)
	first, err := fixture.selectOn(t, established, body)
	if err != nil {
		t.Fatal(err)
	}
	established.PolicySession = first.RaylineARC.PolicySession

	front.mu.Lock()
	front.hangStarts, front.hanging = 1, make(chan struct{}, 1)
	front.mu.Unlock()
	type outcome struct {
		result  *selection.SelectionResult
		err     error
		elapsed time.Duration
	}
	done := make(chan outcome, 1)
	go func() {
		fresh, _ := raylinearc.NewEpisodeState(2)
		began := time.Now()
		result, err := fixture.selectOnWith(t, fresh, body, func(arcContext *selection.RaylineARCSelectionContext) {
			arcContext.EpisodeIDHash = strings.Repeat("b", 64)
		})
		done <- outcome{result, err, time.Since(began)}
	}()
	select {
	case <-front.hanging:
	case <-time.After(5 * time.Second):
		t.Fatal("the new episode's start never reached the server")
	}
	if inflight := gate.Inflight(); inflight != 0 {
		t.Fatalf("a hanging session start holds %d admission slot(s)", inflight)
	}
	if _, err := fixture.selectOn(t, established, body); err != nil {
		t.Fatalf("the established episode was shed while another start hung: %v", err)
	}

	got := <-done
	if got.err != nil {
		t.Fatalf("the retried start failed the turn: %v", got.err)
	}
	if got.result.RaylineARC.PolicySession == nil {
		t.Fatal("the retried turn records no session")
	}
	if got.elapsed > 2*startTimeout+time.Second {
		t.Fatalf("the retried turn took %s", got.elapsed)
	}
	front.mu.Lock()
	attempts := front.startAttempts
	front.mu.Unlock()
	if attempts != 3 { // the established episode's, the hung one, its retry
		t.Fatalf("start attempts %d, want 3", attempts)
	}
	if gate.Inflight() != 0 {
		t.Fatalf("slots left held: %d", gate.Inflight())
	}
}

// A start that hangs on both attempts fails the turn, bounded by two start
// timeouts, and no decide is sent.
func TestPolicySessionStartHangingTwiceFailsTheTurnBounded(t *testing.T) {
	const startTimeout = 200 * time.Millisecond
	gate := raylinearc.NewAdmissionGate(1)
	fixture, front := newPolicySessionFixtureWith(t, startTimeout, gate)
	front.hangStarts = 2
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	state, _ := raylinearc.NewEpisodeState(2)

	began := time.Now()
	if _, err := fixture.selectOn(t, state, body); err == nil {
		t.Fatal("a turn whose starts both hung was decided")
	}
	if elapsed := time.Since(began); elapsed > 2*startTimeout+time.Second {
		t.Fatalf("the failed turn took %s", elapsed)
	}
	front.mu.Lock()
	attempts := front.startAttempts
	front.mu.Unlock()
	if attempts != 2 {
		t.Fatalf("start attempts %d, want 2", attempts)
	}
	for index, container := range front.containers {
		if n := len(container.received()); n != 0 {
			t.Fatalf("container %d received %d decides", index, n)
		}
	}
	if gate.Inflight() != 0 || gate.HighWater() != 0 {
		t.Fatalf("a turn that never decided took a slot: inflight %d high water %d", gate.Inflight(), gate.HighWater())
	}
}
