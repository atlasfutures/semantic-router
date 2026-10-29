package raylinearc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

// forkMessages builds a transcript from role:text pairs.
func forkRequest(system string, turns ...string) EpisodeForkRequest {
	messages := make([]json.RawMessage, 0, len(turns))
	roles := make([]string, 0, len(turns))
	for index, text := range turns {
		role := "user"
		if index%2 == 1 {
			role = "assistant"
		}
		if rest, ok := strings.CutPrefix(text, "user:"); ok {
			role, text = "user", rest
		}
		message, _ := json.Marshal(map[string]string{"role": role, "content": text})
		messages = append(messages, message)
		roles = append(roles, role)
	}
	digests := NewPrefixDigests(messages)
	systemJSON, _ := json.Marshal(system)
	return EpisodeForkRequest{Digests: digests, Roles: roles, Root: ForkRootDigest(digests, systemJSON, nil)}
}

type forkWorld struct {
	t     *testing.T
	store *MemoryEpisodeStore
	root  string
	now   time.Time
	// tokens counts admissions; last is each stream's latest token.
	tokens int
	last   map[string]string
}

func newForkWorld(t *testing.T) *forkWorld {
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{MaxEpisodes: 8, IdleTTL: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	world := &forkWorld{t: t, store: store, root: HashEpisodeID("root"), now: time.Unix(1_700_000_000, 0)}
	store.now = func() time.Time { return world.now }
	return world
}

// admit resolves and admits request; busy leaves the turn in flight.
func (world *forkWorld) admit(request EpisodeForkRequest, busy bool) EpisodeForkResolution {
	world.t.Helper()
	world.now = world.now.Add(time.Second)
	view, err := world.store.ForkView(context.Background(), world.root)
	if err != nil {
		world.t.Fatal(err)
	}
	resolution := ResolveEpisodeFork(view, request, world.now, time.Hour)
	current := view.Main
	if resolution.Fork() {
		current = view.Siblings[resolution.Stream]
	}
	busyUntil := world.now
	if busy {
		busyUntil = world.now.Add(time.Minute)
	}
	evict := resolution.Evict
	if resolution.DisplacedRecord != nil {
		_ = world.store.RecordForkStream(context.Background(), world.root, resolution.Displaced, resolution.DisplacedRecord, evict)
		evict = ""
	}
	if err := world.store.RecordForkStream(context.Background(), world.root, resolution.Stream,
		current.Admitted(request, world.now, busyUntil, world.token(resolution.Stream)), evict); err != nil {
		world.t.Fatal(err)
	}
	return resolution
}

// token is the admission token for stream's next admission.
func (world *forkWorld) token(stream string) string {
	world.tokens++
	token := fmt.Sprintf("t%d", world.tokens)
	if world.last == nil {
		world.last = map[string]string{}
	}
	world.last[stream] = token
	return token
}

// finish clears the busy mark of stream's last admission.
func (world *forkWorld) finish(stream string) {
	_ = world.store.ClearForkBusy(context.Background(), world.root, stream, world.last[stream])
}

func TestForkContinuationStaysOnMain(t *testing.T) {
	world := newForkWorld(t)
	if got := world.admit(forkRequest("s", "u1"), false); got.Fork() || got.Reason != ForkReasonStart {
		t.Fatalf("first turn = %+v", got)
	}
	if got := world.admit(forkRequest("s", "u1", "a1", "u2"), false); got.Fork() || got.Reason != ForkReasonContinuation {
		t.Fatalf("second turn = %+v", got)
	}
}

// The prod shape (episode f9b21a8b): run A's turn is in flight; run B sends
// the same history with a new user message and without A's reply. B is a
// fork, its retries and next turns stay on its sibling, and A's next turn
// stays on the main stream.
func TestForkWhileMainInFlightGetsOneSibling(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "uA"), true) // A in flight
	for _, b := range []EpisodeForkRequest{
		forkRequest("s", "u1", "a1", "uA", "user:uB"), // history incl. A's prompt, no reply
		forkRequest("s", "u1", "a1", "uB"),            // a different last user message
	} {
		got := world.admit(b, true)
		if !got.Fork() || got.Reason != ForkReasonForkBusy || !got.New {
			t.Fatalf("B = %+v, want a new fork of a busy main", got)
		}
		if again := world.admit(b, true); again.Stream != got.Stream {
			t.Fatalf("B's retry went to %q, want %q", again.Stream, got.Stream)
		}
		world.finish(got.Stream)
	}
	b := forkRequest("s", "u1", "a1", "uB")
	view, _ := world.store.ForkView(context.Background(), world.root)
	sibling := ResolveEpisodeFork(view, b, world.now, time.Hour).Stream
	if got := world.admit(forkRequest("s", "u1", "a1", "uB", "aB", "uB2"), false); got.Stream != sibling || got.Reason != ForkReasonContinuation {
		t.Fatalf("B's next turn = %+v, want sibling %q", got, sibling)
	}
	world.finish(EpisodeForkMainStream)
	if got := world.admit(forkRequest("s", "u1", "a1", "uA", "aA", "uA2"), false); got.Fork() {
		t.Fatalf("A's next turn = %+v, want main", got)
	}
}

// Idle: B's history leaves out the main stream's last reply and branches
// off an earlier turn: a fork, not a rewrite of the main stream.
func TestIdleBranchIsAFork(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3"), false)
	got := world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "uB"), false)
	if !got.Fork() || got.Reason != ForkReasonForkBranch {
		t.Fatalf("idle branch = %+v, want fork_branch", got)
	}
	if main := world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3", "a3", "u4"), false); main.Fork() {
		t.Fatalf("main's next turn = %+v", main)
	}
}

// A compaction (the same system and first message, a shorter or summarised
// history) stays on the main stream, as does a recognised compaction with
// a new first message; the replaced branch is kept as a dormant sibling.
func TestCompactionStaysOnMain(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3"), false)
	got := world.admit(forkRequest("s", "u1", "summary", "u4"), false)
	if got.Fork() || got.Reason != ForkReasonRewrite || got.Displaced == "" {
		t.Fatalf("shorter history = %+v, want a rewrite of main", got)
	}
	summary := forkRequest("s", "summary of everything", "ok", "u5")
	summary.Compaction = true
	if got := world.admit(summary, false); got.Fork() || got.Reason != ForkReasonCompaction {
		t.Fatalf("recognised compaction = %+v", got)
	}
	// The displaced branch continuing is a fork of its own, not a rewrite.
	if got := world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3", "a3", "u9"), false); !got.Fork() {
		t.Fatalf("displaced branch continuing = %+v, want a sibling", got)
	}
}

func TestDifferentRootIsAFork(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	got := world.admit(forkRequest("subagent system", "task"), false)
	if !got.Fork() || got.Reason != ForkReasonForkRoot {
		t.Fatalf("subagent = %+v", got)
	}
	if next := world.admit(forkRequest("subagent system", "task", "r", "t2"), false); next.Stream != got.Stream {
		t.Fatalf("subagent's next turn = %+v, want %q", next, got.Stream)
	}
}

func TestSiblingCapEvictsLeastRecentlyAdmitted(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), true)
	first := ""
	for index := range MaxEpisodeForkSiblings + 1 {
		got := world.admit(forkRequest("s", "u1", "a1", fmt.Sprintf("fork %d", index)), false)
		if !got.Fork() || !got.New {
			t.Fatalf("fork %d = %+v", index, got)
		}
		if index == 0 {
			first = got.Stream
		}
		if got.SiblingCount > MaxEpisodeForkSiblings {
			t.Fatalf("fork %d: sibling_count %d", index, got.SiblingCount)
		}
		if index == MaxEpisodeForkSiblings && got.Evict != first {
			t.Fatalf("17th fork evicted %q, want the oldest %q", got.Evict, first)
		}
	}
	view, _ := world.store.ForkView(context.Background(), world.root)
	if len(view.Siblings) != MaxEpisodeForkSiblings {
		t.Fatalf("siblings = %d", len(view.Siblings))
	}
	if _, kept := view.Siblings[first]; kept {
		t.Fatal("the least recently admitted sibling was kept")
	}
}

// A true next turn arriving before the in-flight turn's mark clears (it
// carries the reply) is still the main stream's continuation.
func TestNextTurnBeforeMarkClearsIsAContinuation(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), true)
	if got := world.admit(forkRequest("s", "u1", "a1", "u2"), true); got.Fork() {
		t.Fatalf("next turn = %+v", got)
	}
	if got := world.admit(forkRequest("s", "u1", "a1", "u2"), true); got.Fork() {
		t.Fatalf("retry of the in-flight turn = %+v", got)
	}
}

// A finished turn clears only its own busy mark: a turn admitted on the
// stream after it keeps the stream busy.
func TestFinishedTurnKeepsANewerAdmissionBusy(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), true)
	stale := world.last[EpisodeForkMainStream]
	world.admit(forkRequest("s", "u1", "a1", "u2"), true)
	_ = world.store.ClearForkBusy(context.Background(), world.root, EpisodeForkMainStream, stale)
	if got := world.admit(forkRequest("s", "u1", "a1", "user:beside"), true); !got.Fork() || got.Reason != ForkReasonForkBusy {
		t.Fatalf("request beside the newer turn = %+v, want fork_parent_busy", got)
	}
}

// The same messages under a different system prompt beside a turn in
// flight are another conversation, not a retry.
func TestDifferentSystemBesideATurnInFlightForks(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), true)
	if got := world.admit(forkRequest("other system", "u1", "a1", "u2"), true); !got.Fork() {
		t.Fatalf("different system = %+v, want a fork", got)
	}
}

// A compaction that extends no prefix belongs to the stream that spoke last,
// not by default to the main stream.
func TestSiblingCompactionStaysOnTheSibling(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), false)
	fork := world.admit(forkRequest("sub", "task"), false)
	world.admit(forkRequest("sub", "task", "r1", "t2"), false)
	compaction := forkRequest("sub", "summary", "ok", "t3")
	compaction.Compaction = true
	if got := world.admit(compaction, false); got.Stream != fork.Stream {
		t.Fatalf("sibling compaction = %+v, want %q", got, fork.Stream)
	}
}
