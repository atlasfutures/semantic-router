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
	// busy are the streams whose lease is held.
	busy map[string]bool
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

// view is the registry with the streams world holds leases on marked busy.
func (world *forkWorld) view() EpisodeForkView {
	view, err := world.store.ForkView(context.Background(), world.root)
	if err != nil {
		world.t.Fatal(err)
	}
	if view.Main != nil {
		view.Main.Leased = world.busy[EpisodeForkMainStream]
	}
	for name, stream := range view.Siblings {
		stream.Leased = world.busy[name]
	}
	return view
}

// admit resolves and admits request; busy leaves its turn in flight (the
// stream's lease held).
func (world *forkWorld) admit(request EpisodeForkRequest, busy bool) EpisodeForkResolution {
	world.t.Helper()
	world.now = world.now.Add(time.Second)
	view := world.view()
	resolution := ResolveEpisodeFork(view, request, world.now, time.Hour)
	current := view.Main
	if resolution.Fork() {
		current = view.Siblings[resolution.Stream]
	}
	evict := resolution.Evict
	if resolution.DisplacedRecord != nil {
		_ = world.store.RecordForkStream(context.Background(), world.root, resolution.Displaced, resolution.DisplacedRecord, evict)
		evict = ""
	}
	if err := world.store.RecordForkStream(context.Background(), world.root, resolution.Stream,
		current.Admitted(request, world.now, HashEpisodeID(resolution.Stream)), evict); err != nil {
		world.t.Fatal(err)
	}
	if world.busy == nil {
		world.busy = map[string]bool{}
	}
	world.busy[resolution.Stream] = busy
	return resolution
}

// finish releases stream's lease.
func (world *forkWorld) finish(stream string) {
	world.busy[stream] = false
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
	for _, b := range []EpisodeForkRequest{
		forkRequest("s", "u1", "a1", "uA", "user:uB"), // history incl. A's prompt, no reply
		forkRequest("s", "u1", "a1", "uB"),            // a different last user message
	} {
		world := newForkWorld(t)
		world.admit(forkRequest("s", "u1"), false)
		world.admit(forkRequest("s", "u1", "a1", "uA"), true) // A in flight
		got := world.admit(b, true)
		if !got.Fork() || got.Reason != ForkReasonForkBusy || !got.New {
			t.Fatalf("B = %+v, want a new fork of a busy main", got)
		}
		if again := world.admit(b, true); again.Stream != got.Stream {
			t.Fatalf("B's retry went to %q, want %q", again.Stream, got.Stream)
		}
		world.finish(got.Stream)
	}
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "uA"), true) // A in flight
	world.admit(forkRequest("s", "u1", "a1", "uB"), false)
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
	// The main stream's newest turn committed: B departs from its message.
	branch := forkRequest("s", "u1", "a1", "u2", "a2", "uB")
	branch.Committed = map[string]bool{EpisodeForkMainStream: true}
	got := world.admit(branch, false)
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
		got := world.admit(forkRequest("s", "u1", "a1", fmt.Sprintf("fork %d", index)), true)
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

// An interrupted turn (its lease released on abort) leaves its stream idle
// at once: a "continue" sent after it stays on the stream.
func TestInterruptedTurnThenContinueStaysOnMain(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), true)
	world.finish(EpisodeForkMainStream) // aborted: the lease is gone
	if got := world.admit(forkRequest("s", "u1", "a1", "u2", "user:continue"), false); got.Fork() {
		t.Fatalf("continue after an interrupted turn = %+v, want main", got)
	}
}

// An edited retry after a failed turn stays on the stream; the same shape
// beside a turn in flight is a parallel request and forks.
func TestEditedRetryAfterFailureStaysAndBesideATurnForks(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3"), false) // failed upstream
	got := world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3 edited"), false)
	if got.Fork() || got.Reason != ForkReasonRetryEdit {
		t.Fatalf("edited retry = %+v, want retry_edit on main", got)
	}
	world.busy[EpisodeForkMainStream] = true // the edited retry's turn is in flight
	beside := world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "another message"), true)
	if !beside.Fork() || beside.Reason != ForkReasonForkBusy {
		t.Fatalf("parallel request = %+v, want fork_parent_busy", beside)
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

// A sibling made by a different system prompt keeps its next turns once the
// main stream is idle again, though both streams' prefixes match.
func TestSameMessagesDifferentRootKeepsItsSibling(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), true)
	fork := world.admit(forkRequest("other", "u1", "a1", "u2"), false)
	if !fork.Fork() {
		t.Fatalf("different root beside a busy turn = %+v", fork)
	}
	world.finish(EpisodeForkMainStream)
	if got := world.admit(forkRequest("other", "u1", "a1", "u2", "a2", "u3"), false); got.Stream != fork.Stream {
		t.Fatalf("its next turn = %+v, want %q", got, fork.Stream)
	}
}

// A request departing from a longer, busy sibling forks from it; a shorter
// idle match on the main stream does not take it.
func TestShorterIdleMatchDoesNotTakeAForkOfABusySibling(t *testing.T) {
	world2 := newForkWorld(t)
	world2.admit(forkRequest("s", "u"), false)
	world2.admit(forkRequest("s", "u", "a", "x"), true)
	sub := world2.admit(forkRequest("s", "u", "a", "user:sub"), true)
	if !sub.Fork() {
		t.Fatalf("sub = %+v", sub)
	}
	world2.finish(EpisodeForkMainStream)
	got := world2.admit(forkRequest("s", "u", "a", "user:sub", "user:parallel"), true)
	if !got.Fork() || got.Stream == sub.Stream || got.Parent != sub.Stream {
		t.Fatalf("parallel to the busy sibling = %+v, want a new fork of %q", got, sub.Stream)
	}
}

// Main busy, an idle sibling with the same messages and another root: a
// request of the main's root sent beside the main's turn forks from main,
// and never joins the other root's sibling.
func TestEqualLengthOtherRootSiblingDoesNotTakeTheRequest(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), true)
	other := world.admit(forkRequest("other", "u1", "a1", "u2"), false)
	got := world.admit(forkRequest("s", "u1", "a1", "u2", "user:directive"), true)
	if !got.Fork() || got.Stream == other.Stream || got.Parent != EpisodeForkMainStream {
		t.Fatalf("directive beside main's turn = %+v, want a new fork of main (not %q)", got, other.Stream)
	}
}

// A write deletes siblings idle past the TTL, so a long-lived root's
// registry stays bounded.
func TestRecordDeletesExpiredSiblings(t *testing.T) {
	world := newForkWorld(t)
	world.admit(forkRequest("s", "u1"), false)
	world.admit(forkRequest("s", "u1", "a1", "u2"), true)
	world.admit(forkRequest("s", "u1", "a1", "user:old fork"), true)
	world.finish(EpisodeForkMainStream)
	world.now = world.now.Add(40 * time.Minute)
	world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3"), false)
	world.now = world.now.Add(40 * time.Minute)
	world.admit(forkRequest("s", "u1", "a1", "u2", "a2", "u3", "a3", "u4"), false)
	world.store.mu.Lock()
	streams := len(world.store.forks[world.root].streams)
	world.store.mu.Unlock()
	if streams != 1 {
		t.Fatalf("registry holds %d streams after the sibling expired, want main alone", streams)
	}
}
