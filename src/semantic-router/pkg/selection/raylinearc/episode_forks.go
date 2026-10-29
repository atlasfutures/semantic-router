/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
	"time"
)

// Episode forks separate parallel requests that share one episode id: a
// subagent forked off the main conversation, two parallel requests of a
// workshop app, a Claude Code fork that repeats the parent's transcript and
// appends its own directive. Each fork is served on a sibling episode of its
// own, with its own lease, state and policy session, so it neither waits on
// the parent's lease nor mixes its transcript into the parent's.
//
// The registry is kept beside the root episode (the one the episode header
// names) and holds, per stream (the main stream and each sibling), the last
// few request prefixes admitted on it and whether a turn is in flight there.
// It is advisory: it never gates a commit, and a lost write only costs a
// fork its sibling (it is then resolved again from the content).

const (
	// MaxEpisodeForkSiblings bounds the siblings one root episode keeps. A
	// new fork beyond it evicts the least recently admitted sibling (LRU):
	// the evicted sibling's episode lives out its idle TTL untouched, and
	// its next turn, no longer found, is resolved as a new fork.
	MaxEpisodeForkSiblings = 16
	// MaxEpisodeForkPrefixes bounds the admitted prefixes kept per stream.
	MaxEpisodeForkPrefixes = 16
	// EpisodeForkMainStream is the main stream's name in the registry.
	EpisodeForkMainStream = "main"
	// EpisodeForkIDSeparator joins a root episode id and a sibling's anchor.
	EpisodeForkIDSeparator = "#fork-"
)

// Fork resolution reasons, as logged.
const (
	ForkReasonStart        = "start"
	ForkReasonContinuation = "continuation"
	ForkReasonRewrite      = "rewrite"
	ForkReasonForkBusy     = "fork_parent_busy"
	ForkReasonForkRoot     = "fork_root_differs"
	ForkReasonForkBranch   = "fork_branch"
	ForkReasonCompaction   = "compaction"
	ForkReasonRetryEdit    = "retry_edit"
)

// EpisodeForkPrefix is one admitted request's message prefix: its length and
// MessagesDigest.
type EpisodeForkPrefix struct {
	Len    int    `json:"l"`
	Digest string `json:"d"`
	// Reply is the digest of the prefix with the next message the stream's
	// next request carried (MessagesDigest of Len+1 messages): the reply this
	// prefix's turn was served, once a continuation has shown it. A request
	// that carries that reply and then departs from the stream branched off
	// it; one that does not carry it (a compaction keeping the opening
	// message) did not.
	Reply string `json:"n,omitempty"`
	// BaseLen and BaseDigest are the prefix up to and including its last
	// assistant message: the context its trailing user message was written
	// against. A request on that same context with another trailing user
	// message is an edited retry of this turn when no turn is in flight.
	BaseLen    int    `json:"bl,omitempty"`
	BaseDigest string `json:"bd,omitempty"`
	// Shell digests the system prompt and tools of a turn with no assistant
	// message (a first turn), whose base is empty: an edit of that turn's
	// message keeps its shell.
	Shell string `json:"s,omitempty"`
}

// EpisodeForkStream is one stream's registry record.
type EpisodeForkStream struct {
	// Prefixes are the admitted request prefixes, oldest first.
	Prefixes []EpisodeForkPrefix `json:"p"`
	// Root is the stream's root digest (ForkRootDigest) as its first
	// admitted request had it.
	Root string `json:"r,omitempty"`
	// LastRoot is the root digest of the newest admitted request: a
	// harness can add tools to a conversation as it goes.
	LastRoot string `json:"lr,omitempty"`
	// AdmittedAtMS is the last admission, unix milliseconds.
	AdmittedAtMS int64 `json:"t"`
	// EpisodeHash is the stream's episode id hash, whose lease says whether
	// a turn is in flight on it.
	EpisodeHash string `json:"h,omitempty"`
	// Leased is read with the view, from the stream's episode lease: a turn
	// is in flight. It lives and dies with the lease (renewed while the turn
	// streams, released on commit or abort, lapsing at lease_ttl_seconds
	// when its replica died), and is never stored.
	Leased bool `json:"-"`
}

// Latest is the stream's newest admitted prefix.
func (stream *EpisodeForkStream) Latest() (EpisodeForkPrefix, bool) {
	if stream == nil || len(stream.Prefixes) == 0 {
		return EpisodeForkPrefix{}, false
	}
	return stream.Prefixes[len(stream.Prefixes)-1], true
}

// Busy reports whether a turn is in flight on the stream: its episode's
// lease is held.
func (stream *EpisodeForkStream) Busy(time.Time) bool {
	return stream != nil && stream.Leased
}

// live reports whether the stream is still in use: admitted within the idle
// TTL, or with a turn in flight however long it has run.
func (stream *EpisodeForkStream) live(now time.Time, idleTTL time.Duration) bool {
	return stream != nil && (stream.Leased || now.Sub(time.UnixMilli(stream.AdmittedAtMS)) < idleTTL)
}

// Clone is a deep copy.
func (stream *EpisodeForkStream) Clone() *EpisodeForkStream {
	if stream == nil {
		return nil
	}
	clone := *stream
	clone.Prefixes = append([]EpisodeForkPrefix(nil), stream.Prefixes...)
	return &clone
}

// Admitted is the stream, served on episodeHash, with request admitted at
// now: its prefix is appended unless it is already the newest, and a newest
// prefix it extends learns its reply.
func (stream *EpisodeForkStream) Admitted(request EpisodeForkRequest, now time.Time, episodeHash string) *EpisodeForkStream {
	next := stream.Clone()
	if next == nil {
		next = &EpisodeForkStream{}
	}
	prefix := request.Prefix()
	latest, ok := next.Latest()
	if ok && latest.Len < prefix.Len && request.extends(latest) && request.nextTurnOf(latest) {
		next.Prefixes[len(next.Prefixes)-1].Reply = request.Digests.Digest(latest.Len + 1)
	}
	if !ok || latest.Len != prefix.Len || latest.Digest != prefix.Digest {
		next.Prefixes = append(next.Prefixes, prefix)
	}
	root := request.Root
	if len(next.Prefixes) > MaxEpisodeForkPrefixes {
		next.Prefixes = next.Prefixes[len(next.Prefixes)-MaxEpisodeForkPrefixes:]
	}
	if next.Root == "" {
		next.Root = root
	}
	next.LastRoot = root
	next.AdmittedAtMS = now.UnixMilli()
	next.EpisodeHash = episodeHash
	next.Leased = false
	return next
}

// EpisodeForkView is a root episode's registry as read: the main stream (nil
// when none was admitted) and the siblings by anchor.
type EpisodeForkView struct {
	Main     *EpisodeForkStream
	Siblings map[string]*EpisodeForkStream
}

// EpisodeForkRegistry is implemented by episode stores that keep the fork
// registry. Writes are per stream: a stream is written only by the request
// holding that stream's lease (or by a rewrite displacing a branch), so
// concurrent writers of one stream do not arise in normal operation.
type EpisodeForkRegistry interface {
	ForkView(ctx context.Context, rootEpisodeIDHash string) (EpisodeForkView, error)
	// RecordForkStream writes stream's record and removes evict (a sibling
	// anchor, or "") in one step. It also deletes siblings idle past the
	// TTL and, past MaxEpisodeForkSiblings, the least recently admitted
	// ones, so the registry stays bounded whatever the writers raced on.
	RecordForkStream(
		ctx context.Context,
		rootEpisodeIDHash string,
		stream string,
		record *EpisodeForkStream,
		evict string,
	) error
}

// EpisodeForkRequest is what fork resolution reads of one request.
type EpisodeForkRequest struct {
	Digests *PrefixDigests
	// Roles are the messages' roles.
	Roles []string
	// Root is ForkRootDigest of the request.
	Root string
	// Shell is ForkShellDigest of the request.
	Shell string
	// Compaction is a recognised compaction continuation.
	Compaction bool
	// Committed lists the streams whose newest admitted turn committed (its
	// reply reached the client). A request that departs from a committed
	// turn's message is a branch, not an edited retry of a failed one.
	Committed map[string]bool
}

// Prefix is the request's whole message prefix, with its base.
func (request EpisodeForkRequest) Prefix() EpisodeForkPrefix {
	count := len(request.Digests.Messages())
	prefix := EpisodeForkPrefix{Len: count, Digest: request.Digests.Digest(count)}
	if base := request.lastAssistant() + 1; base > 0 && base < count {
		prefix.BaseLen, prefix.BaseDigest = base, request.Digests.Digest(base)
	} else if base == 0 {
		prefix.Shell = request.Shell
	}
	return prefix
}

// lastAssistant is the index of the request's last assistant message, -1
// when it has none.
func (request EpisodeForkRequest) lastAssistant() int {
	for index := min(len(request.Roles), len(request.Digests.Messages())) - 1; index >= 0; index-- {
		if request.Roles[index] == "assistant" {
			return index
		}
	}
	return -1
}

// retries reports whether the request is an edited retry of the turn on
// prefix: it shares that turn's base (everything up to its last assistant
// message), carries no assistant message after it, and differs from the
// turn only in what follows.
func (request EpisodeForkRequest) retries(prefix EpisodeForkPrefix) bool {
	if prefix.BaseLen == 0 {
		// A first turn: the same shell, and still no assistant message.
		return prefix.Shell != "" && prefix.Shell == request.Shell && request.lastAssistant() < 0
	}
	if prefix.BaseLen < 0 || prefix.BaseLen >= len(request.Digests.Messages()) ||
		request.Digests.Digest(prefix.BaseLen) != prefix.BaseDigest {
		return false
	}
	return request.lastAssistant() < prefix.BaseLen
}

func (request EpisodeForkRequest) extends(prefix EpisodeForkPrefix) bool {
	return prefix.Len <= len(request.Digests.Messages()) && request.Digests.Digest(prefix.Len) == prefix.Digest
}

// carriesReply reports whether the request extends prefix and carries the
// reply its turn was served.
func (request EpisodeForkRequest) carriesReply(prefix EpisodeForkPrefix) bool {
	return prefix.Reply != "" && prefix.Len < len(request.Digests.Messages()) &&
		request.Digests.Digest(prefix.Len+1) == prefix.Reply
}

// nextTurnOf reports whether the request can follow the turn on prefix:
// it is that turn itself (a retry), or it carries a reply after it. A
// request that appends a user message straight after a turn still in flight
// was sent beside that turn, not after it.
func (request EpisodeForkRequest) nextTurnOf(prefix EpisodeForkPrefix) bool {
	if prefix.Len >= len(request.Digests.Messages()) {
		return true
	}
	return prefix.Len < len(request.Roles) && request.Roles[prefix.Len] == "assistant"
}

// EpisodeForkResolution is where a request is served.
type EpisodeForkResolution struct {
	// Stream is EpisodeForkMainStream or a sibling's anchor.
	Stream string
	// New reports a sibling this request creates.
	New    bool
	Reason string
	// Evict is a sibling anchor to drop to make room, or "".
	Evict string
	// Displaced is the branch a rewrite of a stream replaces, kept as a
	// dormant sibling so that branch, if it continues, is a fork of its own
	// rather than a rewrite back. Empty unless Reason is rewrite.
	Displaced       string
	DisplacedRecord *EpisodeForkStream
	// Parent is the stream the request was resolved against.
	Parent string
	// SiblingCount is the live siblings after this resolution.
	SiblingCount int
}

// Fork reports whether the request is served on a sibling.
func (resolution EpisodeForkResolution) Fork() bool {
	return resolution.Stream != EpisodeForkMainStream
}

// ForkRootDigest digests what makes a conversation the same conversation:
// its system prompt, its tool definitions and its first message, each in the
// canonical form the prefix comparison and the diagnostics use.
func ForkRootDigest(digests *PrefixDigests, system, tools json.RawMessage) string {
	hash := sha256.New()
	for _, part := range [][]byte{canonicalSystem(system), canonicalTools(tools)} {
		hash.Write([]byte(strconv.Itoa(len(part))))
		hash.Write([]byte{':'})
		hash.Write(part)
	}
	if len(digests.Messages()) > 0 {
		first := digests.Canonical(0)
		hash.Write([]byte(strconv.Itoa(len(first))))
		hash.Write([]byte{':'})
		hash.Write(first)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// ForkShellDigest digests a request's system prompt and tool definitions.
func ForkShellDigest(system, tools json.RawMessage) string {
	hash := sha256.New()
	for _, part := range [][]byte{canonicalSystem(system), canonicalTools(tools)} {
		hash.Write([]byte(strconv.Itoa(len(part))))
		hash.Write([]byte{':'})
		hash.Write(part)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// EpisodeForkAnchor names a new sibling: the stream it forked from, the
// request's whole prefix and its root. The same first request of a fork
// (a retry, a second replica) names the same sibling.
func EpisodeForkAnchor(parent string, prefix EpisodeForkPrefix, root string) string {
	digest := sha256.Sum256([]byte(parent + "|" + strconv.Itoa(prefix.Len) + "|" + prefix.Digest + "|" + root))
	return hex.EncodeToString(digest[:8])
}

// SiblingEpisodeID is the raw episode id of rootEpisodeID's sibling anchor.
func SiblingEpisodeID(rootEpisodeID, anchor string) string {
	return rootEpisodeID + EpisodeForkIDSeparator + anchor
}

// ResolveEpisodeFork decides which stream of a root episode a main (or
// unknown) request belongs to. The rule, in order:
//
//  1. Continuation: the request extends the newest admitted prefix of a live
//     stream (the main stream or a sibling): it is that stream's next turn,
//     or a retry of its turn in flight. The longest such prefix wins, the
//     main stream on a tie. It is served there, waiting on that stream's
//     lease as any sequential turn does.
//  2. Otherwise its home is the stream with the longest admitted prefix the
//     request extends, or the main stream when it extends none.
//  3. A home with a turn in flight: fork. The request is parallel to that
//     turn, so it gets a new sibling and never waits on the home's lease.
//  4. A home with nothing admitted: the request starts it.
//  5. An idle home and a request that is a recognised compaction, or keeps
//     the home's root (the same system prompt, tools and first message):
//     rewrite of the home (an edit, a rewind, an unrecognised compaction).
//     It stays on the home, which reads it as today's prefix_break or
//     compaction; the branch it replaces is kept as a dormant sibling, so a
//     branch that was in fact a parallel fork and continues later becomes
//     a sibling instead of rewriting the home back.
//  6. Otherwise (a different system prompt, tool set or first message):
//     fork, on a new sibling.
//
// A new sibling beyond MaxEpisodeForkSiblings evicts the least recently
// admitted one.
func ResolveEpisodeFork(
	view EpisodeForkView,
	request EpisodeForkRequest,
	now time.Time,
	idleTTL time.Duration,
) EpisodeForkResolution {
	streams := map[string]*EpisodeForkStream{}
	if view.Main != nil {
		streams[EpisodeForkMainStream] = view.Main
	}
	for anchor, stream := range view.Siblings {
		if anchor != EpisodeForkMainStream && stream.live(now, idleTTL) {
			streams[anchor] = stream
		}
	}
	names := make([]string, 0, len(streams))
	for name := range streams {
		names = append(names, name)
	}
	// Main first, then siblings in a fixed order, so ties are deterministic.
	sort.Slice(names, func(i, j int) bool {
		if (names[i] == EpisodeForkMainStream) != (names[j] == EpisodeForkMainStream) {
			return names[i] == EpisodeForkMainStream
		}
		return names[i] < names[j]
	})
	siblings := len(streams)
	if view.Main != nil {
		siblings--
	}
	resolution := EpisodeForkResolution{SiblingCount: siblings}

	continued, continuedLen, continuedRoot := "", -1, false
	home, homeLen, homeRoot := EpisodeForkMainStream, -1, false
	branched := false
	for _, name := range names {
		stream := streams[name]
		for index, prefix := range stream.Prefixes {
			if !request.extends(prefix) {
				continue
			}
			latest := index == len(stream.Prefixes)-1
			sameRoot := stream.lastRoot() == request.Root
			if latest && (!stream.Busy(now) || request.nextTurnOf(prefix) && sameRoot) &&
				(prefix.Len > continuedLen || prefix.Len == continuedLen && sameRoot && !continuedRoot) {
				// Longest first; on a tie the stream whose root the request
				// keeps, then the main stream (names are main first).
				continued, continuedLen, continuedRoot = name, prefix.Len, sameRoot
			}
			if prefix.Len > homeLen || prefix.Len == homeLen && sameRoot && !homeRoot {
				home, homeLen, homeRoot = name, prefix.Len, sameRoot
				branched = !latest && request.carriesReply(prefix)
			}
		}
	}
	// A continuation is accepted only when no stream holds a longer prefix
	// the request extends: a shorter idle match must not take a request
	// that departs from a longer, busier stream.
	// On an equal length, a continuation of another root does not take a
	// request from the stream whose root it keeps.
	if continued != "" && (continuedLen > homeLen ||
		continuedLen == homeLen && (continued == home || continuedRoot || !homeRoot)) {
		resolution.Stream, resolution.Parent, resolution.Reason = continued, continued, ForkReasonContinuation
		return resolution
	}
	// An edited retry of an idle stream's newest turn that never committed
	// (the turn failed or was interrupted, and the user changed the message)
	// stays on that stream. When in doubt the request stays: a stream whose
	// commit cannot be read counts as not committed. Beside a turn in flight the same shape is a parallel request
	// (a second run with a new message), which rule 3 forks.
	retried, retriedLen, retriedAt := "", -1, int64(0)
	for _, name := range names {
		stream := streams[name]
		latest, ok := stream.Latest()
		if !ok || stream.Busy(now) || request.Committed[name] || latest.BaseLen < homeLen ||
			latest.BaseLen < retriedLen || !request.retries(latest) {
			continue
		}
		// The longest base; on a tie the turn admitted last, which is the
		// one that just failed.
		if latest.BaseLen == retriedLen && stream.AdmittedAtMS <= retriedAt {
			continue
		}
		retried, retriedLen, retriedAt = name, latest.BaseLen, stream.AdmittedAtMS
	}
	if retried != "" {
		resolution.Stream, resolution.Parent, resolution.Reason = retried, retried, ForkReasonRetryEdit
		return resolution
	}
	if homeLen < 0 && request.Compaction {
		// A compaction replaces the history it would have extended, so no
		// prefix names its stream. It is the stream that spoke last: a
		// harness compacts right after a turn of the conversation it
		// compacts. A stream in flight cannot be the one compacting.
		home = mostRecentIdleStream(names, streams, now)
	}
	resolution.Parent = home
	homeStream := streams[home]
	switch {
	case homeStream.Busy(now):
		resolution.Reason = ForkReasonForkBusy
	case homeStream == nil || len(homeStream.Prefixes) == 0:
		resolution.Stream, resolution.Reason = home, ForkReasonStart
		return resolution
	case request.Compaction:
		// A recognised compaction is the home's own next context.
	case branched:
		resolution.Reason = ForkReasonForkBranch
	case homeStream.Root == "" || homeStream.Root == request.Root || homeStream.lastRoot() == request.Root:
		resolution.Reason = ForkReasonRewrite
	default:
		resolution.Reason = ForkReasonForkRoot
	}
	if resolution.Reason == "" || resolution.Reason == ForkReasonRewrite {
		if resolution.Reason == "" {
			resolution.Reason = ForkReasonCompaction
		}
		resolution.Stream = home
		latest, _ := homeStream.Latest()
		resolution.Displaced = EpisodeForkAnchor(home, latest, homeStream.Root)
		if _, known := streams[resolution.Displaced]; !known {
			resolution.DisplacedRecord = &EpisodeForkStream{
				Prefixes: []EpisodeForkPrefix{latest}, Root: homeStream.Root, AdmittedAtMS: homeStream.AdmittedAtMS,
			}
			resolution.Evict = lruSibling(streams, now, resolution.SiblingCount)
			if resolution.Evict == "" {
				resolution.SiblingCount++
			}
		} else {
			resolution.Displaced = ""
		}
		return resolution
	}
	resolution.Stream = EpisodeForkAnchor(home, request.Prefix(), request.Root)
	if _, known := streams[resolution.Stream]; known {
		// The same fork's first request, again: its retry, or a second
		// replica's copy.
		return resolution
	}
	resolution.New = true
	resolution.Evict = lruSibling(streams, now, resolution.SiblingCount)
	if resolution.Evict == "" {
		resolution.SiblingCount++
	}
	return resolution
}

func (stream *EpisodeForkStream) lastRoot() string {
	if stream.LastRoot != "" {
		return stream.LastRoot
	}
	return stream.Root
}

// mostRecentIdleStream is the idle live stream admitted last, the main
// stream on a tie or when none is idle.
func mostRecentIdleStream(names []string, streams map[string]*EpisodeForkStream, now time.Time) string {
	chosen, chosenAt := EpisodeForkMainStream, int64(-1)
	if main := streams[EpisodeForkMainStream]; main != nil && !main.Busy(now) {
		chosenAt = main.AdmittedAtMS
	}
	for _, name := range names {
		stream := streams[name]
		if name == EpisodeForkMainStream || stream.Busy(now) {
			continue
		}
		if stream.AdmittedAtMS > chosenAt {
			chosen, chosenAt = name, stream.AdmittedAtMS
		}
	}
	return chosen
}

// lruSibling is the sibling to evict before adding one, or "" when there is
// room: the least recently admitted idle sibling, else the least recently
// admitted one.
func lruSibling(streams map[string]*EpisodeForkStream, now time.Time, siblings int) string {
	if siblings < MaxEpisodeForkSiblings {
		return ""
	}
	victim, victimBusy, victimAt := "", true, int64(0)
	for name, stream := range streams {
		if name == EpisodeForkMainStream {
			continue
		}
		busy := stream.Busy(now)
		if victim == "" || (victimBusy && !busy) ||
			busy == victimBusy && (stream.AdmittedAtMS < victimAt || stream.AdmittedAtMS == victimAt && name < victim) {
			victim, victimBusy, victimAt = name, busy, stream.AdmittedAtMS
		}
	}
	return victim
}

func marshalEpisodeForkStream(record *EpisodeForkStream) (string, error) {
	payload, err := json.Marshal(record)
	return string(payload), err
}

func unmarshalEpisodeForkStream(payload string) (*EpisodeForkStream, bool) {
	var record EpisodeForkStream
	if err := json.Unmarshal([]byte(payload), &record); err != nil {
		return nil, false
	}
	return &record, true
}

// ForkView reads a root episode's fork registry, each stream marked busy
// while its episode's lease is held.
func (store *MemoryEpisodeStore) ForkView(_ context.Context, rootEpisodeIDHash string) (EpisodeForkView, error) {
	store.mu.Lock()
	defer store.mu.Unlock()
	view := EpisodeForkView{Siblings: map[string]*EpisodeForkStream{}}
	record := store.forks[rootEpisodeIDHash]
	if record == nil {
		return view, nil
	}
	if store.now().Sub(record.lastAccess) >= store.idleTTL {
		delete(store.forks, rootEpisodeIDHash)
		return view, nil
	}
	for name, stream := range record.streams {
		clone := stream.Clone()
		episode := clone.EpisodeHash
		if episode == "" && name == EpisodeForkMainStream {
			episode = rootEpisodeIDHash
		}
		if entry := store.entries[episode]; entry != nil {
			clone.Leased = entry.leased
		}
		if name == EpisodeForkMainStream {
			view.Main = clone
			continue
		}
		view.Siblings[name] = clone
	}
	return view, nil
}

// RecordForkStream writes one stream of a root episode's fork registry.
func (store *MemoryEpisodeStore) RecordForkStream(
	_ context.Context,
	rootEpisodeIDHash string,
	stream string,
	record *EpisodeForkStream,
	evict string,
) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.forks == nil {
		store.forks = map[string]*memoryForkRecord{}
	}
	now := store.now()
	for hash, existing := range store.forks {
		if now.Sub(existing.lastAccess) >= store.idleTTL {
			delete(store.forks, hash)
		}
	}
	entry := store.forks[rootEpisodeIDHash]
	if entry == nil {
		if len(store.forks) >= store.maxEpisodes {
			oldest, oldestAt := "", now
			for hash, existing := range store.forks {
				if oldest == "" || existing.lastAccess.Before(oldestAt) {
					oldest, oldestAt = hash, existing.lastAccess
				}
			}
			delete(store.forks, oldest)
		}
		entry = &memoryForkRecord{streams: map[string]*EpisodeForkStream{}}
		store.forks[rootEpisodeIDHash] = entry
	}
	entry.lastAccess = now
	stored := record.Clone()
	stored.Leased = false
	entry.streams[stream] = stored
	if evict != "" && evict != stream {
		delete(entry.streams, evict)
	}
	pruneEpisodeForkStreams(entry.streams, stream, now, store.idleTTL)
	return nil
}

// pruneEpisodeForkStreams deletes siblings idle past idleTTL and then, past
// MaxEpisodeForkSiblings, the least recently admitted, never keep (the one
// just written) or the main stream. The Redis store does the same in its
// write script.
func pruneEpisodeForkStreams(streams map[string]*EpisodeForkStream, keep string, now time.Time, idleTTL time.Duration) {
	siblings := 0
	for name, stream := range streams {
		if name == EpisodeForkMainStream || name == keep {
			continue
		}
		if !stream.live(now, idleTTL) {
			delete(streams, name)
			continue
		}
		siblings++
	}
	if keep != EpisodeForkMainStream {
		siblings++
	}
	for ; siblings > MaxEpisodeForkSiblings; siblings-- {
		oldest, oldestAt := "", int64(0)
		for name, stream := range streams {
			if name == EpisodeForkMainStream || name == keep {
				continue
			}
			if oldest == "" || stream.AdmittedAtMS < oldestAt || stream.AdmittedAtMS == oldestAt && name < oldest {
				oldest, oldestAt = name, stream.AdmittedAtMS
			}
		}
		if oldest == "" {
			return
		}
		delete(streams, oldest)
	}
}

type memoryForkRecord struct {
	streams    map[string]*EpisodeForkStream
	lastAccess time.Time
}
