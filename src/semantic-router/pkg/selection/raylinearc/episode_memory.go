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
	"errors"
	"strconv"
	"strings"
	"sync"
	"time"
)

type MemoryEpisodeStoreConfig struct {
	MaxEpisodes int
	IdleTTL     time.Duration
	Now         func() time.Time
}

type memoryEpisodeEntry struct {
	// generation is unique for the life of the store; a recreated entry never
	// repeats it, so a relaxed read of an evicted entry can never commit to
	// its replacement.
	generation uint64
	gate       chan struct{}
	state      *EpisodeState
	version    uint64
	ownerToken string
	leased     bool
	references int
	lastAccess time.Time
}

type MemoryEpisodeStore struct {
	mu          sync.Mutex
	entries     map[string]*memoryEpisodeEntry
	maxEpisodes int
	idleTTL     time.Duration
	now         func() time.Time
	generations uint64
	// removals counts entries the store has dropped, by reap or capacity
	// eviction. A read that found an episode absent records it: if nothing
	// was removed since, the episode cannot have been created and dropped in
	// between, so it is still the absence that was read.
	removals uint64
}

func NewMemoryEpisodeStore(
	config MemoryEpisodeStoreConfig,
) (*MemoryEpisodeStore, error) {
	if config.MaxEpisodes <= 0 {
		return nil, errors.New("ARC memory episode capacity must be positive")
	}
	if config.IdleTTL <= 0 {
		return nil, errors.New("ARC memory episode idle TTL must be positive")
	}
	now := config.Now
	if now == nil {
		now = time.Now
	}
	return &MemoryEpisodeStore{
		entries:     make(map[string]*memoryEpisodeEntry),
		maxEpisodes: config.MaxEpisodes,
		idleTTL:     config.IdleTTL,
		now:         now,
	}, nil
}

func (store *MemoryEpisodeStore) Prepare(
	ctx context.Context,
	episodeIDHash string,
	workerCount int,
) (Lease, *EpisodeState, error) {
	if !validEpisodeIDHash(episodeIDHash) || workerCount <= 0 {
		return Lease{}, nil, errors.New("invalid ARC episode prepare request")
	}
	entry, err := store.referenceEntry(episodeIDHash)
	if err != nil {
		return Lease{}, nil, err
	}
	acquired := false
	defer func() {
		if !acquired {
			store.releaseReference(entry)
		}
	}()
	// A caller already gone takes nothing, and nothing held the episode, so
	// the bare context error keeps it out of contention.
	if err := ctx.Err(); err != nil {
		return Lease{}, nil, err
	}
	// Take a free gate before waiting, so running out of time is reported as
	// contention only when another request's lease actually held the gate.
	select {
	case <-entry.gate:
		acquired = true
	default:
		select {
		case <-ctx.Done():
			// Only another request's lease holds the gate, so a wait that
			// runs out of time here is contention.
			return Lease{}, nil, errors.Join(ErrEpisodeLeaseHeld, ctx.Err())
		case <-entry.gate:
			acquired = true
		}
	}
	lease, state, err := store.beginLease(
		episodeIDHash,
		entry,
		workerCount,
	)
	if err != nil {
		entry.gate <- struct{}{}
		store.releaseReference(entry)
		return Lease{}, nil, err
	}
	return lease, state, nil
}

func (store *MemoryEpisodeStore) Commit(
	_ context.Context,
	lease Lease,
	expectedVersion uint64,
	state *EpisodeState,
) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	entry, ok := store.entries[lease.episodeIDHash]
	if !ok || !memoryLeaseMatches(entry, lease, expectedVersion) {
		return ErrEpisodeLeaseLost
	}
	if err := validatePersistedEpisodeState(state, store.now()); err != nil {
		return err
	}
	entry.state = cloneEpisodeState(state)
	entry.lastAccess = store.now()
	store.finishLease(entry)
	return nil
}

func (store *MemoryEpisodeStore) Abort(
	_ context.Context,
	lease Lease,
) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	entry, ok := store.entries[lease.episodeIDHash]
	if !ok || !entry.leased {
		return nil
	}
	if !memoryLeaseMatches(entry, lease, lease.version) {
		return ErrEpisodeLeaseLost
	}
	entry.lastAccess = store.now()
	store.finishLease(entry)
	return nil
}

func (store *MemoryEpisodeStore) Renew(
	_ context.Context,
	lease Lease,
) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	entry, ok := store.entries[lease.episodeIDHash]
	if !ok || !memoryLeaseMatches(entry, lease, lease.version) {
		return ErrEpisodeLeaseLost
	}
	entry.lastAccess = store.now()
	return nil
}

func (store *MemoryEpisodeStore) Ready(context.Context) error {
	if store == nil {
		return errors.New("ARC memory episode store is nil")
	}
	return nil
}

func (store *MemoryEpisodeStore) referenceEntry(
	episodeIDHash string,
) (*memoryEpisodeEntry, error) {
	store.mu.Lock()
	defer store.mu.Unlock()
	now := store.now()
	store.reapLocked(now)
	entry := store.entries[episodeIDHash]
	if entry == nil {
		if len(store.entries) >= store.maxEpisodes {
			if !store.evictOldestUnlocked() {
				return nil, ErrEpisodeCapacity
			}
		}
		store.generations++
		entry = &memoryEpisodeEntry{
			generation: store.generations,
			gate:       make(chan struct{}, 1),
			lastAccess: now,
		}
		entry.gate <- struct{}{}
		store.entries[episodeIDHash] = entry
	}
	entry.references++
	return entry, nil
}

func (store *MemoryEpisodeStore) beginLease(
	episodeIDHash string,
	entry *memoryEpisodeEntry,
	workerCount int,
) (Lease, *EpisodeState, error) {
	store.mu.Lock()
	defer store.mu.Unlock()
	state := entry.state
	if state == nil {
		var err error
		state, err = NewEpisodeState(workerCount)
		if err != nil {
			return Lease{}, nil, err
		}
	} else if len(state.Warmth) != workerCount {
		return Lease{}, nil, errors.New(
			"ARC episode worker count changed",
		)
	}
	owner, err := newEpisodeOwnerToken()
	if err != nil {
		return Lease{}, nil, err
	}
	entry.version++
	entry.ownerToken = owner
	entry.leased = true
	entry.lastAccess = store.now()
	return Lease{
		episodeIDHash: episodeIDHash,
		ownerToken:    owner,
		version:       entry.version,
	}, cloneEpisodeState(state), nil
}

func (store *MemoryEpisodeStore) releaseReference(
	entry *memoryEpisodeEntry,
) {
	store.mu.Lock()
	defer store.mu.Unlock()
	if entry.references > 0 {
		entry.references--
	}
}

func (store *MemoryEpisodeStore) finishLease(
	entry *memoryEpisodeEntry,
) {
	entry.ownerToken = ""
	entry.leased = false
	if entry.references > 0 {
		entry.references--
	}
	entry.gate <- struct{}{}
}

func memoryLeaseMatches(
	entry *memoryEpisodeEntry,
	lease Lease,
	expectedVersion uint64,
) bool {
	return entry.leased &&
		entry.ownerToken == lease.ownerToken &&
		entry.version == lease.version &&
		entry.version == expectedVersion
}

func (store *MemoryEpisodeStore) reapLocked(now time.Time) {
	for key, entry := range store.entries {
		if entry.leased || entry.references > 0 {
			continue
		}
		if now.Sub(entry.lastAccess) >= store.idleTTL {
			delete(store.entries, key)
			store.removals++
		}
	}
}

func (store *MemoryEpisodeStore) evictOldestUnlocked() bool {
	var oldestKey string
	var oldestTime time.Time
	for key, entry := range store.entries {
		if entry.leased || entry.references > 0 {
			continue
		}
		if oldestKey == "" || entry.lastAccess.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.lastAccess
		}
	}
	if oldestKey == "" {
		return false
	}
	delete(store.entries, oldestKey)
	store.removals++
	return true
}

// Snapshot reads the episode and its version without a lease. An episode the
// store has never seen reads as a fresh state at version zero.
func (store *MemoryEpisodeStore) Snapshot(
	_ context.Context,
	episodeIDHash string,
	workerCount int,
) (*EpisodeState, EpisodeReadToken, error) {
	if !validEpisodeIDHash(episodeIDHash) || workerCount <= 0 {
		return nil, EpisodeReadToken{}, errors.New("invalid ARC episode snapshot request")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.reapLocked(store.now())
	entry := store.entries[episodeIDHash]
	if entry == nil || entry.state == nil {
		state, err := NewEpisodeState(workerCount)
		if err != nil {
			return nil, EpisodeReadToken{}, err
		}
		if entry == nil {
			return state, EpisodeReadToken{tag: memoryAbsentTag(store.removals), readAt: store.now()}, nil
		}
		return state, memoryReadToken(entry, store.now()), nil
	}
	if len(entry.state.Warmth) != workerCount {
		return nil, EpisodeReadToken{}, errors.New("ARC episode worker count changed")
	}
	return cloneEpisodeState(entry.state), memoryReadToken(entry, store.now()), nil
}

// memoryAbsentTag names an absent read by the store's removal count.
func memoryAbsentTag(removals uint64) string {
	return "absent:" + strconv.FormatUint(removals, 10)
}

// memoryReadToken names an entry by its generation and version, read at now.
// An entry the store has never seen reads as a token with no generation.
func memoryReadToken(entry *memoryEpisodeEntry, now time.Time) EpisodeReadToken {
	return EpisodeReadToken{version: entry.version, tag: strconv.FormatUint(entry.generation, 10), readAt: now}
}

// CommitIfUnchanged writes state as the episode's next version, provided the
// episode is still at readVersion and no strict lease holds it.
func (store *MemoryEpisodeStore) CommitIfUnchanged(
	_ context.Context,
	episodeIDHash string,
	read EpisodeReadToken,
	state *EpisodeState,
) error {
	if !validEpisodeIDHash(episodeIDHash) {
		return errors.New("invalid ARC episode relaxed commit request")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	now := store.now()
	if staleRead(read, now, store.idleTTL) {
		return ErrEpisodeConflict
	}
	if err := validatePersistedEpisodeState(state, now); err != nil {
		return err
	}
	absentRead := strings.HasPrefix(read.tag, "absent:")
	entry := store.entries[episodeIDHash]
	if entry == nil {
		// Absent then and absent now is the same absence only if no entry
		// was dropped in between; otherwise this episode may have been
		// created and dropped, and the read is stale.
		if !absentRead || read.tag != memoryAbsentTag(store.removals) {
			return ErrEpisodeConflict
		}
		store.reapLocked(now)
		if len(store.entries) >= store.maxEpisodes && !store.evictOldestUnlocked() {
			return ErrEpisodeCapacity
		}
		store.generations++
		entry = &memoryEpisodeEntry{generation: store.generations, gate: make(chan struct{}, 1), lastAccess: now}
		entry.gate <- struct{}{}
		store.entries[episodeIDHash] = entry
	}
	if entry.leased {
		return ErrEpisodeConflict
	}
	// A read of an entry the store had not seen may commit only to the entry
	// this commit creates, not one created since.
	if !absentRead || entry.version != 0 || entry.state != nil {
		if read.version != entry.version || read.tag != strconv.FormatUint(entry.generation, 10) {
			return ErrEpisodeConflict
		}
	}
	entry.version++
	entry.state = cloneEpisodeState(state)
	entry.lastAccess = now
	return nil
}
