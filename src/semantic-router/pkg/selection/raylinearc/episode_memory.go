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
	"sync"
	"time"
)

type MemoryEpisodeStoreConfig struct {
	MaxEpisodes int
	IdleTTL     time.Duration
	Now         func() time.Time
}

type memoryEpisodeEntry struct {
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
	select {
	case <-ctx.Done():
		return Lease{}, nil, ctx.Err()
	case <-entry.gate:
		acquired = true
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
		entry = &memoryEpisodeEntry{
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
	return true
}
