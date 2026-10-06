package raylinearc

import (
	"context"
	"errors"
	"time"

	"github.com/redis/go-redis/v9"
)

// A two-stage policy package will not decide a mid-conversation turn cold:
// with no attributed assistant message and no episode memory it refuses
// (selection_refused, stage_one_held_unknown) until the caller holds the
// model serving the conversation. The episode remembers that model only for
// its idle TTL, so a resume after the episode cooled, or after an eviction,
// has lost it.
//
// ServedWorkerStore reads, per episode, the worker that last served a
// committed turn, kept for ServedWorkerTTL -- far longer than the episode.
//
// The record has no writer of its own. A turn sets EpisodeState.ServedWorker
// on the state it commits, and the episode commit writes the record in the
// same step: one Redis script with the episode's keys, or under the memory
// store's lock. A record is therefore ordered exactly as the commits are --
// by the store, with no clock and no second round trip -- and it exists only
// for a turn whose commit succeeded. A failed or conflicting commit writes
// none.

// ServedWorkerTTL is how long the last served worker outlives its episode.
const ServedWorkerTTL = 7 * 24 * time.Hour

// ServedWorkerStore reads the worker that last served an episode, or "" when
// none is recorded.
type ServedWorkerStore interface {
	LastServedWorker(ctx context.Context, episodeIDHash string) (string, error)
}

func (store *RedisEpisodeStore) servedKey(episodeIDHash string) string {
	return store.keyPrefix + episodeIDHash + ":served"
}

// LastServedWorker reads the recorded worker, or "" when none is.
func (store *RedisEpisodeStore) LastServedWorker(ctx context.Context, episodeIDHash string) (string, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return "", errors.New("invalid ARC served-worker read")
	}
	worker, err := store.client.Get(ctx, store.servedKey(episodeIDHash)).Result()
	if errors.Is(err, redis.Nil) {
		return "", nil
	}
	if err != nil {
		return "", boundedRedisEpisodeError("served read", err)
	}
	return worker, nil
}

// commitKeys are the episode's keys and its served-worker key, for the
// commit scripts that write both.
func (store *RedisEpisodeStore) commitKeys(episodeIDHash string) []string {
	return append(store.keys(episodeIDHash), store.servedKey(episodeIDHash))
}

type memoryServedWorker struct {
	worker  string
	expires time.Time
}

// recordServedLocked writes the served-worker record of a successful commit.
// The caller holds store.mu. The memory store keeps at most its episode
// capacity of records, dropping the one that expires soonest.
func (store *MemoryEpisodeStore) recordServedLocked(episodeIDHash, worker string, now time.Time) {
	if worker == "" {
		return
	}
	if store.served == nil {
		store.served = make(map[string]memoryServedWorker)
	}
	if _, present := store.served[episodeIDHash]; !present && len(store.served) >= store.maxEpisodes {
		oldest := ""
		for key, record := range store.served {
			if !record.expires.After(now) {
				delete(store.served, key)
				continue
			}
			if oldest == "" || record.expires.Before(store.served[oldest].expires) {
				oldest = key
			}
		}
		if len(store.served) >= store.maxEpisodes && oldest != "" {
			delete(store.served, oldest)
		}
	}
	store.served[episodeIDHash] = memoryServedWorker{worker: worker, expires: now.Add(ServedWorkerTTL)}
}

// SeedServedWorker records a served worker as a committed turn would, without
// an episode: for tests and for carrying records into a fresh store.
func (store *MemoryEpisodeStore) SeedServedWorker(episodeIDHash, worker string) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || worker == "" {
		return errors.New("invalid ARC served-worker record")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.recordServedLocked(episodeIDHash, worker, store.now())
	return nil
}

// LastServedWorker reads the recorded worker, or "" when none is.
func (store *MemoryEpisodeStore) LastServedWorker(_ context.Context, episodeIDHash string) (string, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return "", errors.New("invalid ARC served-worker read")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	record, ok := store.served[episodeIDHash]
	if !ok || !record.expires.After(store.now()) {
		return "", nil
	}
	return record.worker, nil
}
