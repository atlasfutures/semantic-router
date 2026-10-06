package raylinearc

import (
	"context"
	"errors"
	"time"

	"github.com/redis/go-redis/v9"
)

// A two-stage policy package will not decide a mid-conversation turn cold:
// with no attributed assistant message and no episode memory it refuses
// (selection_refused, stage_one_held_unknown) until the caller names the
// model that is serving the conversation. The episode remembers that model
// only for its idle TTL, so a resume after the episode cooled, or after an
// eviction, has lost it.
//
// ServedWorkerStore keeps, per episode, the worker that last served a
// committed turn, for far longer than the episode itself. It is written by
// the router alone (the episode key is the gateway's), so a client cannot
// set it, and it is read only to answer that refusal.

// ServedWorkerTTL is how long the last served worker outlives its episode.
const ServedWorkerTTL = 7 * 24 * time.Hour

// ServedWorkerStore records and reads the worker that last served an
// episode. LastServedWorker returns "" when none is recorded.
type ServedWorkerStore interface {
	RecordServedWorker(ctx context.Context, episodeIDHash, worker string) error
	LastServedWorker(ctx context.Context, episodeIDHash string) (string, error)
}

func (store *RedisEpisodeStore) servedKey(episodeIDHash string) string {
	return store.keyPrefix + episodeIDHash + ":served"
}

// RecordServedWorker stores the worker with ServedWorkerTTL.
func (store *RedisEpisodeStore) RecordServedWorker(ctx context.Context, episodeIDHash, worker string) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || worker == "" {
		return errors.New("invalid ARC served-worker record")
	}
	if err := store.client.Set(ctx, store.servedKey(episodeIDHash), worker, ServedWorkerTTL).Err(); err != nil {
		return boundedRedisEpisodeError("served record", err)
	}
	return nil
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

type memoryServedWorker struct {
	worker  string
	expires time.Time
}

// RecordServedWorker stores the worker with ServedWorkerTTL. The memory
// store keeps at most its episode capacity of records, dropping the one that
// expires soonest.
func (store *MemoryEpisodeStore) RecordServedWorker(_ context.Context, episodeIDHash, worker string) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || worker == "" {
		return errors.New("invalid ARC served-worker record")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	now := store.now()
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
