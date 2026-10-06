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
// ServedModelStore reads, per episode, the model that last served a
// committed turn, kept for ServedModelTTL -- far longer than the episode.
//
// The record has no writer of its own. A turn sets EpisodeState.ServedModel
// on the state it commits, and the episode commit writes the record in the
// same step: one Redis script with the episode's keys, or under the memory
// store's lock. A record is therefore ordered exactly as the commits are --
// by the store, with no clock and no second round trip -- and it exists only
// for a turn whose commit succeeded. A failed or conflicting commit writes
// none.

// ServedModelTTL is how long the last served model outlives its episode.
const ServedModelTTL = 7 * 24 * time.Hour

// ServedModelStore reads the model that last served an episode, or "" when
// none is recorded.
type ServedModelStore interface {
	LastServedModel(ctx context.Context, episodeIDHash string) (string, error)
}

func (store *RedisEpisodeStore) servedKey(episodeIDHash string) string {
	return store.keyPrefix + episodeIDHash + ":served"
}

// LastServedModel reads the recorded model, or "" when none is.
func (store *RedisEpisodeStore) LastServedModel(ctx context.Context, episodeIDHash string) (string, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return "", errors.New("invalid ARC served-model read")
	}
	model, err := store.client.Get(ctx, store.servedKey(episodeIDHash)).Result()
	if errors.Is(err, redis.Nil) {
		return "", nil
	}
	if err != nil {
		return "", boundedRedisEpisodeError("served read", err)
	}
	return model, nil
}

// commitKeys are the episode's keys and its served-model key, for the
// commit scripts that write both.
func (store *RedisEpisodeStore) commitKeys(episodeIDHash string) []string {
	return append(store.keys(episodeIDHash), store.servedKey(episodeIDHash))
}

type memoryServedModel struct {
	model   string
	expires time.Time
}

// recordServedLocked writes the served-model record of a successful commit.
// The caller holds store.mu. The memory store keeps at most its episode
// capacity of records, dropping the one that expires soonest.
func (store *MemoryEpisodeStore) recordServedLocked(episodeIDHash, model string, now time.Time) {
	if model == "" {
		return
	}
	if store.served == nil {
		store.served = make(map[string]memoryServedModel)
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
	store.served[episodeIDHash] = memoryServedModel{model: model, expires: now.Add(ServedModelTTL)}
}

// SeedServedModel records a served model as a committed turn would, without
// an episode: for tests and for carrying records into a fresh store.
func (store *MemoryEpisodeStore) SeedServedModel(episodeIDHash, model string) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || model == "" {
		return errors.New("invalid ARC served-model record")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.recordServedLocked(episodeIDHash, model, store.now())
	return nil
}

// LastServedModel reads the recorded model, or "" when none is.
func (store *MemoryEpisodeStore) LastServedModel(_ context.Context, episodeIDHash string) (string, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return "", errors.New("invalid ARC served-model read")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	record, ok := store.served[episodeIDHash]
	if !ok || !record.expires.After(store.now()) {
		return "", nil
	}
	return record.model, nil
}
