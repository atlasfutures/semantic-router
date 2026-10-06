package raylinearc

import (
	"context"
	"errors"
	"strconv"
	"strings"
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
//
// A record is written after its turn's episode commit, outside the lease, so
// two turns' writes can land out of order. Each carries the stamp its turn
// took just before committing: while it still held the lease, or, relaxed,
// before a conditional commit that only succeeds after every earlier one.
// A write lands only over an older or equal stamp, so the record names the
// latest committed turn's worker. Stamps come from the replicas' clocks; two
// commits of one episode are a whole request apart, far beyond their skew.
type ServedWorkerStore interface {
	RecordServedWorker(ctx context.Context, episodeIDHash, worker string, stamp time.Time) error
	LastServedWorker(ctx context.Context, episodeIDHash string) (string, error)
}

// servedRecord encodes a record as "<stamp unix microseconds>|<worker>".
// Microseconds, so the Redis script compares stamps exactly: a Lua number is
// a double, exact to 2^53, which nanoseconds since 1970 exceed.
func servedRecord(worker string, stamp time.Time) string {
	return strconv.FormatInt(stamp.UnixMicro(), 10) + "|" + worker
}

func parseServedRecord(record string) (worker string, stamp int64, ok bool) {
	head, worker, found := strings.Cut(record, "|")
	if !found {
		return "", 0, false
	}
	stamp, err := strconv.ParseInt(head, 10, 64)
	return worker, stamp, err == nil && worker != ""
}

// redisRecordServedScript sets the record unless the stored one is newer.
var redisRecordServedScript = redis.NewScript(`
local current = redis.call("GET", KEYS[1])
if current then
  local stamp = tonumber(string.match(current, "^(%d+)|"))
  if stamp and stamp > tonumber(ARGV[1]) then return 0 end
end
redis.call("SET", KEYS[1], ARGV[2], "PX", ARGV[3])
return 1
`)

func (store *RedisEpisodeStore) servedKey(episodeIDHash string) string {
	return store.keyPrefix + episodeIDHash + ":served"
}

// RecordServedWorker stores the worker with ServedWorkerTTL unless a newer
// stamp is already recorded.
func (store *RedisEpisodeStore) RecordServedWorker(ctx context.Context, episodeIDHash, worker string, stamp time.Time) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || worker == "" || strings.Contains(worker, "|") || stamp.IsZero() {
		return errors.New("invalid ARC served-worker record")
	}
	err := redisRecordServedScript.Run(ctx, store.client, []string{store.servedKey(episodeIDHash)},
		stamp.UnixMicro(), servedRecord(worker, stamp), ServedWorkerTTL.Milliseconds()).Err()
	if err != nil {
		return boundedRedisEpisodeError("served record", err)
	}
	return nil
}

// LastServedWorker reads the recorded worker, or "" when none is.
func (store *RedisEpisodeStore) LastServedWorker(ctx context.Context, episodeIDHash string) (string, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return "", errors.New("invalid ARC served-worker read")
	}
	record, err := store.client.Get(ctx, store.servedKey(episodeIDHash)).Result()
	if errors.Is(err, redis.Nil) {
		return "", nil
	}
	if err != nil {
		return "", boundedRedisEpisodeError("served read", err)
	}
	worker, _, _ := parseServedRecord(record)
	return worker, nil
}

type memoryServedWorker struct {
	worker  string
	stamp   int64
	expires time.Time
}

// RecordServedWorker stores the worker with ServedWorkerTTL unless a newer
// stamp is already recorded. The memory store keeps at most its episode
// capacity of records, dropping the one that expires soonest.
func (store *MemoryEpisodeStore) RecordServedWorker(_ context.Context, episodeIDHash, worker string, stamp time.Time) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || worker == "" || stamp.IsZero() {
		return errors.New("invalid ARC served-worker record")
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	now := store.now()
	if current, ok := store.served[episodeIDHash]; ok && current.expires.After(now) && current.stamp > stamp.UnixMicro() {
		return nil
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
	store.served[episodeIDHash] = memoryServedWorker{worker: worker, stamp: stamp.UnixMicro(), expires: now.Add(ServedWorkerTTL)}
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
