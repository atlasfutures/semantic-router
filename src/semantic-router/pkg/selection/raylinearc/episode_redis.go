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
	"crypto/tls"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

const redisAcquirePoll = 20 * time.Millisecond

var (
	redisPrepareScript = redis.NewScript(`
if redis.call("SET", KEYS[1], ARGV[1], "NX", "PX", ARGV[2]) then
  local fence = redis.call("INCR", KEYS[2])
  redis.call("PEXPIRE", KEYS[2], ARGV[3])
  local state = redis.call("GET", KEYS[3])
  if not state then
    state = ""
  else
    redis.call("PEXPIRE", KEYS[3], ARGV[3])
  end
  return {fence, state}
end
return {0, ""}
`)
	redisCommitScript = redis.NewScript(`
if redis.call("GET", KEYS[1]) ~= ARGV[1] then return 0 end
if tonumber(redis.call("GET", KEYS[2]) or "-1") ~= tonumber(ARGV[2]) then
  return 0
end
redis.call("SET", KEYS[3], ARGV[3], "PX", ARGV[4])
redis.call("PEXPIRE", KEYS[2], ARGV[4])
redis.call("DEL", KEYS[1])
return 1
`)
	redisAbortScript = redis.NewScript(`
local owner = redis.call("GET", KEYS[1])
if not owner then return 2 end
if owner ~= ARGV[1] then return 0 end
if tonumber(redis.call("GET", KEYS[2]) or "-1") ~= tonumber(ARGV[2]) then
  return 0
end
redis.call("DEL", KEYS[1])
return 1
`)
	redisRenewScript = redis.NewScript(`
if redis.call("GET", KEYS[1]) ~= ARGV[1] then return 0 end
if tonumber(redis.call("GET", KEYS[2]) or "-1") ~= tonumber(ARGV[2]) then
  return 0
end
redis.call("PEXPIRE", KEYS[1], ARGV[3])
redis.call("PEXPIRE", KEYS[2], ARGV[4])
if redis.call("EXISTS", KEYS[3]) == 1 then
  redis.call("PEXPIRE", KEYS[3], ARGV[4])
end
return 1
`)
)

type RedisEpisodeStoreConfig struct {
	Address   string
	DB        int
	Username  string
	Password  string
	UseTLS    bool
	PoolSize  int
	KeyPrefix string
	LeaseTTL  time.Duration
	IdleTTL   time.Duration
	Now       func() time.Time
}

type RedisEpisodeStore struct {
	client    *redis.Client
	keyPrefix string
	leaseTTL  time.Duration
	idleTTL   time.Duration
	now       func() time.Time
}

func NewRedisEpisodeStore(
	config RedisEpisodeStoreConfig,
) (*RedisEpisodeStore, error) {
	if config.Address == "" || config.KeyPrefix == "" {
		return nil, errors.New("ARC Redis episode address and prefix are required")
	}
	if config.LeaseTTL <= 0 || config.IdleTTL < config.LeaseTTL {
		return nil, errors.New("ARC Redis episode TTLs are invalid")
	}
	options := &redis.Options{
		Addr:     config.Address,
		DB:       config.DB,
		Username: config.Username,
		Password: config.Password,
		PoolSize: config.PoolSize,
	}
	if config.UseTLS {
		options.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	}
	now := config.Now
	if now == nil {
		now = time.Now
	}
	return &RedisEpisodeStore{
		client:    redis.NewClient(options),
		keyPrefix: config.KeyPrefix,
		leaseTTL:  config.LeaseTTL,
		idleTTL:   config.IdleTTL,
		now:       now,
	}, nil
}

func (store *RedisEpisodeStore) Prepare(
	ctx context.Context,
	episodeIDHash string,
	workerCount int,
) (Lease, *EpisodeState, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) ||
		workerCount <= 0 {
		return Lease{}, nil, errors.New("invalid ARC Redis prepare request")
	}
	owner, err := newEpisodeOwnerToken()
	if err != nil {
		return Lease{}, nil, err
	}
	keys := store.keys(episodeIDHash)
	for {
		lease, state, acquired, err := store.tryPrepare(
			ctx,
			keys,
			episodeIDHash,
			owner,
			workerCount,
		)
		if err != nil || acquired {
			return lease, state, err
		}
		timer := time.NewTimer(redisAcquirePoll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return Lease{}, nil, ctx.Err()
		case <-timer.C:
		}
	}
}

func (store *RedisEpisodeStore) Commit(
	ctx context.Context,
	lease Lease,
	expectedVersion uint64,
	state *EpisodeState,
) error {
	if expectedVersion != lease.version {
		return ErrEpisodeLeaseLost
	}
	payload, err := marshalEpisodeState(state, expectedVersion, store.now())
	if err != nil {
		return err
	}
	keys := store.keys(lease.episodeIDHash)
	result, err := redisCommitScript.Run(
		ctx,
		store.client,
		keys,
		lease.ownerToken,
		expectedVersion,
		payload,
		store.idleTTL.Milliseconds(),
	).Int()
	if err != nil {
		return boundedRedisEpisodeError("commit", err)
	}
	if result != 1 {
		return ErrEpisodeLeaseLost
	}
	return nil
}

func (store *RedisEpisodeStore) Abort(
	ctx context.Context,
	lease Lease,
) error {
	keys := store.keys(lease.episodeIDHash)
	result, err := redisAbortScript.Run(
		ctx,
		store.client,
		keys[:2],
		lease.ownerToken,
		lease.version,
	).Int()
	if err != nil {
		return boundedRedisEpisodeError("abort", err)
	}
	if result == 0 {
		return ErrEpisodeLeaseLost
	}
	return nil
}

func (store *RedisEpisodeStore) Renew(
	ctx context.Context,
	lease Lease,
) error {
	keys := store.keys(lease.episodeIDHash)
	// Renewal is episode activity: extend the lease, and keep the fence and
	// any committed state alive for the idle window, matching the memory
	// backend's idle semantics.
	result, err := redisRenewScript.Run(
		ctx,
		store.client,
		keys,
		lease.ownerToken,
		lease.version,
		store.leaseTTL.Milliseconds(),
		store.idleTTL.Milliseconds(),
	).Int()
	if err != nil {
		return boundedRedisEpisodeError("renew", err)
	}
	if result != 1 {
		return ErrEpisodeLeaseLost
	}
	return nil
}

func (store *RedisEpisodeStore) Ready(ctx context.Context) error {
	if store == nil || store.client == nil {
		return errors.New("ARC Redis episode store is nil")
	}
	if err := store.client.Ping(ctx).Err(); err != nil {
		return boundedRedisEpisodeError("probe", err)
	}
	return nil
}

func (store *RedisEpisodeStore) Close() error {
	if store == nil || store.client == nil {
		return nil
	}
	return store.client.Close()
}

func (store *RedisEpisodeStore) tryPrepare(
	ctx context.Context,
	keys []string,
	episodeIDHash string,
	owner string,
	workerCount int,
) (Lease, *EpisodeState, bool, error) {
	raw, err := redisPrepareScript.Run(
		ctx,
		store.client,
		keys,
		owner,
		store.leaseTTL.Milliseconds(),
		store.idleTTL.Milliseconds(),
	).Slice()
	if err != nil {
		return Lease{}, nil, false, boundedRedisEpisodeError("prepare", err)
	}
	if len(raw) != 2 {
		return Lease{}, nil, false, errors.New(
			"ARC Redis prepare response contract mismatch",
		)
	}
	version, err := redisResultUint64(raw[0])
	if err != nil {
		return Lease{}, nil, false, err
	}
	if version == 0 {
		return Lease{}, nil, false, nil
	}
	payload, ok := raw[1].(string)
	if !ok {
		return Lease{}, nil, false, errors.New(
			"ARC Redis state response contract mismatch",
		)
	}
	state, storedVersion, err := unmarshalEpisodeState(
		[]byte(payload),
		workerCount,
		store.now(),
	)
	if err != nil {
		store.releaseFailedPrepare(ctx, keys, owner, version)
		return Lease{}, nil, false, err
	}
	if storedVersion >= version {
		store.releaseFailedPrepare(ctx, keys, owner, version)
		return Lease{}, nil, false, errors.New(
			"ARC Redis state fence is invalid",
		)
	}
	return Lease{
		episodeIDHash: episodeIDHash,
		ownerToken:    owner,
		version:       version,
	}, state, true, nil
}

func (store *RedisEpisodeStore) releaseFailedPrepare(
	_ context.Context,
	keys []string,
	owner string,
	version uint64,
) {
	ctx, cancel := context.WithTimeout(
		context.Background(),
		episodeFinalizeTimeout,
	)
	defer cancel()
	_, _ = redisAbortScript.Run(
		ctx,
		store.client,
		keys[:2],
		owner,
		version,
	).Result()
}

const episodeFinalizeTimeout = 2 * time.Second

func (store *RedisEpisodeStore) keys(
	episodeIDHash string,
) []string {
	base := store.keyPrefix + episodeIDHash
	return []string{
		base + ":lease",
		base + ":fence",
		base + ":state",
	}
}

func redisResultUint64(value any) (uint64, error) {
	switch typed := value.(type) {
	case int64:
		if typed <= 0 {
			return 0, nil
		}
		return uint64(typed), nil // #nosec G115 -- positive Redis integer.
	case string:
		parsed, err := strconv.ParseUint(typed, 10, 64)
		if err == nil {
			return parsed, nil
		}
	}
	return 0, errors.New("ARC Redis fence response contract mismatch")
}

func boundedRedisEpisodeError(stage string, err error) error {
	if errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return fmt.Errorf("ARC Redis episode %s failed", stage)
}
