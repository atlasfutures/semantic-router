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

	"github.com/redis/go-redis/v9"
)

// EpisodeStagedTurnStore keeps a delta cell's staged-turn record (ADR 0155)
// beside the episode, so the next turn finds it whichever cell instance
// serves it. The record is opaque here: a few digests and counts the cell
// encodes, never conversation content. It lives under its own key, outside
// the episode state's schema, so an image that does not know it ignores it.
type EpisodeStagedTurnStore interface {
	// LoadStagedTurn answers the episode's record, or nil when it has none.
	LoadStagedTurn(ctx context.Context, episodeIDHash string) ([]byte, error)
	// StoreStagedTurn replaces the episode's record; it expires with the
	// episode's idle TTL.
	StoreStagedTurn(ctx context.Context, episodeIDHash string, record []byte) error
	// DeleteStagedTurn deletes the episode's record: any record when match
	// is nil, otherwise only one equal to match.
	DeleteStagedTurn(ctx context.Context, episodeIDHash string, match []byte) error
}

// maxStagedTurnRecordBytes bounds a record read back, whoever wrote it.
const maxStagedTurnRecordBytes = 4096

var redisStagedTurnDeleteIfScript = redis.NewScript(`
if redis.call("GET", KEYS[1]) == ARGV[1] then return redis.call("DEL", KEYS[1]) end
return 0
`)

func (store *RedisEpisodeStore) stagedTurnKey(episodeIDHash string) string {
	return store.keyPrefix + episodeIDHash + ":staged"
}

// LoadStagedTurn reads the episode's staged-turn record.
func (store *RedisEpisodeStore) LoadStagedTurn(ctx context.Context, episodeIDHash string) ([]byte, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return nil, errors.New("invalid ARC Redis staged turn request")
	}
	record, err := store.client.Get(ctx, store.stagedTurnKey(episodeIDHash)).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, boundedRedisEpisodeError("staged turn read", err)
	}
	if len(record) > maxStagedTurnRecordBytes {
		return nil, errors.New("ARC Redis staged turn exceeds size limit")
	}
	return record, nil
}

// StoreStagedTurn writes the episode's staged-turn record.
func (store *RedisEpisodeStore) StoreStagedTurn(ctx context.Context, episodeIDHash string, record []byte) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || len(record) == 0 || len(record) > maxStagedTurnRecordBytes {
		return errors.New("invalid ARC Redis staged turn write")
	}
	if err := store.client.Set(ctx, store.stagedTurnKey(episodeIDHash), record, store.idleTTL).Err(); err != nil {
		return boundedRedisEpisodeError("staged turn write", err)
	}
	return nil
}

// DeleteStagedTurn deletes the episode's staged-turn record.
func (store *RedisEpisodeStore) DeleteStagedTurn(ctx context.Context, episodeIDHash string, match []byte) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return errors.New("invalid ARC Redis staged turn delete")
	}
	key := store.stagedTurnKey(episodeIDHash)
	var err error
	if match == nil {
		err = store.client.Del(ctx, key).Err()
	} else {
		err = redisStagedTurnDeleteIfScript.Run(ctx, store.client, []string{key}, match).Err()
	}
	if err != nil {
		return boundedRedisEpisodeError("staged turn delete", err)
	}
	return nil
}
