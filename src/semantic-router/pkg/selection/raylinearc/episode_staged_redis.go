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

	"github.com/redis/go-redis/v9"
)

// EpisodeStagedTurnStore keeps a delta cell's staged-turn record (ADR 0155)
// beside the episode, so the next turn finds it whichever cell instance
// serves it. The record is opaque here: a few digests and counts the cell
// encodes, never conversation content. It lives under its own key, outside
// the episode state's schema, so an image that does not know it ignores it.
//
// Every write is fenced by the episode turn the record belongs to: a writer
// never replaces or deletes a record of a later turn, so a turn that lost
// its lease cannot undo what the turn after it recorded.
type EpisodeStagedTurnStore interface {
	// LoadStagedTurn answers the episode's record, or nil when it has none.
	LoadStagedTurn(ctx context.Context, episodeIDHash string) ([]byte, error)
	// StoreStagedTurn replaces the episode's record with turn's, unless it
	// holds a later turn's; it expires with the episode's idle TTL.
	StoreStagedTurn(ctx context.Context, episodeIDHash string, turn uint64, record []byte) error
	// DeleteStagedTurn deletes the episode's record if it belongs to turn or
	// an earlier one, and, when match is set, only if it equals match.
	DeleteStagedTurn(ctx context.Context, episodeIDHash string, turn uint64, match []byte) error
}

// maxStagedTurnRecordBytes bounds a record read back, whoever wrote it.
const maxStagedTurnRecordBytes = 4096

// The record is a hash of the turn it belongs to and its bytes. A field that
// does not parse counts as turn 0, the earliest.
var redisStagedTurnStoreScript = redis.NewScript(`
local held = tonumber(redis.call("HGET", KEYS[1], "turn") or "0") or 0
if held > tonumber(ARGV[1]) then return 0 end
redis.call("DEL", KEYS[1])
redis.call("HSET", KEYS[1], "turn", ARGV[1], "record", ARGV[2])
redis.call("PEXPIRE", KEYS[1], ARGV[3])
return 1
`)

var redisStagedTurnDeleteScript = redis.NewScript(`
local held = tonumber(redis.call("HGET", KEYS[1], "turn") or "0") or 0
if held > tonumber(ARGV[1]) then return 0 end
if ARGV[2] == "1" and redis.call("HGET", KEYS[1], "record") ~= ARGV[3] then return 0 end
return redis.call("DEL", KEYS[1])
`)

func (store *RedisEpisodeStore) stagedTurnKey(episodeIDHash string) string {
	return store.keyPrefix + episodeIDHash + ":staged"
}

// LoadStagedTurn reads the episode's staged-turn record.
func (store *RedisEpisodeStore) LoadStagedTurn(ctx context.Context, episodeIDHash string) ([]byte, error) {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return nil, errors.New("invalid ARC Redis staged turn request")
	}
	record, err := store.client.HGet(ctx, store.stagedTurnKey(episodeIDHash), "record").Bytes()
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

// StoreStagedTurn writes the episode's staged-turn record for turn.
func (store *RedisEpisodeStore) StoreStagedTurn(ctx context.Context, episodeIDHash string, turn uint64, record []byte) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) || len(record) == 0 || len(record) > maxStagedTurnRecordBytes {
		return errors.New("invalid ARC Redis staged turn write")
	}
	err := redisStagedTurnStoreScript.Run(ctx, store.client, []string{store.stagedTurnKey(episodeIDHash)},
		strconv.FormatUint(turn, 10), record, store.idleTTL.Milliseconds()).Err()
	if err != nil {
		return boundedRedisEpisodeError("staged turn write", err)
	}
	return nil
}

// DeleteStagedTurn deletes the episode's staged-turn record of turn or an
// earlier one.
func (store *RedisEpisodeStore) DeleteStagedTurn(ctx context.Context, episodeIDHash string, turn uint64, match []byte) error {
	if store == nil || !validEpisodeIDHash(episodeIDHash) {
		return errors.New("invalid ARC Redis staged turn delete")
	}
	matched, value := "0", []byte{}
	if match != nil {
		matched, value = "1", match
	}
	err := redisStagedTurnDeleteScript.Run(ctx, store.client, []string{store.stagedTurnKey(episodeIDHash)},
		strconv.FormatUint(turn, 10), matched, value).Err()
	if err != nil {
		return boundedRedisEpisodeError("staged turn delete", err)
	}
	return nil
}
