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
	"strings"

	"github.com/redis/go-redis/v9"
)

// The fork registry is one hash beside the root episode's keys: a field per
// stream, each a small JSON record, expiring with the episode's idle TTL.
var redisForkRecordScript = redis.NewScript(`
redis.call("HSET", KEYS[1], ARGV[1], ARGV[2])
if ARGV[5] ~= "" then redis.call("HSET", KEYS[1], ARGV[1] .. "|busy", ARGV[5]) end
if ARGV[3] ~= "" and ARGV[3] ~= ARGV[1] then
  redis.call("HDEL", KEYS[1], ARGV[3], ARGV[3] .. "|busy")
end
redis.call("PEXPIRE", KEYS[1], ARGV[4])
return 1
`)

// A finished turn clears only the busy mark its own admission set.
var redisForkClearBusyScript = redis.NewScript(`
local value = redis.call("HGET", KEYS[1], ARGV[1])
if value and string.sub(value, 1, string.len(ARGV[2]) + 1) == ARGV[2] .. "|" then
  redis.call("HDEL", KEYS[1], ARGV[1])
end
return 1
`)

func (store *RedisEpisodeStore) forkKey(rootEpisodeIDHash string) string {
	return store.keyPrefix + rootEpisodeIDHash + ":forks"
}

// ForkView reads a root episode's fork registry. A record that does not
// decode is skipped: the registry is advisory.
func (store *RedisEpisodeStore) ForkView(ctx context.Context, rootEpisodeIDHash string) (EpisodeForkView, error) {
	view := EpisodeForkView{Siblings: map[string]*EpisodeForkStream{}}
	if store == nil || !validEpisodeIDHash(rootEpisodeIDHash) {
		return view, errors.New("invalid ARC Redis fork view request")
	}
	fields, err := store.client.HGetAll(ctx, store.forkKey(rootEpisodeIDHash)).Result()
	if err != nil {
		return view, boundedRedisEpisodeError("fork_view", err)
	}
	busy := map[string]string{}
	for name, payload := range fields {
		if stream, isBusy := strings.CutSuffix(name, "|busy"); isBusy {
			busy[stream] = payload
			continue
		}
		record, ok := unmarshalEpisodeForkStream(payload)
		if !ok {
			continue
		}
		if name == EpisodeForkMainStream {
			view.Main = record
			continue
		}
		view.Siblings[name] = record
	}
	for stream, value := range busy {
		record := view.Siblings[stream]
		if stream == EpisodeForkMainStream {
			record = view.Main
		}
		if record != nil {
			record.BusyToken, record.BusyUntilMS, _ = parseEpisodeForkBusy(value)
		}
	}
	return view, nil
}

// RecordForkStream writes one stream of a root episode's fork registry.
func (store *RedisEpisodeStore) RecordForkStream(
	ctx context.Context,
	rootEpisodeIDHash string,
	stream string,
	record *EpisodeForkStream,
	evict string,
) error {
	if store == nil || !validEpisodeIDHash(rootEpisodeIDHash) || stream == "" || record == nil {
		return errors.New("invalid ARC Redis fork record request")
	}
	payload, err := marshalEpisodeForkStream(record)
	if err != nil {
		return err
	}
	if err := redisForkRecordScript.Run(
		ctx, store.client, []string{store.forkKey(rootEpisodeIDHash)},
		stream, payload, evict, store.idleTTL.Milliseconds(), episodeForkBusyValue(record),
	).Err(); err != nil {
		return boundedRedisEpisodeError("fork_record", err)
	}
	return nil
}

// ClearForkBusy removes stream's busy mark if token set it.
func (store *RedisEpisodeStore) ClearForkBusy(ctx context.Context, rootEpisodeIDHash, stream, token string) error {
	if store == nil || !validEpisodeIDHash(rootEpisodeIDHash) || token == "" {
		return errors.New("invalid ARC Redis fork busy request")
	}
	if err := redisForkClearBusyScript.Run(
		ctx, store.client, []string{store.forkKey(rootEpisodeIDHash)}, EpisodeForkBusyField(stream), token,
	).Err(); err != nil {
		return boundedRedisEpisodeError("fork_clear_busy", err)
	}
	return nil
}
