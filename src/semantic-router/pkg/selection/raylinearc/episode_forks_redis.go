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

// The fork registry is one hash beside the root episode's keys: a field per
// stream, each a small JSON record, expiring with the episode's idle TTL.
// A write also deletes siblings idle past the TTL and, past the sibling cap,
// the least recently admitted ones (pruneEpisodeForkStreams), so the hash
// stays bounded whatever concurrent writers raced on. A field that does not
// decode is deleted with them.
var redisForkRecordScript = redis.NewScript(`
redis.call("HSET", KEYS[1], ARGV[1], ARGV[2])
if ARGV[3] ~= "" and ARGV[3] ~= ARGV[1] then redis.call("HDEL", KEYS[1], ARGV[3]) end
local now, idle, cap = tonumber(ARGV[5]), tonumber(ARGV[4]), tonumber(ARGV[6])
local fields = redis.call("HGETALL", KEYS[1])
local live = {}
local count = 0
for i = 1, #fields, 2 do
  local name = fields[i]
  if name ~= "main" and name ~= ARGV[1] then
    local ok, record = pcall(cjson.decode, fields[i + 1])
    local at = ok and type(record) == "table" and tonumber(record["t"]) or nil
    if at == nil or now - at >= idle then
      redis.call("HDEL", KEYS[1], name)
    else
      table.insert(live, {name, at})
    end
  end
end
count = #live
if ARGV[1] ~= "main" then count = count + 1 end
if count > cap then
  table.sort(live, function(a, b)
    if a[2] == b[2] then return a[1] < b[1] end
    return a[2] < b[2]
  end)
  for i = 1, count - cap do
    if live[i] then redis.call("HDEL", KEYS[1], live[i][1]) end
  end
end
redis.call("PEXPIRE", KEYS[1], ARGV[4])
return 1
`)

func (store *RedisEpisodeStore) forkKey(rootEpisodeIDHash string) string {
	return store.keyPrefix + rootEpisodeIDHash + ":forks"
}

// ForkView reads a root episode's fork registry, each stream marked busy
// while its episode's lease key exists. A record that does not decode is
// skipped: the registry is advisory.
func (store *RedisEpisodeStore) ForkView(ctx context.Context, rootEpisodeIDHash string) (EpisodeForkView, error) {
	view := EpisodeForkView{Siblings: map[string]*EpisodeForkStream{}}
	if store == nil || !validEpisodeIDHash(rootEpisodeIDHash) {
		return view, errors.New("invalid ARC Redis fork view request")
	}
	fields, err := store.client.HGetAll(ctx, store.forkKey(rootEpisodeIDHash)).Result()
	if err != nil {
		return view, boundedRedisEpisodeError("fork_view", err)
	}
	records := make([]*EpisodeForkStream, 0, len(fields))
	for name, payload := range fields {
		record, ok := unmarshalEpisodeForkStream(payload)
		if !ok {
			continue
		}
		if name == EpisodeForkMainStream && record.EpisodeHash == "" {
			record.EpisodeHash = rootEpisodeIDHash
		}
		if !validEpisodeIDHash(record.EpisodeHash) {
			continue
		}
		records = append(records, record)
		if name == EpisodeForkMainStream {
			view.Main = record
			continue
		}
		view.Siblings[name] = record
	}
	if len(records) == 0 {
		return view, nil
	}
	pipe := store.client.Pipeline()
	leases := make([]*redis.IntCmd, len(records))
	for index, record := range records {
		leases[index] = pipe.Exists(ctx, store.keys(record.EpisodeHash)[0])
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return view, boundedRedisEpisodeError("fork_view", err)
	}
	for index, record := range records {
		record.Leased = leases[index].Val() > 0
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
		stream, payload, evict, store.idleTTL.Milliseconds(), store.now().UnixMilli(), MaxEpisodeForkSiblings,
	).Err(); err != nil {
		return boundedRedisEpisodeError("fork_record", err)
	}
	return nil
}
