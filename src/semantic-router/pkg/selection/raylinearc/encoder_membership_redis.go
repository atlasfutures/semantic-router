/*
Copyright 2026 vLLM Semantic Router.

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
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"
)

const encoderMembershipKeySuffix = "encoder-membership"

var redisMembershipCompareAndSwapScript = redis.NewScript(`
local current = redis.call("GET", KEYS[1])
if not current then
  if tonumber(ARGV[1]) ~= 0 then return 0 end
  redis.call("SET", KEYS[1], ARGV[2])
  return 1
end
local decoded = cjson.decode(current)
if tonumber(decoded["revision"] or "0") ~= tonumber(ARGV[1]) then return 0 end
redis.call("SET", KEYS[1], ARGV[2])
return 1
`)

// RedisEncoderMembershipSource keeps the controller document alongside the
// fenced episode state. It deliberately has no TTL: losing a membership
// document must fail readiness rather than silently reverting a fleet.
type RedisEncoderMembershipSource struct {
	store *RedisEpisodeStore
}

func NewRedisEncoderMembershipSource(
	store *RedisEpisodeStore,
) (*RedisEncoderMembershipSource, error) {
	if store == nil || store.client == nil || store.keyPrefix == "" {
		return nil, errors.New("ARC Redis membership source is unavailable")
	}
	return &RedisEncoderMembershipSource{store: store}, nil
}

func (source *RedisEncoderMembershipSource) Load(
	ctx context.Context,
) (EncoderMembershipSnapshot, error) {
	if source == nil || source.store == nil || source.store.client == nil {
		return EncoderMembershipSnapshot{}, errors.New("ARC Redis membership source is unavailable")
	}
	payload, err := source.store.client.Get(ctx, source.key()).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return EncoderMembershipSnapshot{}, ErrEncoderMembershipNotFound
		}
		return EncoderMembershipSnapshot{}, boundedRedisMembershipError("load", err)
	}
	var snapshot EncoderMembershipSnapshot
	if err := json.Unmarshal(payload, &snapshot); err != nil {
		return EncoderMembershipSnapshot{}, errors.New("ARC Redis membership document is invalid")
	}
	if err := validateEncoderMembershipSnapshot(snapshot); err != nil {
		return EncoderMembershipSnapshot{}, err
	}
	return cloneEncoderMembershipSnapshot(snapshot), nil
}

func (source *RedisEncoderMembershipSource) CompareAndSwap(
	ctx context.Context,
	expectedRevision uint64,
	next EncoderMembershipSnapshot,
) error {
	if source == nil || source.store == nil || source.store.client == nil {
		return errors.New("ARC Redis membership source is unavailable")
	}
	if err := validateEncoderMembershipSnapshot(next); err != nil {
		return err
	}
	if next.Revision != expectedRevision+1 {
		return errors.New("ARC encoder membership revision must advance by one")
	}
	payload, err := json.Marshal(next)
	if err != nil {
		return errors.New("ARC encoder membership document cannot be encoded")
	}
	updated, err := redisMembershipCompareAndSwapScript.Run(
		ctx,
		source.store.client,
		[]string{source.key()},
		expectedRevision,
		payload,
	).Int()
	if err != nil {
		return boundedRedisMembershipError("compare_and_swap", err)
	}
	if updated != 1 {
		return ErrEncoderMembershipConflict
	}
	return nil
}

func (source *RedisEncoderMembershipSource) key() string {
	return source.store.keyPrefix + encoderMembershipKeySuffix
}

func (source *RedisEncoderMembershipSource) Close() error {
	if source == nil || source.store == nil {
		return nil
	}
	return source.store.Close()
}

func boundedRedisMembershipError(stage string, err error) error {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return fmt.Errorf("ARC Redis membership %s failed", stage)
}
