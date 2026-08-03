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

	"github.com/redis/go-redis/v9"
)

type encoderMembershipStateWire struct {
	EncoderOwner         *string   `json:"encoder_owner"`
	EncoderVisitedOwners *[]string `json:"encoder_visited_owners"`
}

// CountEncoderReferences scans only bounded state ownership fields so a
// membership controller can establish drain completion without exposing an
// episode identifier or request content. Any malformed persisted state fails
// the operation closed instead of being treated as drained.
func (store *RedisEpisodeStore) CountEncoderReferences(
	ctx context.Context,
	replicaIDs []string,
) (map[string]int, error) {
	if store == nil || store.client == nil || len(replicaIDs) == 0 {
		return nil, errors.New("ARC Redis membership reference request is invalid")
	}
	targets, counts, err := encoderMembershipReferenceTargets(replicaIDs)
	if err != nil {
		return nil, err
	}
	iterator := store.client.Scan(ctx, 0, store.keyPrefix+"*:state", 128).Iterator()
	for iterator.Next(ctx) {
		references, referenceErr := store.encoderMembershipStateReferences(
			ctx,
			iterator.Val(),
		)
		if referenceErr != nil {
			return nil, referenceErr
		}
		for replicaID := range references {
			if _, wanted := targets[replicaID]; wanted {
				counts[replicaID]++
			}
		}
	}
	if err := iterator.Err(); err != nil {
		return nil, boundedRedisMembershipError("scan_state", err)
	}
	return counts, nil
}

func encoderMembershipReferenceTargets(
	replicaIDs []string,
) (map[string]struct{}, map[string]int, error) {
	targets := make(map[string]struct{}, len(replicaIDs))
	counts := make(map[string]int, len(replicaIDs))
	for _, replicaID := range replicaIDs {
		if !validEncoderReplicaID(replicaID) {
			return nil, nil, errors.New("ARC Redis membership replica ID is invalid")
		}
		targets[replicaID] = struct{}{}
		counts[replicaID] = 0
	}
	return targets, counts, nil
}

func (store *RedisEpisodeStore) encoderMembershipStateReferences(
	ctx context.Context,
	key string,
) (map[string]struct{}, error) {
	payload, err := store.client.Get(ctx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, boundedRedisMembershipError("scan_state", err)
	}
	return decodeEncoderMembershipStateReferences(payload)
}

func decodeEncoderMembershipStateReferences(
	payload []byte,
) (map[string]struct{}, error) {
	var state encoderMembershipStateWire
	if err := json.Unmarshal(payload, &state); err != nil {
		return nil, errors.New("ARC Redis membership state is invalid")
	}
	references := make(map[string]struct{}, 1)
	if state.EncoderOwner != nil {
		references[*state.EncoderOwner] = struct{}{}
	}
	if state.EncoderVisitedOwners != nil {
		for _, replicaID := range *state.EncoderVisitedOwners {
			references[replicaID] = struct{}{}
		}
	}
	return references, nil
}
