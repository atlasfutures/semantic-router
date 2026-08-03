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
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

const (
	episodeStateSchemaV1 = "rayline.arc.episode-state.v1"
	episodeStateSchema   = "rayline.arc.episode-state.v2"
	maxFutureClockSkew   = 5 * time.Minute
	episodeOwnerBytes    = 24
	maxEpisodeStateBytes = 64 * 1024
)

var (
	ErrEpisodeLeaseLost = errors.New("ARC episode lease lost")
	ErrEpisodeCapacity  = errors.New("ARC episode store capacity reached")
)

// Lease is an opaque fenced ownership grant. Its owner token and key are kept
// private so callers cannot serialize or log them accidentally.
type Lease struct {
	episodeIDHash string
	ownerToken    string
	version       uint64
}

func (lease Lease) Version() uint64 {
	return lease.version
}

type EpisodeStore interface {
	Prepare(
		context.Context,
		string,
		int,
	) (Lease, *EpisodeState, error)
	Commit(
		context.Context,
		Lease,
		uint64,
		*EpisodeState,
	) error
	Abort(context.Context, Lease) error
}

type EpisodeLeaseRenewer interface {
	Renew(context.Context, Lease) error
}

type EpisodeStoreReadiness interface {
	Ready(context.Context) error
}

type episodeStateWire struct {
	SchemaVersion        string               `json:"schema_version"`
	Version              uint64               `json:"version"`
	PreviousArm          *int                 `json:"previous_arm"`
	TurnIndex            uint64               `json:"turn_index"`
	Warmth               []*episodeWarmthWire `json:"warmth"`
	EncoderOwner         *string              `json:"encoder_owner,omitempty"`
	EncoderVisitedOwners *[]string            `json:"encoder_visited_owners,omitempty"`
}

type episodeWarmthWire struct {
	LastUsedUnixMS  int64 `json:"last_used_unix_ms"`
	LastInputTokens int   `json:"last_input_tokens"`
}

func newEpisodeOwnerToken() (string, error) {
	raw := make([]byte, episodeOwnerBytes)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("create ARC episode owner token: %w", err)
	}
	return hex.EncodeToString(raw), nil
}

func cloneEpisodeState(state *EpisodeState) *EpisodeState {
	if state == nil {
		return nil
	}
	cloned := &EpisodeState{
		PreviousArm:          cloneEpisodeArm(state.PreviousArm),
		TurnIndex:            state.TurnIndex,
		Warmth:               make([]*WorkerWarmth, len(state.Warmth)),
		EncoderOwner:         state.EncoderOwner,
		EncoderVisitedOwners: append([]string(nil), state.EncoderVisitedOwners...),
	}
	for index, warmth := range state.Warmth {
		if warmth == nil {
			continue
		}
		value := *warmth
		cloned.Warmth[index] = &value
	}
	return cloned
}

func cloneEpisodeArm(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func marshalEpisodeState(
	state *EpisodeState,
	version uint64,
	now time.Time,
) ([]byte, error) {
	if err := validatePersistedEpisodeState(state, now); err != nil {
		return nil, err
	}
	wire := episodeStateWire{
		SchemaVersion: episodeStateSchema,
		Version:       version,
		PreviousArm:   cloneEpisodeArm(state.PreviousArm),
		TurnIndex:     state.TurnIndex,
		Warmth:        make([]*episodeWarmthWire, len(state.Warmth)),
	}
	owner := state.EncoderOwner
	visited := append([]string{}, state.EncoderVisitedOwners...)
	wire.EncoderOwner = &owner
	wire.EncoderVisitedOwners = &visited
	for index, warmth := range state.Warmth {
		if warmth == nil {
			continue
		}
		wire.Warmth[index] = &episodeWarmthWire{
			LastUsedUnixMS:  warmth.LastUsed.UTC().UnixMilli(),
			LastInputTokens: warmth.LastInputTokens,
		}
	}
	payload, err := json.Marshal(wire)
	if err != nil {
		return nil, errors.New("marshal ARC episode state")
	}
	if len(payload) > maxEpisodeStateBytes {
		return nil, errors.New("ARC episode state exceeds size limit")
	}
	return payload, nil
}

func unmarshalEpisodeState(
	payload []byte,
	workerCount int,
	now time.Time,
) (*EpisodeState, uint64, error) {
	if len(payload) == 0 {
		state, err := NewEpisodeState(workerCount)
		return state, 0, err
	}
	if len(payload) > maxEpisodeStateBytes {
		return nil, 0, errors.New("ARC episode state exceeds size limit")
	}
	var wire episodeStateWire
	if err := decodeStrictJSON(payload, &wire); err != nil {
		return nil, 0, errors.New("decode ARC episode state")
	}
	if len(wire.Warmth) != workerCount {
		return nil, 0, errors.New("ARC episode state contract mismatch")
	}
	owner, visited, err := decodeEpisodeStateAffinity(wire)
	if err != nil {
		return nil, 0, err
	}
	state := episodeStateFromWire(wire, workerCount, owner, visited)
	if err := validatePersistedEpisodeState(state, now); err != nil {
		return nil, 0, err
	}
	return state, wire.Version, nil
}

func decodeEpisodeStateAffinity(
	wire episodeStateWire,
) (string, []string, error) {
	switch wire.SchemaVersion {
	case episodeStateSchemaV1:
		if wire.EncoderOwner != nil || wire.EncoderVisitedOwners != nil {
			return "", nil, errors.New("ARC episode state contract mismatch")
		}
		return "", nil, nil
	case episodeStateSchema:
		if wire.EncoderOwner == nil || wire.EncoderVisitedOwners == nil ||
			*wire.EncoderVisitedOwners == nil {
			return "", nil, errors.New("ARC episode state contract mismatch")
		}
		return *wire.EncoderOwner,
			append([]string(nil), (*wire.EncoderVisitedOwners)...), nil
	default:
		return "", nil, errors.New("ARC episode state contract mismatch")
	}
}

func episodeStateFromWire(
	wire episodeStateWire,
	workerCount int,
	owner string,
	visited []string,
) *EpisodeState {
	state := &EpisodeState{
		PreviousArm:          cloneEpisodeArm(wire.PreviousArm),
		TurnIndex:            wire.TurnIndex,
		Warmth:               make([]*WorkerWarmth, workerCount),
		EncoderOwner:         owner,
		EncoderVisitedOwners: visited,
	}
	for index, warmth := range wire.Warmth {
		if warmth == nil {
			continue
		}
		state.Warmth[index] = &WorkerWarmth{
			LastUsed:        time.UnixMilli(warmth.LastUsedUnixMS).UTC(),
			LastInputTokens: warmth.LastInputTokens,
		}
	}
	return state
}

func validatePersistedEpisodeState(
	state *EpisodeState,
	now time.Time,
) error {
	if now.IsZero() {
		return errors.New("ARC episode store clock is required")
	}
	if state == nil {
		return errors.New("ARC episode state is required")
	}
	if err := validateEpisodeState(state, len(state.Warmth)); err != nil {
		return err
	}
	futureLimit := now.Add(maxFutureClockSkew)
	for _, warmth := range state.Warmth {
		if warmth != nil && warmth.LastUsed.After(futureLimit) {
			return errors.New("ARC episode warmth is implausibly future-dated")
		}
	}
	return nil
}
