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
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"net/http"
	"regexp"
	"slices"
	"sync"
	"time"
)

const (
	// EncoderFailoverSchemaV1 identifies the first production replica
	// membership and failover contract. Configuration must opt into this exact
	// value so future semantics cannot be activated by an in-place reinterpretation.
	EncoderFailoverSchemaV1 = "rayline.arc.encoder-failover.v1"

	EncoderReplicaActive   EncoderReplicaState = "active"
	EncoderReplicaDraining EncoderReplicaState = "draining"

	maxEncoderReplicas       = 8
	maxEncoderReplicaIDBytes = 64
)

var encoderReplicaIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

type EncoderReplicaState string

// EncoderReplica binds a stable, deployment-owned identity to one retained
// encoder client. Draining replicas continue serving episodes they already
// own, but rendezvous assignment excludes them for new episodes.
type EncoderReplica struct {
	ID     string
	State  EncoderReplicaState
	Client *EncoderClient
}

// EncoderPoolConfig is intentionally small and versioned. A remap is allowed
// only after one of the explicitly configured HTTP statuses. Transport,
// timeout, decode, and contract failures remain fail-closed because the first
// replica may already have mutated its retained session.
type EncoderPoolConfig struct {
	SchemaVersion          string
	UnavailableStatusCodes []int
	UnavailableCooldown    time.Duration
	MaxRemaps              int
}

// EncoderAffinity is persisted with the ARC episode transaction. Owner is the
// currently sticky replica and Visited contains every replica that may still
// retain state for eventual close fanout.
type EncoderAffinity struct {
	Owner   string
	Visited []string
}

// EncoderCloseReport contains bounded aggregate close outcomes only. Replica
// identities and endpoint details never cross this interface.
type EncoderCloseReport struct {
	Attempted   int
	Closed      int
	Unavailable int
	Failed      int
}

type EncoderPool struct {
	replicas       []EncoderReplica
	byID           map[string]int
	config         EncoderPoolConfig
	unavailableMu  sync.Mutex
	unavailableTil map[string]time.Time
	now            func() time.Time
}

func NewEncoderPool(
	replicas []EncoderReplica,
	config EncoderPoolConfig,
) (*EncoderPool, error) {
	if err := validateEncoderPoolConfig(replicas, config); err != nil {
		return nil, err
	}
	cloned := append([]EncoderReplica(nil), replicas...)
	byID := make(map[string]int, len(cloned))
	for index := range cloned {
		byID[cloned[index].ID] = index
	}
	config.UnavailableStatusCodes = append(
		[]int(nil),
		config.UnavailableStatusCodes...,
	)
	return &EncoderPool{
		replicas:       cloned,
		byID:           byID,
		config:         config,
		unavailableTil: make(map[string]time.Time),
		now:            time.Now,
	}, nil
}

func validateEncoderPoolConfig(
	replicas []EncoderReplica,
	config EncoderPoolConfig,
) error {
	if err := validateEncoderPoolContract(replicas, config); err != nil {
		return err
	}
	if err := validateEncoderUnavailableStatuses(
		config.UnavailableStatusCodes,
	); err != nil {
		return err
	}
	return validateEncoderReplicas(replicas)
}

func validateEncoderPoolContract(
	replicas []EncoderReplica,
	config EncoderPoolConfig,
) error {
	if config.SchemaVersion != EncoderFailoverSchemaV1 {
		return errors.New("ARC encoder replica contract version is unsupported")
	}
	if len(replicas) < 2 || len(replicas) > maxEncoderReplicas {
		return errors.New("ARC encoder replica count must be between 2 and 8")
	}
	if config.UnavailableCooldown <= 0 {
		return errors.New("ARC encoder unavailable cooldown must be positive")
	}
	if config.MaxRemaps != 1 {
		return errors.New("ARC encoder replica contract requires exactly one remap")
	}
	if len(config.UnavailableStatusCodes) == 0 {
		return errors.New("ARC encoder unavailable status set is required")
	}
	return nil
}

func validateEncoderUnavailableStatuses(statuses []int) error {
	seenStatus := make(map[int]struct{}, len(statuses))
	for _, status := range statuses {
		if status < http.StatusBadRequest || status > 599 {
			return errors.New("ARC encoder unavailable status is invalid")
		}
		if _, exists := seenStatus[status]; exists {
			return errors.New("ARC encoder unavailable status is duplicated")
		}
		seenStatus[status] = struct{}{}
	}
	return nil
}

func validateEncoderReplicas(replicas []EncoderReplica) error {
	seenIDs := make(map[string]struct{}, len(replicas))
	seenClients := make(map[*EncoderClient]struct{}, len(replicas))
	active := 0
	for index := range replicas {
		isActive, err := validateEncoderReplica(
			&replicas[index],
			seenIDs,
			seenClients,
		)
		if err != nil {
			return err
		}
		if isActive {
			active++
		}
	}
	if active == 0 {
		return errors.New("ARC encoder replica set requires an active member")
	}
	return nil
}

func validateEncoderReplica(
	replica *EncoderReplica,
	seenIDs map[string]struct{},
	seenClients map[*EncoderClient]struct{},
) (bool, error) {
	if !validEncoderReplicaID(replica.ID) {
		return false, errors.New("ARC encoder replica ID is invalid")
	}
	if _, exists := seenIDs[replica.ID]; exists {
		return false, errors.New("ARC encoder replica ID is duplicated")
	}
	seenIDs[replica.ID] = struct{}{}
	if replica.Client == nil || !replica.Client.config.RetainedSession {
		return false, errors.New("ARC encoder replicas require retained clients")
	}
	if replica.Client.config.MaxRetries != 0 {
		return false, errors.New(
			"ARC encoder replica clients cannot retry internally",
		)
	}
	if _, exists := seenClients[replica.Client]; exists {
		return false, errors.New("ARC encoder replica client is duplicated")
	}
	seenClients[replica.Client] = struct{}{}
	switch replica.State {
	case EncoderReplicaActive:
		return true, nil
	case EncoderReplicaDraining:
		return false, nil
	default:
		return false, errors.New("ARC encoder replica state is invalid")
	}
}

func validEncoderReplicaID(value string) bool {
	return len(value) > 0 &&
		len(value) <= maxEncoderReplicaIDBytes &&
		encoderReplicaIDPattern.MatchString(value)
}

// Encode preserves the single-client interface for callers that have no
// persisted affinity yet. Production callers should use EncodeWithAffinity so
// ownership survives router replicas and process restarts.
func (pool *EncoderPool) Encode(
	ctx context.Context,
	episodeIDHash string,
	turns []Turn,
) (*EncoderResult, error) {
	return pool.EncodeWithAffinity(ctx, episodeIDHash, turns, EncoderAffinity{})
}

func (pool *EncoderPool) EncodeWithAffinity(
	ctx context.Context,
	episodeIDHash string,
	turns []Turn,
	affinity EncoderAffinity,
) (*EncoderResult, error) {
	if pool == nil {
		return nil, encoderFailure(EncoderFailureRequest, "replica_pool")
	}
	if !validEpisodeIDHash(episodeIDHash) {
		return nil, encoderFailure(EncoderFailureRequest, "episode_hash")
	}
	visited, err := normalizeEncoderAffinity(affinity)
	if err != nil {
		return nil, err
	}
	primary, err := pool.primaryReplica(episodeIDHash, affinity.Owner, nil)
	if err != nil {
		return nil, err
	}
	attempted := []string{primary.ID}
	remapped := affinity.Owner != "" && primary.ID != affinity.Owner
	visited = appendUniqueReplicaID(visited, primary.ID)
	result, encodeErr := primary.Client.Encode(ctx, episodeIDHash, turns)
	if encodeErr == nil {
		return annotateEncoderResult(
			pool,
			result,
			primary.ID,
			attempted,
			visited,
			remapped,
		), nil
	}
	if !pool.remappableFailure(primary.ID, encodeErr) {
		return nil, encodeErr
	}
	secondary, err := pool.primaryReplica(episodeIDHash, "", attempted)
	if err != nil {
		return nil, err
	}
	attempted = append(attempted, secondary.ID)
	visited = appendUniqueReplicaID(visited, secondary.ID)
	result, encodeErr = secondary.Client.Encode(ctx, episodeIDHash, turns)
	if encodeErr != nil {
		if pool.explicitlyUnavailable(encodeErr) {
			pool.markUnavailable(secondary.ID)
		}
		return nil, encodeErr
	}
	return annotateEncoderResult(
		pool,
		result,
		secondary.ID,
		attempted,
		visited,
		true,
	), nil
}

func normalizeEncoderAffinity(affinity EncoderAffinity) ([]string, error) {
	if affinity.Owner != "" && !validEncoderReplicaID(affinity.Owner) {
		return nil, encoderFailure(EncoderFailureContract, "replica_owner")
	}
	if len(affinity.Visited) > maxEncoderReplicas {
		return nil, encoderFailure(EncoderFailureContract, "replica_visited")
	}
	visited := make([]string, 0, len(affinity.Visited)+1)
	for _, replicaID := range affinity.Visited {
		if !validEncoderReplicaID(replicaID) || slices.Contains(visited, replicaID) {
			return nil, encoderFailure(EncoderFailureContract, "replica_visited")
		}
		visited = append(visited, replicaID)
	}
	if affinity.Owner != "" {
		visited = appendUniqueReplicaID(visited, affinity.Owner)
	}
	return visited, nil
}

func (pool *EncoderPool) primaryReplica(
	episodeIDHash string,
	owner string,
	excluded []string,
) (*EncoderReplica, error) {
	now := pool.now().UTC()
	if owner != "" && !slices.Contains(excluded, owner) {
		index, exists := pool.byID[owner]
		if !exists {
			return nil, encoderFailure(EncoderFailureContract, "replica_owner_missing")
		}
		if !pool.unavailable(owner, now) {
			return &pool.replicas[index], nil
		}
	}
	selected := -1
	var selectedScore [sha256.Size]byte
	for index := range pool.replicas {
		replica := &pool.replicas[index]
		if replica.State != EncoderReplicaActive ||
			slices.Contains(excluded, replica.ID) ||
			pool.unavailable(replica.ID, now) {
			continue
		}
		score := encoderReplicaScore(episodeIDHash, replica.ID)
		if selected < 0 || bytes.Compare(score[:], selectedScore[:]) > 0 {
			selected = index
			selectedScore = score
		}
	}
	if selected < 0 {
		return nil, encoderFailure(EncoderFailureStatus, "replica_set_unavailable")
	}
	return &pool.replicas[selected], nil
}

func encoderReplicaScore(episodeIDHash, replicaID string) [sha256.Size]byte {
	return sha256.Sum256([]byte(episodeIDHash + "\x00" + replicaID))
}

func (pool *EncoderPool) remappableFailure(replicaID string, err error) bool {
	if pool.config.MaxRemaps != 1 || !pool.explicitlyUnavailable(err) {
		return false
	}
	pool.markUnavailable(replicaID)
	return true
}

func (pool *EncoderPool) explicitlyUnavailable(err error) bool {
	var failure *EncoderFailure
	return errors.As(err, &failure) &&
		failure.Class == EncoderFailureStatus &&
		failure.StatusCode != 0 &&
		slices.Contains(pool.config.UnavailableStatusCodes, failure.StatusCode)
}

func (pool *EncoderPool) markUnavailable(replicaID string) {
	pool.unavailableMu.Lock()
	defer pool.unavailableMu.Unlock()
	pool.unavailableTil[replicaID] = pool.now().UTC().Add(
		pool.config.UnavailableCooldown,
	)
}

func (pool *EncoderPool) unavailable(replicaID string, now time.Time) bool {
	pool.unavailableMu.Lock()
	defer pool.unavailableMu.Unlock()
	until, exists := pool.unavailableTil[replicaID]
	if !exists {
		return false
	}
	if !now.Before(until) {
		delete(pool.unavailableTil, replicaID)
		return false
	}
	return true
}

func annotateEncoderResult(
	pool *EncoderPool,
	result *EncoderResult,
	owner string,
	attempted []string,
	visited []string,
	remapped bool,
) *EncoderResult {
	if result == nil {
		return nil
	}
	result.ReplicaID = owner
	result.ReplicaIndex = pool.byID[owner]
	result.ReplicaAttempts = len(attempted)
	result.ReplicaFailover = remapped || len(attempted) > 1
	result.VisitedReplicaIDs = append([]string(nil), visited...)
	return result
}

func appendUniqueReplicaID(values []string, value string) []string {
	if value == "" || slices.Contains(values, value) {
		return values
	}
	return append(values, value)
}

// Probe checks every configured replica, including draining members because
// they can still own live sessions. It does not stop after the first failure.
func (pool *EncoderPool) Probe(ctx context.Context, correlation string) error {
	if pool == nil {
		return encoderFailure(EncoderFailureRequest, "replica_pool")
	}
	var failures []error
	for index := range pool.replicas {
		if err := pool.replicas[index].Client.Probe(ctx, correlation); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

// CloseSession fans one close request out to every configured replica in the
// persisted visited set. An explicit configured unavailability response is a
// successful cleanup outcome: that replica cannot retain a resident session.
func (pool *EncoderPool) CloseSession(
	ctx context.Context,
	episodeIDHash string,
	visited []string,
) (EncoderCloseReport, error) {
	if pool == nil {
		return EncoderCloseReport{}, encoderFailure(
			EncoderFailureRequest,
			"replica_pool",
		)
	}
	normalized, err := normalizeEncoderAffinity(EncoderAffinity{Visited: visited})
	if err != nil {
		return EncoderCloseReport{}, err
	}
	report := EncoderCloseReport{}
	var failures []error
	type closeOutcome struct {
		replicaID string
		err       error
	}
	outcomes := make(chan closeOutcome, len(normalized))
	launched := 0
	for _, replicaID := range normalized {
		index, exists := pool.byID[replicaID]
		if !exists {
			report.Failed++
			failures = append(failures, encoderFailure(
				EncoderFailureContract,
				"close_replica_missing",
			))
			continue
		}
		launched++
		client := pool.replicas[index].Client
		go func() {
			outcomes <- closeOutcome{
				replicaID: replicaID,
				err:       client.CloseSession(ctx, episodeIDHash),
			}
		}()
	}
	for range launched {
		outcome := <-outcomes
		report.Attempted++
		switch {
		case outcome.err == nil:
			report.Closed++
		case pool.explicitlyUnavailable(outcome.err):
			pool.markUnavailable(outcome.replicaID)
			report.Unavailable++
		default:
			report.Failed++
			failures = append(failures, outcome.err)
		}
	}
	return report, errors.Join(failures...)
}

func (pool *EncoderPool) Close() {
	if pool == nil {
		return
	}
	for index := range pool.replicas {
		pool.replicas[index].Client.Close()
	}
}
