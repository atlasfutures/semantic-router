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
	"slices"
	"sync"
	"time"
)

// PolicyFailoverSchemaV1 identifies the first policy-service replica and
// failover contract. Configuration must opt into this exact value.
const PolicyFailoverSchemaV1 = "rayline.arc.policy-failover.v1"

// PolicyFailoverStatuses are the only HTTP statuses a policy replica may be
// declared unavailable by. Each one is answered before the service touches
// the episode: 404 (the replica has not loaded the package), 429 (the
// replica is at its session cap) and 503 (no container or backend). A 409
// (session_busy) means the episode is in flight on that replica and must
// never move; a 5xx gateway error or a timeout may follow a decide that
// already advanced the replica's session, so both fail closed.
var PolicyFailoverStatuses = []int{http.StatusNotFound, http.StatusTooManyRequests, http.StatusServiceUnavailable}

// PolicyServiceReplica is one policy-service deployment with a stable,
// deployment-owned identity. A draining replica keeps serving the episodes
// that already own it and receives no new ones.
type PolicyServiceReplica struct {
	ID     string
	State  EncoderReplicaState
	Client *PolicyServiceClient
}

// PolicyServicePoolConfig is the versioned failover contract.
type PolicyServicePoolConfig struct {
	SchemaVersion          string
	UnavailableStatusCodes []int
	UnavailableCooldown    time.Duration
	MaxRemaps              int
}

// PolicyServicePool routes each episode to one policy replica: the owner the
// episode state persisted, else the rendezvous (highest random weight) choice
// among active replicas. A policy replica keeps the episode's retained
// encoder session and its strict episode memory in its own process, so a
// turn that reaches another replica is correct but rebuilds its session.
// The owner therefore moves only when it is gone from the configuration,
// cooling down, or answers one of the configured unavailable statuses.
type PolicyServicePool struct {
	replicas       []PolicyServiceReplica
	byID           map[string]int
	config         PolicyServicePoolConfig
	unavailableMu  sync.Mutex
	unavailableTil map[string]time.Time
	// unready holds the replicas whose last readiness check failed. They
	// take no episode, owned or new, until a check passes (Reprobe).
	unready map[string]struct{}
	now     func() time.Time
}

// NewPolicyServicePool validates the replicas and the failover contract.
func NewPolicyServicePool(replicas []PolicyServiceReplica, config PolicyServicePoolConfig) (*PolicyServicePool, error) {
	if err := validatePolicyServicePool(replicas, config); err != nil {
		return nil, err
	}
	cloned := append([]PolicyServiceReplica(nil), replicas...)
	byID := make(map[string]int, len(cloned))
	for index := range cloned {
		byID[cloned[index].ID] = index
	}
	config.UnavailableStatusCodes = append([]int(nil), config.UnavailableStatusCodes...)
	return &PolicyServicePool{
		replicas:       cloned,
		byID:           byID,
		config:         config,
		unavailableTil: make(map[string]time.Time),
		unready:        make(map[string]struct{}),
		now:            time.Now,
	}, nil
}

func validatePolicyServicePool(replicas []PolicyServiceReplica, config PolicyServicePoolConfig) error {
	if config.SchemaVersion != PolicyFailoverSchemaV1 {
		return errors.New("ARC policy replica contract version is unsupported")
	}
	if len(replicas) < 1 || len(replicas) > maxEncoderReplicas {
		return errors.New("ARC policy replica count must be between 1 and 8")
	}
	if config.UnavailableCooldown <= 0 {
		return errors.New("ARC policy unavailable cooldown must be positive")
	}
	if config.MaxRemaps != 1 {
		return errors.New("ARC policy replica contract requires exactly one remap")
	}
	if len(config.UnavailableStatusCodes) == 0 {
		return errors.New("ARC policy unavailable status set is required")
	}
	seenStatus := make(map[int]struct{}, len(config.UnavailableStatusCodes))
	for _, status := range config.UnavailableStatusCodes {
		if !slices.Contains(PolicyFailoverStatuses, status) {
			return errors.New("ARC policy unavailable status is not remappable")
		}
		if _, exists := seenStatus[status]; exists {
			return errors.New("ARC policy unavailable status is duplicated")
		}
		seenStatus[status] = struct{}{}
	}
	seenIDs := make(map[string]struct{}, len(replicas))
	seenClients := make(map[*PolicyServiceClient]struct{}, len(replicas))
	active := 0
	for _, replica := range replicas {
		if !validEncoderReplicaID(replica.ID) {
			return errors.New("ARC policy replica ID is invalid")
		}
		if _, exists := seenIDs[replica.ID]; exists {
			return errors.New("ARC policy replica ID is duplicated")
		}
		seenIDs[replica.ID] = struct{}{}
		if replica.Client == nil {
			return errors.New("ARC policy replica client is required")
		}
		if _, exists := seenClients[replica.Client]; exists {
			return errors.New("ARC policy replica client is duplicated")
		}
		seenClients[replica.Client] = struct{}{}
		switch replica.State {
		case EncoderReplicaActive:
			active++
		case EncoderReplicaDraining:
		default:
			return errors.New("ARC policy replica state is invalid")
		}
	}
	if active == 0 {
		return errors.New("ARC policy replica set requires an active member")
	}
	return nil
}

// Probe runs check against every replica. A replica that fails it is
// unready: it takes no episode, owned or new, until a later check passes
// (Reprobe). The pool is ready when at least one active replica passes;
// otherwise Probe returns the first failure.
func (pool *PolicyServicePool) Probe(
	ctx context.Context,
	check func(context.Context, *PolicyServiceClient) error,
) error {
	var first error
	ready := false
	for index := range pool.replicas {
		replica := &pool.replicas[index]
		if err := check(ctx, replica.Client); err != nil {
			pool.setReady(replica.ID, false)
			if first == nil {
				first = err
			}
			continue
		}
		pool.setReady(replica.ID, true)
		if replica.State == EncoderReplicaActive {
			ready = true
		}
	}
	if ready {
		return nil
	}
	if first == nil {
		first = &PolicyServiceError{Class: "replica_set_unavailable"}
	}
	return first
}

// Reprobe checks the unready replicas every interval until ctx ends, and
// returns each to service once its check passes.
func (pool *PolicyServicePool) Reprobe(
	ctx context.Context,
	check func(context.Context, *PolicyServiceClient) error,
	interval time.Duration,
) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		for index := range pool.replicas {
			replica := &pool.replicas[index]
			if pool.isReady(replica.ID) {
				continue
			}
			if check(ctx, replica.Client) == nil {
				pool.setReady(replica.ID, true)
			}
		}
	}
}

func (pool *PolicyServicePool) setReady(replicaID string, ready bool) {
	pool.unavailableMu.Lock()
	defer pool.unavailableMu.Unlock()
	if ready {
		delete(pool.unready, replicaID)
		return
	}
	pool.unready[replicaID] = struct{}{}
}

func (pool *PolicyServicePool) isReady(replicaID string) bool {
	pool.unavailableMu.Lock()
	defer pool.unavailableMu.Unlock()
	_, unready := pool.unready[replicaID]
	return !unready
}

// Bind starts one turn's routing for an episode. affinity is the owner and
// visited set the episode state persisted (both empty for a new episode).
func (pool *PolicyServicePool) Bind(episodeIDHash string, affinity EncoderAffinity) *PolicyServiceBinding {
	return &PolicyServiceBinding{
		pool:      pool,
		episode:   episodeIDHash,
		persisted: affinity.Owner,
		visited:   pool.configuredVisited(affinity),
	}
}

// configuredVisited keeps only the visited replicas still configured. The
// policy-service mode has no session close fan-out that would read a removed
// member, and pruning keeps the persisted set inside its bound as replicas
// are rotated.
func (pool *PolicyServicePool) configuredVisited(affinity EncoderAffinity) []string {
	visited := make([]string, 0, len(affinity.Visited)+1)
	for _, id := range affinity.Visited {
		if _, configured := pool.byID[id]; configured {
			visited = appendUniqueReplicaID(visited, id)
		}
	}
	return visited
}

// PolicyServiceBinding routes every call of one turn: the first call places
// the episode, and later calls (session_busy retries, re-asks) stay on the
// replica that call settled on, so a strict episode is never in flight on
// two replicas within a turn. It is not safe for concurrent use.
type PolicyServiceBinding struct {
	pool      *PolicyServicePool
	episode   string
	persisted string
	current   *PolicyServiceReplica
	attempted []string
	visited   []string
	remapped  bool
}

// Decide asks the bound replica for one decision. A configured unavailable
// status cools that replica down and, once per turn, moves the episode to
// the next rendezvous choice. Every other failure, a timeout or transport
// error included, is returned as is: the replica may already have advanced
// its session, so the episode does not move.
func (binding *PolicyServiceBinding) Decide(
	ctx context.Context,
	request PolicyDecisionRequest,
) (*PolicyDecisionResponse, error) {
	pool := binding.pool
	if binding.current == nil {
		replica, err := pool.primaryReplica(binding.episode, binding.persisted, nil)
		if err != nil {
			return nil, err
		}
		binding.place(replica)
		binding.remapped = binding.persisted != "" && replica.ID != binding.persisted
	}
	response, err := binding.current.Client.Decide(ctx, request)
	if err == nil || !pool.explicitlyUnavailable(err) {
		return response, err
	}
	pool.markUnavailable(binding.current.ID)
	if len(binding.attempted) > pool.config.MaxRemaps {
		return nil, err
	}
	secondary, placeErr := pool.primaryReplica(binding.episode, "", binding.attempted)
	if placeErr != nil {
		return nil, err
	}
	binding.place(secondary)
	binding.remapped = true
	response, err = secondary.Client.Decide(ctx, request)
	if err != nil && pool.explicitlyUnavailable(err) {
		pool.markUnavailable(secondary.ID)
	}
	return response, err
}

func (binding *PolicyServiceBinding) place(replica *PolicyServiceReplica) {
	binding.current = replica
	binding.attempted = append(binding.attempted, replica.ID)
	binding.visited = appendUniqueReplicaID(binding.visited, replica.ID)
}

// Annotate records the turn's owner on result, which the episode commit
// persists. It leaves result unchanged when no call was routed.
func (binding *PolicyServiceBinding) Annotate(result *EncoderResult) {
	if binding == nil || binding.current == nil || result == nil {
		return
	}
	result.ReplicaID = binding.current.ID
	result.ReplicaIndex = binding.pool.byID[binding.current.ID]
	result.ReplicaAttempts = len(binding.attempted)
	result.ReplicaFailover = binding.remapped || len(binding.attempted) > 1
	result.VisitedReplicaIDs = append([]string(nil), binding.visited...)
}

func (pool *PolicyServicePool) primaryReplica(
	episodeIDHash string,
	owner string,
	excluded []string,
) (*PolicyServiceReplica, error) {
	now := pool.now().UTC()
	// A persisted owner that is no longer configured, unready or cooling
	// down is placed afresh among the active replicas and the turn records a
	// failover. Removing a replica from the list is how a deployment scales
	// down; its episodes rebuild elsewhere.
	if owner != "" && !slices.Contains(excluded, owner) {
		if index, exists := pool.byID[owner]; exists && !pool.unavailable(owner, now) {
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
		return nil, &PolicyServiceError{Class: "replica_set_unavailable", Status: http.StatusServiceUnavailable}
	}
	return &pool.replicas[selected], nil
}

func (pool *PolicyServicePool) explicitlyUnavailable(err error) bool {
	var failure *PolicyServiceError
	return errors.As(err, &failure) &&
		failure.Status != 0 &&
		failure.Class != "timeout" && failure.Class != "transport" &&
		slices.Contains(pool.config.UnavailableStatusCodes, failure.Status)
}

func (pool *PolicyServicePool) markUnavailable(replicaID string) {
	pool.unavailableMu.Lock()
	defer pool.unavailableMu.Unlock()
	pool.unavailableTil[replicaID] = pool.now().UTC().Add(pool.config.UnavailableCooldown)
}

func (pool *PolicyServicePool) unavailable(replicaID string, now time.Time) bool {
	pool.unavailableMu.Lock()
	defer pool.unavailableMu.Unlock()
	if _, unready := pool.unready[replicaID]; unready {
		return true
	}
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
