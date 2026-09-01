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
	"errors"
	"sync"
	"time"
)

type EncoderMembershipClientFactory func(
	EncoderMembershipReplica,
) (*EncoderClient, error)

type DynamicEncoderPoolConfig struct {
	Source          EncoderMembershipReader
	ClientFactory   EncoderMembershipClientFactory
	PoolConfig      EncoderPoolConfig
	RefreshInterval time.Duration
}

// DynamicEncoderPool keeps request handling on immutable EncoderPool
// snapshots. Membership refresh cannot interrupt a request and old clients
// remain available for any late close fanout until router shutdown. This is a
// deliberate availability/correctness trade-off: endpoint connections are
// retained longer, while a premature membership removal still fails closed.
type DynamicEncoderPool struct {
	source        EncoderMembershipReader
	clientFactory EncoderMembershipClientFactory
	poolConfig    EncoderPoolConfig
	refresh       time.Duration

	refreshMu sync.Mutex
	mu        sync.RWMutex
	current   *dynamicEncoderPoolSnapshot
	history   []*dynamicEncoderPoolSnapshot
	clients   map[string]*EncoderClient
	endpoints map[string]string
	lastError error
	closed    bool

	done      chan struct{}
	closeOnce sync.Once
}

type dynamicEncoderPoolSnapshot struct {
	membership EncoderMembershipSnapshot
	pool       *EncoderPool
}

func NewDynamicEncoderPool(
	ctx context.Context,
	config DynamicEncoderPoolConfig,
) (*DynamicEncoderPool, error) {
	if config.Source == nil || config.ClientFactory == nil ||
		config.RefreshInterval <= 0 {
		return nil, errors.New("ARC dynamic encoder pool configuration is incomplete")
	}
	pool := &DynamicEncoderPool{
		source:        config.Source,
		clientFactory: config.ClientFactory,
		poolConfig:    config.PoolConfig,
		refresh:       config.RefreshInterval,
		clients:       make(map[string]*EncoderClient),
		endpoints:     make(map[string]string),
		done:          make(chan struct{}),
	}
	if err := pool.Refresh(ctx); err != nil {
		return nil, err
	}
	go pool.runRefreshLoop()
	return pool, nil
}

func (pool *DynamicEncoderPool) runRefreshLoop() {
	ticker := time.NewTicker(pool.refresh)
	defer ticker.Stop()
	for {
		select {
		case <-pool.done:
			return
		case <-ticker.C:
			refreshContext, cancel := context.WithTimeout(
				context.Background(),
				pool.refresh,
			)
			_ = pool.Refresh(refreshContext)
			cancel()
		}
	}
}

// Refresh adopts one newer reviewed snapshot. Errors leave the current
// snapshot intact, so a source outage cannot turn an already-ready router into
// an unbounded discovery client or silently change request semantics.
func (pool *DynamicEncoderPool) Refresh(ctx context.Context) error {
	if pool == nil {
		return errors.New("ARC dynamic encoder pool is nil")
	}
	pool.refreshMu.Lock()
	defer pool.refreshMu.Unlock()
	if pool.isClosed() {
		return errors.New("ARC dynamic encoder pool is closed")
	}
	membership, err := pool.source.Load(ctx)
	if err != nil {
		pool.recordRefreshError(err)
		return err
	}
	current := pool.currentSnapshot()
	if refreshErr := validateDynamicMembershipRefresh(current, membership); refreshErr != nil {
		pool.recordRefreshError(refreshErr)
		return refreshErr
	}
	if current != nil && membership.Revision == current.membership.Revision {
		pool.recordRefreshError(nil)
		return nil
	}
	snapshot, err := pool.buildSnapshot(ctx, membership)
	if err != nil {
		pool.recordRefreshError(err)
		return err
	}
	return pool.installSnapshot(snapshot)
}

func validateDynamicMembershipRefresh(
	current *dynamicEncoderPoolSnapshot,
	membership EncoderMembershipSnapshot,
) error {
	if err := validateEncoderMembershipSnapshot(membership); err != nil {
		return err
	}
	if current == nil {
		return nil
	}
	if membership.Revision < current.membership.Revision {
		return errors.New("ARC encoder membership revision moved backwards")
	}
	if membership.Revision == current.membership.Revision {
		if !sameEncoderMembership(current.membership, membership) {
			return errors.New("ARC encoder membership revision was rewritten")
		}
		return nil
	}
	if membership.Revision != current.membership.Revision+1 {
		return errors.New("ARC encoder membership revision skipped")
	}
	return checkedMembershipSuccessor(current.membership, membership)
}

func (pool *DynamicEncoderPool) installSnapshot(
	snapshot *dynamicEncoderPoolSnapshot,
) error {
	pool.mu.Lock()
	defer pool.mu.Unlock()
	if pool.closed {
		return errors.New("ARC dynamic encoder pool is closed")
	}
	if pool.current != nil {
		pool.history = append(pool.history, pool.current)
	}
	pool.current = snapshot
	pool.lastError = nil
	return nil
}

func (pool *DynamicEncoderPool) buildSnapshot(
	ctx context.Context,
	membership EncoderMembershipSnapshot,
) (*dynamicEncoderPoolSnapshot, error) {
	replicas := make([]EncoderReplica, 0, len(membership.Replicas))
	created := make([]*EncoderClient, 0, len(membership.Replicas))
	probeNewCapacity := pool.currentSnapshot() != nil
	for _, member := range membership.Replicas {
		client, clientCreated, err := pool.clientForMembershipMember(
			ctx,
			member,
			probeNewCapacity,
		)
		if err != nil {
			closeEncoderClients(created)
			return nil, err
		}
		if clientCreated {
			created = append(created, client)
		}
		replicas = append(replicas, EncoderReplica{
			ID:     member.ID,
			State:  member.State,
			Client: client,
		})
	}
	encoderPool, err := NewEncoderPool(replicas, pool.poolConfig)
	if err != nil {
		closeEncoderClients(created)
		return nil, err
	}
	pool.mu.Lock()
	for index, member := range membership.Replicas {
		if _, exists := pool.clients[member.ID]; !exists {
			pool.clients[member.ID] = replicas[index].Client
			pool.endpoints[member.ID] = normalizeEncoderMembershipEndpoint(
				member.BaseURL,
			)
		}
	}
	pool.mu.Unlock()
	return &dynamicEncoderPoolSnapshot{
		membership: cloneEncoderMembershipSnapshot(membership),
		pool:       encoderPool,
	}, nil
}

func (pool *DynamicEncoderPool) clientForMembershipMember(
	ctx context.Context,
	member EncoderMembershipReplica,
	probeNewCapacity bool,
) (*EncoderClient, bool, error) {
	endpoint := normalizeEncoderMembershipEndpoint(member.BaseURL)
	pool.mu.RLock()
	existing, known := pool.clients[member.ID]
	knownEndpoint := pool.endpoints[member.ID]
	pool.mu.RUnlock()
	if known && knownEndpoint != endpoint {
		return nil, false, errors.New("ARC encoder membership changed a stable replica endpoint")
	}
	if existing != nil {
		return existing, false, nil
	}
	client, err := pool.clientFactory(member)
	if err != nil {
		return nil, false, err
	}
	if probeNewCapacity {
		if err := client.Probe(
			ctx,
			"dynamic-membership-register:"+member.ID,
		); err != nil {
			client.Close()
			return nil, false, err
		}
	}
	return client, true, nil
}

func closeEncoderClients(clients []*EncoderClient) {
	for _, client := range clients {
		client.Close()
	}
}

func (pool *DynamicEncoderPool) Encode(
	ctx context.Context,
	episodeIDHash string,
	turns []Turn,
) (*EncoderResult, error) {
	return pool.EncodeWithAffinity(ctx, episodeIDHash, turns, EncoderAffinity{})
}

func (pool *DynamicEncoderPool) EncodeWithAffinity(
	ctx context.Context,
	episodeIDHash string,
	turns []Turn,
	affinity EncoderAffinity,
) (*EncoderResult, error) {
	snapshot := pool.currentSnapshot()
	if snapshot == nil {
		return nil, encoderFailure(EncoderFailureRequest, "dynamic_replica_pool")
	}
	return snapshot.pool.EncodeWithAffinity(ctx, episodeIDHash, turns, affinity)
}

func (pool *DynamicEncoderPool) Probe(
	ctx context.Context,
	correlation string,
) error {
	snapshot := pool.currentSnapshot()
	if snapshot == nil {
		return encoderFailure(EncoderFailureRequest, "dynamic_replica_pool")
	}
	return snapshot.pool.Probe(ctx, correlation)
}

func (pool *DynamicEncoderPool) CloseSession(
	ctx context.Context,
	episodeIDHash string,
	visited []string,
) (EncoderCloseReport, error) {
	normalized, err := normalizeEncoderAffinity(EncoderAffinity{Visited: visited})
	if err != nil {
		return EncoderCloseReport{}, err
	}
	report := EncoderCloseReport{}
	var failures []error
	for _, replicaID := range normalized {
		ownerPool := pool.poolForReplica(replicaID)
		if ownerPool == nil {
			report.Failed++
			failures = append(failures, encoderFailure(
				EncoderFailureContract,
				"close_replica_missing",
			))
			continue
		}
		partial, closeErr := ownerPool.CloseSession(
			ctx,
			episodeIDHash,
			[]string{replicaID},
		)
		report.Attempted += partial.Attempted
		report.Closed += partial.Closed
		report.Unavailable += partial.Unavailable
		report.Failed += partial.Failed
		if closeErr != nil {
			failures = append(failures, closeErr)
		}
	}
	return report, errors.Join(failures...)
}

func (pool *DynamicEncoderPool) currentSnapshot() *dynamicEncoderPoolSnapshot {
	if pool == nil {
		return nil
	}
	pool.mu.RLock()
	defer pool.mu.RUnlock()
	return pool.current
}

func (pool *DynamicEncoderPool) poolForReplica(replicaID string) *EncoderPool {
	if pool == nil {
		return nil
	}
	pool.mu.RLock()
	defer pool.mu.RUnlock()
	if pool.current != nil {
		if _, exists := pool.current.pool.byID[replicaID]; exists {
			return pool.current.pool
		}
	}
	for index := len(pool.history) - 1; index >= 0; index-- {
		if _, exists := pool.history[index].pool.byID[replicaID]; exists {
			return pool.history[index].pool
		}
	}
	return nil
}

func (pool *DynamicEncoderPool) Membership() (EncoderMembershipSnapshot, error) {
	if pool == nil {
		return EncoderMembershipSnapshot{}, errors.New("ARC dynamic encoder pool is nil")
	}
	pool.mu.RLock()
	defer pool.mu.RUnlock()
	if pool.current == nil {
		return EncoderMembershipSnapshot{}, errors.New("ARC dynamic encoder pool is not ready")
	}
	return cloneEncoderMembershipSnapshot(pool.current.membership), pool.lastError
}

func (pool *DynamicEncoderPool) Close() {
	if pool == nil {
		return
	}
	pool.closeOnce.Do(func() {
		close(pool.done)
		pool.refreshMu.Lock()
		pool.mu.Lock()
		pool.closed = true
		for _, client := range pool.clients {
			client.Close()
		}
		pool.clients = nil
		pool.current = nil
		pool.history = nil
		closer, ok := pool.source.(interface{ Close() error })
		pool.mu.Unlock()
		pool.refreshMu.Unlock()
		if ok {
			_ = closer.Close()
		}
	})
}

func (pool *DynamicEncoderPool) isClosed() bool {
	pool.mu.RLock()
	defer pool.mu.RUnlock()
	return pool.closed
}

func (pool *DynamicEncoderPool) recordRefreshError(err error) {
	pool.mu.Lock()
	defer pool.mu.Unlock()
	pool.lastError = err
}
