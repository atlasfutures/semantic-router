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
	"strings"
	"time"
)

// EncoderMembershipController owns the only automatic removal path. The
// controller first publishes draining, then waits a complete episode idle
// window and verifies that Redis has no owner or visited-owner reference before
// a revisioned compare-and-set removes the member.
type EncoderMembershipController struct {
	source     EncoderMembershipSource
	references EncoderMembershipReferenceCounter
	idleTTL    time.Duration
	now        func() time.Time
}

type EncoderMembershipControllerConfig struct {
	Source     EncoderMembershipSource
	References EncoderMembershipReferenceCounter
	IdleTTL    time.Duration
	Now        func() time.Time
}

// EncoderMembershipDrainReport contains only aggregate control-plane state.
// It never contains prompt, endpoint, or episode data.
type EncoderMembershipDrainReport struct {
	Revision        uint64
	Draining        int
	Eligible        int
	Referenced      int
	Removed         int
	MinimumRetained int
}

func NewEncoderMembershipController(
	config EncoderMembershipControllerConfig,
) (*EncoderMembershipController, error) {
	if config.Source == nil || config.References == nil || config.IdleTTL <= 0 {
		return nil, errors.New("ARC encoder membership controller configuration is incomplete")
	}
	now := config.Now
	if now == nil {
		now = time.Now
	}
	return &EncoderMembershipController{
		source:     config.Source,
		references: config.References,
		idleTTL:    config.IdleTTL,
		now:        now,
	}, nil
}

// RegisterActive adds reviewed capacity with a new stable identity. Repeating
// the exact registration is idempotent; changing an existing identity's
// endpoint or reactivating a draining identity is rejected by design.
func (controller *EncoderMembershipController) RegisterActive(
	ctx context.Context,
	replicaID string,
	baseURL string,
) (EncoderMembershipSnapshot, error) {
	if controller == nil || !validEncoderReplicaID(replicaID) ||
		strings.TrimSpace(baseURL) == "" {
		return EncoderMembershipSnapshot{}, errors.New("ARC encoder registration request is invalid")
	}
	current, err := controller.source.Load(ctx)
	if err != nil {
		return EncoderMembershipSnapshot{}, err
	}
	if index := membershipReplicaIndex(current, replicaID); index >= 0 {
		existing := current.Replicas[index]
		if existing.State == EncoderReplicaActive &&
			normalizeEncoderMembershipEndpoint(existing.BaseURL) ==
				normalizeEncoderMembershipEndpoint(baseURL) {
			return current, nil
		}
		return EncoderMembershipSnapshot{}, errors.New("ARC encoder registration identity already exists")
	}
	next := cloneEncoderMembershipSnapshot(current)
	next.Revision++
	next.Replicas = append(next.Replicas, EncoderMembershipReplica{
		ID:      replicaID,
		BaseURL: strings.TrimSpace(baseURL),
		State:   EncoderReplicaActive,
	})
	if err := checkedMembershipSuccessor(current, next); err != nil {
		return EncoderMembershipSnapshot{}, err
	}
	if err := controller.source.CompareAndSwap(ctx, current.Revision, next); err != nil {
		return EncoderMembershipSnapshot{}, err
	}
	return next, nil
}

// BeginDrain is idempotent. A member cannot move from draining back to active;
// recovering capacity must use a new stable replica identity instead.
func (controller *EncoderMembershipController) BeginDrain(
	ctx context.Context,
	replicaID string,
) (EncoderMembershipSnapshot, error) {
	if controller == nil || !validEncoderReplicaID(replicaID) {
		return EncoderMembershipSnapshot{}, errors.New("ARC encoder drain request is invalid")
	}
	current, err := controller.source.Load(ctx)
	if err != nil {
		return EncoderMembershipSnapshot{}, err
	}
	index := membershipReplicaIndex(current, replicaID)
	if index < 0 {
		return EncoderMembershipSnapshot{}, errors.New("ARC encoder drain replica is unknown")
	}
	if current.Replicas[index].State == EncoderReplicaDraining {
		return current, nil
	}
	next := cloneEncoderMembershipSnapshot(current)
	next.Revision++
	next.Replicas[index].State = EncoderReplicaDraining
	started := controller.now().UTC()
	next.Replicas[index].DrainStartedAt = &started
	if err := checkedMembershipSuccessor(current, next); err != nil {
		return EncoderMembershipSnapshot{}, err
	}
	if err := controller.source.CompareAndSwap(ctx, current.Revision, next); err != nil {
		return EncoderMembershipSnapshot{}, err
	}
	return next, nil
}

// Reconcile removes every eligible zero-reference draining replica in one CAS
// revision. A concurrent membership change returns the normal CAS conflict,
// leaving the previous safe document in place for a later reconciliation.
func (controller *EncoderMembershipController) Reconcile(
	ctx context.Context,
) (EncoderMembershipDrainReport, error) {
	if controller == nil {
		return EncoderMembershipDrainReport{}, errors.New("ARC encoder membership controller is nil")
	}
	current, err := controller.source.Load(ctx)
	if err != nil {
		return EncoderMembershipDrainReport{}, err
	}
	report := EncoderMembershipDrainReport{Revision: current.Revision}
	eligibleIDs, draining := controller.eligibleDrainingReplicaIDs(current)
	report.Draining = draining
	report.Eligible = len(eligibleIDs)
	if len(eligibleIDs) == 0 {
		return report, nil
	}
	references, err := controller.references.CountEncoderReferences(
		ctx,
		eligibleIDs,
	)
	if err != nil {
		return EncoderMembershipDrainReport{}, err
	}
	removeIDs, referenced := unreferencedMembershipReplicaIDs(
		eligibleIDs,
		references,
	)
	report.Referenced = referenced
	if len(removeIDs) == 0 {
		return report, nil
	}
	if len(current.Replicas)-len(removeIDs) < 2 {
		report.MinimumRetained = len(removeIDs)
		return report, nil
	}
	next := membershipWithoutReplicas(current, removeIDs)
	if err := checkedMembershipSuccessor(current, next); err != nil {
		return EncoderMembershipDrainReport{}, err
	}
	if err := controller.source.CompareAndSwap(ctx, current.Revision, next); err != nil {
		return EncoderMembershipDrainReport{}, err
	}
	report.Revision = next.Revision
	report.Removed = len(removeIDs)
	return report, nil
}

func (controller *EncoderMembershipController) eligibleDrainingReplicaIDs(
	membership EncoderMembershipSnapshot,
) ([]string, int) {
	eligible := make([]string, 0, len(membership.Replicas))
	draining := 0
	now := controller.now().UTC()
	for _, replica := range membership.Replicas {
		if replica.State != EncoderReplicaDraining {
			continue
		}
		draining++
		if replica.DrainStartedAt != nil &&
			!now.Before(replica.DrainStartedAt.UTC().Add(controller.idleTTL)) {
			eligible = append(eligible, replica.ID)
		}
	}
	return eligible, draining
}

func unreferencedMembershipReplicaIDs(
	replicaIDs []string,
	references map[string]int,
) (map[string]struct{}, int) {
	removeIDs := make(map[string]struct{}, len(replicaIDs))
	referenced := 0
	for _, replicaID := range replicaIDs {
		if references[replicaID] > 0 {
			referenced++
			continue
		}
		removeIDs[replicaID] = struct{}{}
	}
	return removeIDs, referenced
}

func membershipWithoutReplicas(
	current EncoderMembershipSnapshot,
	removeIDs map[string]struct{},
) EncoderMembershipSnapshot {
	next := cloneEncoderMembershipSnapshot(current)
	next.Revision++
	next.Replicas = next.Replicas[:0]
	for _, replica := range current.Replicas {
		if _, remove := removeIDs[replica.ID]; !remove {
			next.Replicas = append(next.Replicas, replica)
		}
	}
	return next
}
