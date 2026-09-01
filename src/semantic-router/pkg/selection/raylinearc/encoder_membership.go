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
	"fmt"
	"net/url"
	"strings"
	"time"
)

const EncoderMembershipSchemaV1 = "rayline.arc.encoder-membership.v1"

var (
	ErrEncoderMembershipNotFound = errors.New("ARC encoder membership is not published")
	ErrEncoderMembershipConflict = errors.New("ARC encoder membership revision conflict")
)

// EncoderMembershipReplica is a stable retained-encoder identity published by
// the controller. DrainStartedAt is required only while the member is
// draining, so a controller can prove that the entire episode idle window has
// elapsed before removal.
type EncoderMembershipReplica struct {
	ID             string              `json:"id"`
	BaseURL        string              `json:"base_url"`
	State          EncoderReplicaState `json:"state"`
	DrainStartedAt *time.Time          `json:"drain_started_at,omitempty"`
}

// EncoderMembershipSnapshot is the reviewed, revisioned dynamic membership
// document. A router accepts only strictly increasing revisions; endpoint
// identity is additionally pinned in-process by DynamicEncoderPool.
type EncoderMembershipSnapshot struct {
	SchemaVersion string                     `json:"schema_version"`
	Revision      uint64                     `json:"revision"`
	Replicas      []EncoderMembershipReplica `json:"replicas"`
}

// EncoderMembershipReader is the runtime's read-only control-plane seam.
// Mutating membership remains a controller-only capability.
type EncoderMembershipReader interface {
	Load(context.Context) (EncoderMembershipSnapshot, error)
}

// EncoderMembershipSource is the controller's mutation-capable extension of
// the runtime read seam.
type EncoderMembershipSource interface {
	EncoderMembershipReader
	CompareAndSwap(context.Context, uint64, EncoderMembershipSnapshot) error
}

// EncoderMembershipReferenceCounter returns the number of persisted episode
// states which name each replica as an owner or a visited owner. It exposes no
// prompt or episode identity data.
type EncoderMembershipReferenceCounter interface {
	CountEncoderReferences(context.Context, []string) (map[string]int, error)
}

func validateEncoderMembershipSnapshot(snapshot EncoderMembershipSnapshot) error {
	if snapshot.SchemaVersion != EncoderMembershipSchemaV1 {
		return errors.New("ARC encoder membership schema version is unsupported")
	}
	if snapshot.Revision == 0 {
		return errors.New("ARC encoder membership revision is required")
	}
	if len(snapshot.Replicas) < 2 || len(snapshot.Replicas) > maxEncoderReplicas {
		return errors.New("ARC encoder membership replica count must be between 2 and 8")
	}
	seenIDs := make(map[string]struct{}, len(snapshot.Replicas))
	seenURLs := make(map[string]struct{}, len(snapshot.Replicas))
	active := 0
	for index := range snapshot.Replicas {
		if err := validateEncoderMembershipReplica(
			&snapshot.Replicas[index],
			seenIDs,
			seenURLs,
		); err != nil {
			return err
		}
		if snapshot.Replicas[index].State == EncoderReplicaActive {
			active++
		}
	}
	if active == 0 {
		return errors.New("ARC encoder membership requires an active member")
	}
	return nil
}

func validateEncoderMembershipReplica(
	replica *EncoderMembershipReplica,
	seenIDs map[string]struct{},
	seenURLs map[string]struct{},
) error {
	if err := validateEncoderMembershipReplicaID(replica, seenIDs); err != nil {
		return err
	}
	if err := validateEncoderMembershipReplicaURL(replica, seenURLs); err != nil {
		return err
	}
	return validateEncoderMembershipReplicaState(replica)
}

func validateEncoderMembershipReplicaID(
	replica *EncoderMembershipReplica,
	seenIDs map[string]struct{},
) error {
	if replica == nil || !validEncoderReplicaID(replica.ID) {
		return errors.New("ARC encoder membership replica ID is invalid")
	}
	if _, exists := seenIDs[replica.ID]; exists {
		return errors.New("ARC encoder membership replica ID is duplicated")
	}
	seenIDs[replica.ID] = struct{}{}
	return nil
}

func validateEncoderMembershipReplicaURL(
	replica *EncoderMembershipReplica,
	seenURLs map[string]struct{},
) error {
	parsed, err := url.Parse(strings.TrimSpace(replica.BaseURL))
	if err != nil || parsed.Host == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return errors.New("ARC encoder membership replica URL is invalid")
	}
	urlKey := strings.TrimRight(strings.TrimSpace(replica.BaseURL), "/")
	if _, exists := seenURLs[urlKey]; exists {
		return errors.New("ARC encoder membership replica URL is duplicated")
	}
	seenURLs[urlKey] = struct{}{}
	return nil
}

func validateEncoderMembershipReplicaState(
	replica *EncoderMembershipReplica,
) error {
	switch replica.State {
	case EncoderReplicaActive:
		if replica.DrainStartedAt != nil {
			return errors.New("ARC active encoder membership replica has a drain start")
		}
	case EncoderReplicaDraining:
		if replica.DrainStartedAt == nil || replica.DrainStartedAt.IsZero() {
			return errors.New("ARC draining encoder membership replica requires a drain start")
		}
	default:
		return errors.New("ARC encoder membership replica state is invalid")
	}
	return nil
}

func cloneEncoderMembershipSnapshot(
	snapshot EncoderMembershipSnapshot,
) EncoderMembershipSnapshot {
	cloned := EncoderMembershipSnapshot{
		SchemaVersion: snapshot.SchemaVersion,
		Revision:      snapshot.Revision,
		Replicas:      make([]EncoderMembershipReplica, len(snapshot.Replicas)),
	}
	for index := range snapshot.Replicas {
		cloned.Replicas[index] = snapshot.Replicas[index]
		if snapshot.Replicas[index].DrainStartedAt != nil {
			value := *snapshot.Replicas[index].DrainStartedAt
			cloned.Replicas[index].DrainStartedAt = &value
		}
	}
	return cloned
}

func membershipReplicaIndex(
	snapshot EncoderMembershipSnapshot,
	id string,
) int {
	for index := range snapshot.Replicas {
		if snapshot.Replicas[index].ID == id {
			return index
		}
	}
	return -1
}

func checkedMembershipSuccessor(
	previous EncoderMembershipSnapshot,
	next EncoderMembershipSnapshot,
) error {
	if err := validateEncoderMembershipSnapshot(next); err != nil {
		return err
	}
	if next.Revision != previous.Revision+1 {
		return fmt.Errorf("ARC encoder membership revision must advance by one")
	}
	for _, replica := range previous.Replicas {
		index := membershipReplicaIndex(next, replica.ID)
		if index < 0 {
			continue
		}
		nextReplica := next.Replicas[index]
		if strings.TrimRight(replica.BaseURL, "/") !=
			strings.TrimRight(nextReplica.BaseURL, "/") {
			return errors.New("ARC encoder membership cannot change a stable replica endpoint")
		}
		if replica.State == EncoderReplicaDraining &&
			nextReplica.State == EncoderReplicaActive {
			return errors.New("ARC encoder membership cannot reactivate a draining replica")
		}
	}
	return nil
}
