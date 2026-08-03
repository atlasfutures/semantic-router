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
	"strings"
	"time"
)

func sameEncoderMembership(
	left EncoderMembershipSnapshot,
	right EncoderMembershipSnapshot,
) bool {
	if left.SchemaVersion != right.SchemaVersion ||
		left.Revision != right.Revision ||
		len(left.Replicas) != len(right.Replicas) {
		return false
	}
	for index := range left.Replicas {
		if left.Replicas[index].ID != right.Replicas[index].ID ||
			left.Replicas[index].BaseURL != right.Replicas[index].BaseURL ||
			left.Replicas[index].State != right.Replicas[index].State ||
			!sameMembershipTime(
				left.Replicas[index].DrainStartedAt,
				right.Replicas[index].DrainStartedAt,
			) {
			return false
		}
	}
	return true
}

func sameMembershipTime(left *time.Time, right *time.Time) bool {
	if left == nil || right == nil {
		return left == right
	}
	return left.Equal(*right)
}

func normalizeEncoderMembershipEndpoint(value string) string {
	return strings.TrimRight(strings.TrimSpace(value), "/")
}
