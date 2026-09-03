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

import "errors"

// A hard constraint is a fact about the request, not a preference: an arm the
// request cannot use is not a worse arm, it is no arm at all. This file holds
// the arithmetic that keeps that distinction out of the artifact's own scoring
// policy.

// ErrNoEligibleArm reports that a hard constraint removed every arm. The
// policy cannot name the constraint, so the caller owns the failure class.
var ErrNoEligibleArm = errors.New("ARC policy has no eligible arm")

// validateExclusion refuses a mask that does not describe this worker pool.
// A mask of the wrong length would silently exclude the wrong arms.
func validateExclusion(excluded []bool, workerCount int) error {
	if len(excluded) != 0 && len(excluded) != workerCount {
		return errors.New("ARC exclusion count does not match worker count")
	}
	return nil
}

// argmaxEligible is ArgmaxFirst over the arms the constraint left standing.
// It keeps ArgmaxFirst's total order and its first-wins tie break, so an
// unconstrained call picks exactly what it always picked.
func argmaxEligible(scores []float32, excluded []bool) (int, bool) {
	if len(excluded) == 0 {
		return ArgmaxFirst(scores)
	}
	best := -1
	var bestKey uint32
	for index, value := range scores {
		if float32IsNaNOrInf(value) {
			return 0, false
		}
		if excluded[index] {
			continue
		}
		key := float32TotalOrderKey(value)
		if best < 0 || key > bestKey {
			best = index
			bestKey = key
		}
	}
	if best < 0 {
		return 0, false
	}
	return best, true
}

// eligiblePreviousArm hides an excluded previous arm from the stay margin.
// The margin exists to resist churn, not to override a constraint the request
// itself imposes.
func eligiblePreviousArm(previousArm *int, excluded []bool) *int {
	if previousArm == nil || len(excluded) == 0 {
		return previousArm
	}
	if *previousArm < 0 || *previousArm >= len(excluded) {
		return previousArm
	}
	if excluded[*previousArm] {
		return nil
	}
	return previousArm
}
