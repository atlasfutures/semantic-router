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
	"errors"
	"fmt"
	"math"
	"time"
)

const (
	cacheReturnWarmSeconds  = 150.0
	cacheReturnColdSeconds  = 300.0
	cacheReturnWarmHitRatio = 0.8
	cacheReturnColdHitRatio = 0.2
)

type WorkerWarmth struct {
	LastUsed        time.Time
	LastInputTokens int
}

type EpisodeState struct {
	PreviousArm          *int
	TurnIndex            uint64
	Warmth               []*WorkerWarmth
	EncoderOwner         string
	EncoderVisitedOwners []string
}

func NewEpisodeState(workerCount int) (*EpisodeState, error) {
	if workerCount <= 0 {
		return nil, errors.New("worker count must be positive")
	}
	return &EpisodeState{
		Warmth: make([]*WorkerWarmth, workerCount),
	}, nil
}

// Commit advances episode state after the caller has observed upstream 2xx
// response headers. Selection itself never mutates the state.
func (state *EpisodeState) Commit(
	arm int,
	inputTokens int,
	now time.Time,
) error {
	if state == nil {
		return errors.New("episode state is required")
	}
	if arm < 0 || arm >= len(state.Warmth) {
		return errors.New("selected arm is out of range")
	}
	if inputTokens < 0 {
		return errors.New("input token count must be nonnegative")
	}
	if now.IsZero() {
		return errors.New("commit time is required")
	}
	selected := arm
	state.PreviousArm = &selected
	if state.TurnIndex < math.MaxUint64 {
		state.TurnIndex++
	}
	state.Warmth[arm] = &WorkerWarmth{
		LastUsed:        now,
		LastInputTokens: inputTokens,
	}
	return nil
}

type Decision struct {
	SelectedArm                 int
	SelectedWorker              string
	RawScores                   []float32
	AdjustedScores              []float32
	SwitchCostUSD               []float64
	CacheMissTokens             []int
	Stayed                      bool
	ColdSwitchUpgradeExemptions []bool
	StayUpgradeExempted         bool
}

type Policy struct {
	workers  []WorkerManifest
	contract PolicyManifest
}

type policyAdjustment struct {
	scores            []float32
	switchCost        []float64
	cacheMissTokens   []int
	upgradeExemptions []bool
}

func newPolicy(manifest *Manifest) *Policy {
	workers := append([]WorkerManifest(nil), manifest.Workers...)
	return &Policy{
		workers:  workers,
		contract: manifest.Policy,
	}
}

func (policy *Policy) Select(
	rawScores []float32,
	state *EpisodeState,
	inputTokens int,
	now time.Time,
) (Decision, error) {
	if err := policy.validateSelectionInputs(
		rawScores,
		state,
		inputTokens,
		now,
	); err != nil {
		return Decision{}, err
	}
	adjustment := policy.adjustScores(
		rawScores,
		state,
		inputTokens,
		now,
	)
	tentative, ok := ArgmaxFirst(adjustment.scores)
	if !ok {
		return Decision{}, errors.New("ARC policy has no candidate arm")
	}
	selected, stayed, stayUpgradeExempted := policy.applyStayMargin(
		tentative,
		adjustment.scores,
		state.PreviousArm,
	)
	return Decision{
		SelectedArm:                 selected,
		SelectedWorker:              policy.workers[selected].ID,
		RawScores:                   append([]float32(nil), rawScores...),
		AdjustedScores:              adjustment.scores,
		SwitchCostUSD:               adjustment.switchCost,
		CacheMissTokens:             adjustment.cacheMissTokens,
		Stayed:                      stayed,
		ColdSwitchUpgradeExemptions: adjustment.upgradeExemptions,
		StayUpgradeExempted:         stayUpgradeExempted,
	}, nil
}

func (policy *Policy) validateSelectionInputs(
	rawScores []float32,
	state *EpisodeState,
	inputTokens int,
	now time.Time,
) error {
	if err := validateSelectionRequest(
		policy,
		rawScores,
		inputTokens,
		now,
	); err != nil {
		return err
	}
	return validateEpisodeState(state, len(policy.workers))
}

func validateSelectionRequest(
	policy *Policy,
	rawScores []float32,
	inputTokens int,
	now time.Time,
) error {
	if policy == nil {
		return errors.New("ARC policy is required")
	}
	if len(rawScores) == 0 || len(rawScores) != len(policy.workers) {
		return errors.New(
			"ARC score count does not match worker count",
		)
	}
	if inputTokens < 0 {
		return errors.New("input token count must be nonnegative")
	}
	if now.IsZero() {
		return errors.New("selection time is required")
	}
	for _, value := range rawScores {
		if float32IsNaNOrInf(value) {
			return errors.New(
				"ARC scores contain a non-finite value",
			)
		}
	}
	return nil
}

func validateEpisodeState(state *EpisodeState, workerCount int) error {
	if state == nil {
		return errors.New("episode state is required")
	}
	if len(state.Warmth) != workerCount {
		return errors.New(
			"episode warmth count does not match worker count",
		)
	}
	if state.PreviousArm != nil &&
		(*state.PreviousArm < 0 || *state.PreviousArm >= workerCount) {
		return errors.New("previous arm is out of range")
	}
	if err := validatePersistedEncoderAffinity(
		state.EncoderOwner,
		state.EncoderVisitedOwners,
	); err != nil {
		return err
	}
	return validateWarmth(state.Warmth)
}

func validatePersistedEncoderAffinity(owner string, visited []string) error {
	if owner != "" && !validEncoderReplicaID(owner) {
		return errors.New("encoder owner is invalid")
	}
	if len(visited) > maxEncoderReplicas {
		return errors.New("encoder visited-owner count exceeds limit")
	}
	seen := make(map[string]struct{}, len(visited))
	for _, replicaID := range visited {
		if !validEncoderReplicaID(replicaID) {
			return errors.New("encoder visited owner is invalid")
		}
		if _, exists := seen[replicaID]; exists {
			return errors.New("encoder visited owner is duplicated")
		}
		seen[replicaID] = struct{}{}
	}
	if owner != "" {
		if _, exists := seen[owner]; !exists {
			return errors.New("encoder owner is absent from visited owners")
		}
	}
	return nil
}

func validateWarmth(warmth []*WorkerWarmth) error {
	for index, value := range warmth {
		if value == nil {
			continue
		}
		if value.LastUsed.IsZero() || value.LastInputTokens < 0 {
			return fmt.Errorf("worker %d warmth is invalid", index)
		}
	}
	return nil
}

func (policy *Policy) adjustScores(
	rawScores []float32,
	state *EpisodeState,
	inputTokens int,
	now time.Time,
) policyAdjustment {
	result := policyAdjustment{
		scores:            append([]float32(nil), rawScores...),
		switchCost:        make([]float64, len(rawScores)),
		cacheMissTokens:   make([]int, len(rawScores)),
		upgradeExemptions: make([]bool, len(rawScores)),
	}
	if state.PreviousArm == nil {
		return result
	}
	previous := *state.PreviousArm
	previousRate := policy.workers[previous].EstimatedInputCostPerToken
	for index := range policy.workers {
		if index == previous {
			continue
		}
		worker := &policy.workers[index]
		exempt := policy.contract.ColdSwitchUpgradeExempt &&
			worker.EstimatedInputCostPerToken > previousRate
		result.upgradeExemptions[index] = exempt
		if exempt {
			continue
		}
		missTokens, cost := switchCostForWorker(
			worker,
			state.Warmth[index],
			inputTokens,
			now,
		)
		result.cacheMissTokens[index] = missTokens
		result.switchCost[index] = cost
		result.scores[index] -= float32(
			float64(policy.contract.ColdSwitchMarginPerUSD) * cost,
		)
	}
	return result
}

func (policy *Policy) applyStayMargin(
	tentative int,
	adjustedScores []float32,
	previousArm *int,
) (int, bool, bool) {
	if previousArm == nil || tentative == *previousArm {
		return tentative, false, false
	}
	previous := *previousArm
	upgradeExempted := policy.contract.StayMarginUpgradeExempt &&
		policy.workers[tentative].EstimatedInputCostPerToken >
			policy.workers[previous].EstimatedInputCostPerToken
	if upgradeExempted ||
		adjustedScores[tentative]-adjustedScores[previous] >
			policy.contract.PreviousWorkerStayMargin {
		return tentative, false, upgradeExempted
	}
	return previous, true, false
}

func switchCostForWorker(
	worker *WorkerManifest,
	warmth *WorkerWarmth,
	inputTokens int,
	now time.Time,
) (int, float64) {
	cacheablePrefix := 0
	var seconds *float64
	if warmth != nil {
		cacheablePrefix = min(warmth.LastInputTokens, inputTokens)
		elapsed := now.Sub(warmth.LastUsed).Seconds()
		if elapsed < 0 {
			elapsed = 0
		}
		seconds = &elapsed
	}
	uncachedSuffix := inputTokens - cacheablePrefix
	missRatio := 1 - expectedCacheHitRatio(seconds)
	decayed := int(math.Round(float64(cacheablePrefix) * missRatio))
	missTokens := uncachedSuffix + decayed
	discount := max(
		worker.EstimatedInputCostPerToken-
			worker.EstimatedCacheReadCostPerToken,
		0,
	)
	return missTokens, float64(missTokens) * discount
}

func expectedCacheHitRatio(seconds *float64) float64 {
	if seconds == nil {
		return 0
	}
	if *seconds <= cacheReturnWarmSeconds {
		return cacheReturnWarmHitRatio
	}
	if *seconds >= cacheReturnColdSeconds {
		return cacheReturnColdHitRatio
	}
	fraction := (*seconds - cacheReturnWarmSeconds) /
		(cacheReturnColdSeconds - cacheReturnWarmSeconds)
	return cacheReturnWarmHitRatio +
		fraction*(cacheReturnColdHitRatio-cacheReturnWarmHitRatio)
}
