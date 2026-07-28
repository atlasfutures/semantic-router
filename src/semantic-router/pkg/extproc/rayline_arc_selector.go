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

package extproc

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type raylineARCEncoder interface {
	Encode(
		context.Context,
		string,
		[]raylinearc.Turn,
	) (*raylinearc.EncoderResult, error)
}

type raylineARCScorer interface {
	WorkerIDs() []string
	ArtifactID() string
	EncoderRevision() string
	Select(
		[]float32,
		*raylinearc.EpisodeState,
		int,
		time.Time,
	) (raylinearc.Decision, error)
}

type runtimeARCScorer struct {
	runtime *raylinearc.Runtime
	policy  *raylinearc.Policy
}

func (scorer *runtimeARCScorer) WorkerIDs() []string {
	return scorer.runtime.WorkerIDs()
}

func (scorer *runtimeARCScorer) ArtifactID() string {
	return scorer.runtime.ArtifactID()
}

func (scorer *runtimeARCScorer) EncoderRevision() string {
	return scorer.runtime.EncoderRevision()
}

func (scorer *runtimeARCScorer) Select(
	embedding []float32,
	state *raylinearc.EpisodeState,
	inputTokens int,
	now time.Time,
) (raylinearc.Decision, error) {
	rawScores, err := scorer.runtime.Scores(
		embedding,
		state.PreviousArm,
		state.TurnIndex,
	)
	if err != nil {
		return raylinearc.Decision{}, err
	}
	return scorer.policy.Select(rawScores, state, inputTokens, now)
}

type raylineARCSelector struct {
	scorer           raylineARCScorer
	encoder          raylineARCEncoder
	artifactRevision string
	now              func() time.Time
}

type raylineARCSelectionFailure struct {
	class string
}

func (failure *raylineARCSelectionFailure) Error() string {
	return fmt.Sprintf("Rayline ARC selection failed (class=%s)", failure.class)
}

func newRaylineARCSelector(
	scorer raylineARCScorer,
	encoder raylineARCEncoder,
	artifactRevision string,
) *raylineARCSelector {
	return &raylineARCSelector{
		scorer:           scorer,
		encoder:          encoder,
		artifactRevision: artifactRevision,
		now:              time.Now,
	}
}

func (selector *raylineARCSelector) Select(
	ctx context.Context,
	selCtx *selection.SelectionContext,
) (*selection.SelectionResult, error) {
	arcContext, workerIDs, state, err := selector.prepareSelection(selCtx)
	if err != nil {
		return nil, err
	}
	encoded, latency, err := selector.encode(ctx, arcContext)
	if err != nil {
		return nil, err
	}
	decision, err := selector.scorer.Select(
		encoded.Embedding,
		state,
		encoded.SerializedTokens,
		selector.now(),
	)
	if err != nil {
		return nil, arcSelectionFailure("policy")
	}
	if !validARCDecision(decision, workerIDs) {
		return nil, arcSelectionFailure("artifact_result")
	}
	return selector.selectionResult(
		selCtx,
		arcContext,
		workerIDs,
		state,
		encoded,
		decision,
		latency,
	), nil
}

func (selector *raylineARCSelector) prepareSelection(
	selCtx *selection.SelectionContext,
) (
	*selection.RaylineARCSelectionContext,
	[]string,
	*raylinearc.EpisodeState,
	error,
) {
	if selector == nil || selector.scorer == nil || selector.encoder == nil {
		return nil, nil, nil, arcSelectionFailure("not_ready")
	}
	if selCtx == nil || selCtx.RaylineARC == nil {
		return nil, nil, nil, arcSelectionFailure("missing_context")
	}
	arcContext := selCtx.RaylineARC
	if arcContext.PreparationFailure != "" {
		return nil, nil, nil, arcSelectionFailure(
			arcContext.PreparationFailure,
		)
	}
	workerIDs := selector.scorer.WorkerIDs()
	if len(workerIDs) != len(selCtx.CandidateModels) {
		return nil, nil, nil, arcSelectionFailure("candidate_count")
	}
	for index := range workerIDs {
		if selCtx.CandidateModels[index].Model != workerIDs[index] {
			return nil, nil, nil, arcSelectionFailure("candidate_order")
		}
	}
	state := arcContext.State
	if state == nil {
		var err error
		state, err = raylinearc.NewEpisodeState(len(workerIDs))
		if err != nil {
			return nil, nil, nil, arcSelectionFailure("episode_state")
		}
	}
	return arcContext, workerIDs, state, nil
}

func (selector *raylineARCSelector) encode(
	ctx context.Context,
	arcContext *selection.RaylineARCSelectionContext,
) (*raylinearc.EncoderResult, time.Duration, error) {
	started := selector.now()
	encoded, err := selector.encoder.Encode(
		ctx,
		arcContext.EpisodeIDHash,
		arcContext.Turns,
	)
	encoderLatency := selector.now().Sub(started)
	if err != nil {
		return nil, encoderLatency, boundedARCEncoderFailure(err)
	}
	return encoded, encoderLatency, nil
}

func boundedARCEncoderFailure(err error) error {
	var encoderFailure *raylinearc.EncoderFailure
	if errors.As(err, &encoderFailure) {
		return arcSelectionFailure(
			"encoder_" + string(encoderFailure.Class),
		)
	}
	return arcSelectionFailure("encoder")
}

func validARCDecision(
	decision raylinearc.Decision,
	workerIDs []string,
) bool {
	if decision.SelectedArm < 0 ||
		decision.SelectedArm >= len(workerIDs) {
		return false
	}
	return decision.SelectedWorker == workerIDs[decision.SelectedArm]
}

func (selector *raylineARCSelector) selectionResult(
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	workerIDs []string,
	state *raylinearc.EpisodeState,
	encoded *raylinearc.EncoderResult,
	decision raylinearc.Decision,
	encoderLatency time.Duration,
) *selection.SelectionResult {
	selected := &selCtx.CandidateModels[decision.SelectedArm]
	score := float64(decision.AdjustedScores[decision.SelectedArm])
	allScores := make(map[string]float64, len(workerIDs))
	for index, workerID := range workerIDs {
		allScores[workerID] = float64(decision.AdjustedScores[index])
	}
	return &selection.SelectionResult{
		SelectedModel: selected.Model,
		LoRAName:      selected.LoRAName,
		Score:         score,
		Confidence:    1,
		Method:        selection.MethodRaylineARC,
		Tier:          selection.TierExperimental,
		Reasoning:     "artifact-owned ARC policy",
		AllScores:     allScores,
		RaylineARC: &selection.RaylineARCTrace{
			ArtifactID:          selector.scorer.ArtifactID(),
			ArtifactRevision:    selector.artifactRevision,
			EncoderRevision:     selector.scorer.EncoderRevision(),
			EpisodeIDHash:       arcContext.EpisodeIDHash,
			SelectedArm:         decision.SelectedArm,
			PreviousArm:         cloneInt(state.PreviousArm),
			RawScores:           append([]float32(nil), decision.RawScores...),
			AdjustedScores:      append([]float32(nil), decision.AdjustedScores...),
			SwitchCostUSD:       append([]float64(nil), decision.SwitchCostUSD...),
			CacheMissTokens:     append([]int(nil), decision.CacheMissTokens...),
			Stayed:              decision.Stayed,
			UpgradeExemptions:   append([]bool(nil), decision.ColdSwitchUpgradeExemptions...),
			StayUpgradeExempted: decision.StayUpgradeExempted,
			SerializedTokens:    encoded.SerializedTokens,
			FullHistoryTokens:   encoded.FullHistoryTokens,
			TruncatedTokens:     encoded.TruncatedTokens,
			CachedPrefixTokens:  encoded.CachedPrefixTokens,
			EncoderLatency:      encoderLatency,
		},
	}
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func arcSelectionFailure(class string) *raylineARCSelectionFailure {
	return &raylineARCSelectionFailure{class: class}
}

func (selector *raylineARCSelector) Method() selection.SelectionMethod {
	return selection.MethodRaylineARC
}

func (selector *raylineARCSelector) UpdateFeedback(
	context.Context,
	*selection.Feedback,
) error {
	return errors.New("rayline ARC does not accept online selector feedback")
}

func (selector *raylineARCSelector) Tier() selection.AlgorithmTier {
	return selection.TierExperimental
}

func (selector *raylineARCSelector) ExternalDependencies() []selection.Dependency {
	return []selection.Dependency{
		{
			Name:        "Rayline ARC vLLM encoder",
			Type:        selection.DependencyExternalService,
			Description: "Pinned Qwen pooling service with the Rayline ARC I/O plugin",
			Required:    true,
		},
	}
}
