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
	"crypto/sha256"
	"encoding/hex"
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

type raylineARCWorkerProvider interface {
	Worker(int) (raylinearc.WorkerManifest, bool)
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

func (scorer *runtimeARCScorer) Worker(
	index int,
) (raylinearc.WorkerManifest, bool) {
	return scorer.runtime.Worker(index)
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

func (selector *raylineARCSelector) Worker(
	index int,
) (raylinearc.WorkerManifest, bool) {
	if selector == nil || selector.scorer == nil {
		return raylinearc.WorkerManifest{}, false
	}
	provider, ok := selector.scorer.(raylineARCWorkerProvider)
	if !ok {
		return raylinearc.WorkerManifest{}, false
	}
	return provider.Worker(index)
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
	state *raylinearc.EpisodeState,
	encoded *raylinearc.EncoderResult,
	decision raylinearc.Decision,
	encoderLatency time.Duration,
) *selection.SelectionResult {
	selected := &selCtx.CandidateModels[decision.SelectedArm]
	score := float64(decision.AdjustedScores[decision.SelectedArm])
	return &selection.SelectionResult{
		SelectedModel: selected.Model,
		LoRAName:      selected.LoRAName,
		Score:         score,
		Confidence:    1,
		Method:        selection.MethodRaylineARC,
		Tier:          selection.TierExperimental,
		Reasoning:     "artifact-owned ARC policy",
		// Artifact arm IDs stay in the bounded ARC-native trace as ordinal
		// arrays. Do not copy them into the generic model-keyed score channel.
		AllScores: nil,
		RaylineARC: &selection.RaylineARCTrace{
			// The artifact ID and revision are deployment-private pins (the
			// Helm profile sources the revision from a Secret); expose only
			// stable hashes so telemetry can detect drift without leaking.
			ArtifactID:           hashedARCIdentity(selector.scorer.ArtifactID()),
			ArtifactRevision:     hashedARCIdentity(selector.artifactRevision),
			EncoderRevision:      selector.scorer.EncoderRevision(),
			EpisodeIDHash:        arcContext.EpisodeIDHash,
			SelectedArm:          decision.SelectedArm,
			PreviousArm:          cloneInt(state.PreviousArm),
			RawScores:            append([]float32(nil), decision.RawScores...),
			AdjustedScores:       append([]float32(nil), decision.AdjustedScores...),
			SwitchCostUSD:        append([]float64(nil), decision.SwitchCostUSD...),
			CacheMissTokens:      append([]int(nil), decision.CacheMissTokens...),
			Stayed:               decision.Stayed,
			UpgradeExemptions:    append([]bool(nil), decision.ColdSwitchUpgradeExemptions...),
			StayUpgradeExempted:  decision.StayUpgradeExempted,
			SerializedTokens:     encoded.SerializedTokens,
			FullHistoryTokens:    encoded.FullHistoryTokens,
			TruncatedTokens:      encoded.TruncatedTokens,
			CachedPrefixTokens:   encoded.CachedPrefixTokens,
			RetainedPrefixTokens: encoded.RetainedPrefixTokens,
			AppendedTokens:       encoded.AppendedTokens,
			SessionAction:        encoded.SessionAction,
			SessionRevision:      encoded.SessionRevision,
			EncoderLatency:       encoderLatency,
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

func hashedARCIdentity(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:8])
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
