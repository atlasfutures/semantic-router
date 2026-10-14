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
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

type Runtime struct {
	manifest   *Manifest
	head       *Head
	headParity HeadParityReport
}

type headGolden struct {
	SchemaVersion    string           `json:"schema_version"`
	CheckpointSHA256 string           `json:"checkpoint_sha256"`
	ScoreTolerance   float32          `json:"score_tolerance"`
	Pool             []string         `json:"pool"`
	Cases            []headGoldenCase `json:"cases"`
}

type headGoldenCase struct {
	ID                 string    `json:"id"`
	Embedding          []float32 `json:"embedding"`
	PreviousModelIndex int64     `json:"previous_model_index"`
	RouteCallIndex     uint64    `json:"route_call_index"`
	Scores             []float32 `json:"scores"`
	SelectedIndex      int       `json:"selected_index"`
}

func LoadRuntime(runtimeDir string) (*Runtime, error) {
	manifest, err := loadManifest(runtimeDir)
	if err != nil {
		return nil, err
	}
	weightsPath, err := artifactPath(runtimeDir, manifest.Weights.File)
	if err != nil {
		return nil, fmt.Errorf("resolve ARC weights: %w", err)
	}
	weights, err := readVerifiedFile(
		weightsPath,
		manifest.Weights.SHA256,
		maxSafeTensorFileBytes,
	)
	if err != nil {
		return nil, fmt.Errorf("verify ARC weights: %w", err)
	}
	goldenPath, err := artifactPath(runtimeDir, manifest.Golden.Head.File)
	if err != nil {
		return nil, fmt.Errorf("resolve ARC head golden: %w", err)
	}
	golden, err := readVerifiedFile(
		goldenPath,
		manifest.Golden.Head.SHA256,
		maxGoldenBytes,
	)
	if err != nil {
		return nil, fmt.Errorf("verify ARC head golden: %w", err)
	}
	if manifest.Encoder.Golden != nil {
		encoderGoldenPath, pathErr := artifactPath(
			runtimeDir,
			manifest.Encoder.Golden.File,
		)
		if pathErr != nil {
			return nil, fmt.Errorf(
				"resolve ARC encoder golden: %w",
				pathErr,
			)
		}
		_, verifyErr := readVerifiedFile(
			encoderGoldenPath,
			manifest.Encoder.Golden.SHA256,
			maxGoldenBytes,
		)
		if verifyErr != nil {
			return nil, fmt.Errorf(
				"verify ARC encoder golden: %w",
				verifyErr,
			)
		}
	}
	head, err := loadHead(weights, manifest)
	if err != nil {
		return nil, fmt.Errorf("load ARC head: %w", err)
	}
	parity, err := verifyHeadGolden(golden, manifest, head)
	if err != nil {
		return nil, err
	}
	return &Runtime{
		manifest:   manifest,
		head:       head,
		headParity: parity,
	}, nil
}

func (runtime *Runtime) Scores(
	history []float32,
	previousArm *int,
	turnIndex uint64,
) ([]float32, error) {
	return runtime.head.Scores(history, previousArm, turnIndex)
}

func (runtime *Runtime) WorkerIDs() []string {
	ids := make([]string, len(runtime.manifest.Workers))
	for index := range runtime.manifest.Workers {
		ids[index] = runtime.manifest.Workers[index].ID
	}
	return ids
}

func (runtime *Runtime) Worker(index int) (WorkerManifest, bool) {
	if runtime == nil || runtime.manifest == nil ||
		index < 0 || index >= len(runtime.manifest.Workers) {
		return WorkerManifest{}, false
	}
	return cloneWorkerManifest(runtime.manifest.Workers[index]), true
}

func cloneWorkerManifest(worker WorkerManifest) WorkerManifest {
	worker.CapabilityTags = append([]string(nil), worker.CapabilityTags...)
	worker.OpenRouterProviderOrder = append(
		[]string(nil),
		worker.OpenRouterProviderOrder...,
	)
	worker.ExtraBody = append(json.RawMessage(nil), worker.ExtraBody...)
	if worker.MaxCompletionTokens != nil {
		value := *worker.MaxCompletionTokens
		worker.MaxCompletionTokens = &value
	}
	if worker.Temperature != nil {
		value := *worker.Temperature
		worker.Temperature = &value
	}
	if worker.AttemptDeadlineSeconds != nil {
		value := *worker.AttemptDeadlineSeconds
		worker.AttemptDeadlineSeconds = &value
	}
	return worker
}

func (runtime *Runtime) ArtifactID() string {
	return runtime.manifest.ArtifactID
}

func (runtime *Runtime) EncoderRevision() string {
	return runtime.manifest.Encoder.Revision
}

type EncoderContract struct {
	Model         string
	Revision      string
	Dimension     int
	Serialization string
}

func (runtime *Runtime) EncoderContract() EncoderContract {
	return EncoderContract{
		Model:         runtime.manifest.Encoder.Model,
		Revision:      runtime.manifest.Encoder.Revision,
		Dimension:     runtime.manifest.Encoder.Dimension,
		Serialization: runtime.manifest.Encoder.Serialization,
	}
}

func (runtime *Runtime) HeadParity() HeadParityReport {
	return runtime.headParity
}

func (runtime *Runtime) Policy() *Policy {
	return newPolicy(runtime.manifest)
}

func verifyHeadGolden(
	data []byte,
	manifest *Manifest,
	head *Head,
) (HeadParityReport, error) {
	golden, err := decodeHeadGolden(data, manifest)
	if err != nil {
		return HeadParityReport{}, err
	}
	tolerance := min(
		golden.ScoreTolerance,
		manifest.Golden.Head.ScoreTolerance,
	)
	var maxDrift float32
	matches := 0
	for index := range golden.Cases {
		drift, matchesExpected, err := evaluateHeadGoldenCase(
			&golden.Cases[index],
			len(manifest.Workers),
			head,
		)
		if err != nil {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d: %w",
				index,
				err,
			)
		}
		maxDrift = max(maxDrift, drift)
		if matchesExpected {
			matches++
		}
	}
	parity := float32(matches) / float32(len(golden.Cases))
	report := HeadParityReport{
		Cases:           len(golden.Cases),
		MaxScoreDrift:   maxDrift,
		SelectionParity: parity,
		Tolerance:       tolerance,
	}
	if maxDrift > tolerance ||
		parity < manifest.Golden.RequiredSelectionParity {
		return HeadParityReport{}, fmt.Errorf(
			"ARC head parity failed: max drift %.8f (limit %.8f), "+
				"selection parity %.6f (minimum %.6f)",
			maxDrift,
			tolerance,
			parity,
			manifest.Golden.RequiredSelectionParity,
		)
	}
	return report, nil
}

func decodeHeadGolden(
	data []byte,
	manifest *Manifest,
) (*headGolden, error) {
	var golden headGolden
	if err := decodeStrictJSON(data, &golden); err != nil {
		return nil, fmt.Errorf("parse ARC head golden: %w", err)
	}
	if golden.SchemaVersion != HeadGoldenSchema {
		return nil, fmt.Errorf(
			"unsupported head golden schema %q",
			golden.SchemaVersion,
		)
	}
	if golden.CheckpointSHA256 != manifest.Source.Checkpoint.SHA256 {
		return nil, errors.New(
			"head golden checkpoint hash does not match the manifest",
		)
	}
	if !equalStrings(golden.Pool, manifest.Architecture.Pool) {
		return nil, errors.New(
			"head golden pool order does not match the manifest",
		)
	}
	if len(golden.Cases) == 0 {
		return nil, errors.New("head golden has no cases")
	}
	if !finitePositive32(golden.ScoreTolerance) {
		return nil, errors.New(
			"head golden score tolerance must be finite and positive",
		)
	}
	if err := validateHeadGoldenCaseIDs(golden.Cases); err != nil {
		return nil, err
	}
	return &golden, nil
}

func validateHeadGoldenCaseIDs(cases []headGoldenCase) error {
	seen := make(map[string]struct{}, len(cases))
	for index := range cases {
		id := cases[index].ID
		if id == "" {
			return fmt.Errorf("head golden case %d has no id", index)
		}
		if _, duplicate := seen[id]; duplicate {
			// Golden case IDs are artifact-owned content; report the index
			// only.
			return fmt.Errorf("head golden case %d has a duplicate id", index)
		}
		seen[id] = struct{}{}
	}
	return nil
}

func evaluateHeadGoldenCase(
	testCase *headGoldenCase,
	workerCount int,
	head *Head,
) (float32, bool, error) {
	previous, err := goldenPreviousArm(
		testCase.PreviousModelIndex,
		workerCount,
	)
	if err != nil {
		return 0, false, err
	}
	if len(testCase.Scores) != workerCount {
		return 0, false, errors.New(
			"score count does not match worker count",
		)
	}
	if testCase.SelectedIndex < 0 ||
		testCase.SelectedIndex >= workerCount {
		return 0, false, errors.New("selected index is out of range")
	}
	expectedSelection, ok := ArgmaxFirst(testCase.Scores)
	if !ok || expectedSelection != testCase.SelectedIndex {
		return 0, false, errors.New(
			"selected index disagrees with its scores",
		)
	}
	actual, err := head.Scores(
		testCase.Embedding,
		previous,
		testCase.RouteCallIndex,
	)
	if err != nil {
		return 0, false, err
	}
	maxDrift := float32(0)
	for scoreIndex, expected := range testCase.Scores {
		if float32IsNaNOrInf(expected) {
			return 0, false, errors.New("contains a non-finite score")
		}
		drift := abs32(actual[scoreIndex] - expected)
		maxDrift = max(maxDrift, drift)
	}
	selected, ok := ArgmaxFirst(actual)
	return maxDrift, ok && selected == testCase.SelectedIndex, nil
}

func goldenPreviousArm(value int64, workerCount int) (*int, error) {
	if value == -1 {
		return nil, nil
	}
	if value < 0 || value >= int64(workerCount) {
		return nil, errors.New("previous model index is out of range")
	}
	result := int(value)
	return &result, nil
}

func ArgmaxFirst(values []float32) (int, bool) {
	if len(values) == 0 {
		return 0, false
	}
	best := 0
	if float32IsNaNOrInf(values[0]) {
		return 0, false
	}
	bestKey := float32TotalOrderKey(values[0])
	for index := 1; index < len(values); index++ {
		if float32IsNaNOrInf(values[index]) {
			return 0, false
		}
		key := float32TotalOrderKey(values[index])
		if key > bestKey {
			best = index
			bestKey = key
		}
	}
	return best, true
}

func float32TotalOrderKey(value float32) uint32 {
	bits := math.Float32bits(value)
	if bits&(1<<31) != 0 {
		return ^bits
	}
	return bits | (1 << 31)
}

func abs32(value float32) float32 {
	if value < 0 {
		return -value
	}
	return value
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
