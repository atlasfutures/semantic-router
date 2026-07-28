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
		encoderGoldenPath, err := artifactPath(
			runtimeDir,
			manifest.Encoder.Golden.File,
		)
		if err != nil {
			return nil, fmt.Errorf("resolve ARC encoder golden: %w", err)
		}
		if _, err := readVerifiedFile(
			encoderGoldenPath,
			manifest.Encoder.Golden.SHA256,
			maxGoldenBytes,
		); err != nil {
			return nil, fmt.Errorf("verify ARC encoder golden: %w", err)
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
	var golden headGolden
	if err := decodeStrictJSON(data, &golden); err != nil {
		return HeadParityReport{}, fmt.Errorf("parse ARC head golden: %w", err)
	}
	if golden.SchemaVersion != HeadGoldenSchema {
		return HeadParityReport{}, fmt.Errorf(
			"unsupported head golden schema %q",
			golden.SchemaVersion,
		)
	}
	if golden.CheckpointSHA256 != manifest.Source.Checkpoint.SHA256 {
		return HeadParityReport{}, errors.New(
			"head golden checkpoint hash does not match the manifest",
		)
	}
	if !equalStrings(golden.Pool, manifest.Architecture.Pool) {
		return HeadParityReport{}, errors.New(
			"head golden pool order does not match the manifest",
		)
	}
	if len(golden.Cases) == 0 {
		return HeadParityReport{}, errors.New("head golden has no cases")
	}
	if !finitePositive32(golden.ScoreTolerance) {
		return HeadParityReport{}, errors.New(
			"head golden score tolerance must be finite and positive",
		)
	}
	tolerance := min(
		golden.ScoreTolerance,
		manifest.Golden.Head.ScoreTolerance,
	)
	var maxDrift float32
	matches := 0
	seenIDs := make(map[string]struct{}, len(golden.Cases))
	for index := range golden.Cases {
		testCase := &golden.Cases[index]
		if testCase.ID == "" {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d has no id",
				index,
			)
		}
		if _, duplicate := seenIDs[testCase.ID]; duplicate {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d has duplicate id %q",
				index,
				testCase.ID,
			)
		}
		seenIDs[testCase.ID] = struct{}{}
		previous, err := goldenPreviousArm(
			testCase.PreviousModelIndex,
			len(manifest.Workers),
		)
		if err != nil {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d: %w",
				index,
				err,
			)
		}
		if len(testCase.Scores) != len(manifest.Workers) {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d score count does not match worker count",
				index,
			)
		}
		if testCase.SelectedIndex < 0 ||
			testCase.SelectedIndex >= len(manifest.Workers) {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d selected index is out of range",
				index,
			)
		}
		expectedSelection, ok := ArgmaxFirst(testCase.Scores)
		if !ok || expectedSelection != testCase.SelectedIndex {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d selected index disagrees with its scores",
				index,
			)
		}
		actual, err := head.Scores(
			testCase.Embedding,
			previous,
			testCase.RouteCallIndex,
		)
		if err != nil {
			return HeadParityReport{}, fmt.Errorf(
				"head golden case %d: %w",
				index,
				err,
			)
		}
		for scoreIndex, expected := range testCase.Scores {
			if float32IsNaNOrInf(expected) {
				return HeadParityReport{}, fmt.Errorf(
					"head golden case %d contains a non-finite score",
					index,
				)
			}
			drift := abs32(actual[scoreIndex] - expected)
			maxDrift = max(maxDrift, drift)
		}
		selected, ok := ArgmaxFirst(actual)
		if ok && selected == testCase.SelectedIndex {
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
