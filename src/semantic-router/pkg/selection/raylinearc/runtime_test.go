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
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

type syntheticTensor struct {
	shape  []uint64
	values []float32
}

func TestLoadRuntimeAndHeadParity(t *testing.T) {
	t.Parallel()
	runtimeDir := writeSyntheticRuntime(t, nil)

	runtime, err := LoadRuntime(runtimeDir)
	if err != nil {
		t.Fatalf("LoadRuntime() error = %v", err)
	}
	if got, want := runtime.WorkerIDs(), []string{"worker-a", "worker-b"}; !equalStrings(got, want) {
		t.Fatalf("WorkerIDs() = %v, want %v", got, want)
	}
	report := runtime.HeadParity()
	if report.Cases != 2 || report.SelectionParity != 1 ||
		report.MaxScoreDrift > report.Tolerance {
		t.Fatalf("HeadParity() = %+v", report)
	}

	scores, err := runtime.Scores([]float32{3, 4}, nil, 0)
	if err != nil {
		t.Fatalf("Scores() error = %v", err)
	}
	assertScoresClose(t, scores, syntheticExpectedScores(), 1e-6)
	previous := 1
	scores, err = runtime.Scores([]float32{3, 4}, &previous, 9)
	if err != nil {
		t.Fatalf("Scores(previous) error = %v", err)
	}
	assertScoresClose(t, scores, syntheticExpectedScores(), 1e-6)

	for name, history := range map[string][]float32{
		"dimension": {1},
		"zero norm": {0, 0},
		"nonfinite": {float32(math.Inf(1)), 0},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := runtime.Scores(history, nil, 0); err == nil {
				t.Fatal("Scores() unexpectedly succeeded")
			}
		})
	}
	outOfRange := 2
	if _, err := runtime.Scores([]float32{3, 4}, &outOfRange, 0); err == nil {
		t.Fatal("Scores() accepted an out-of-range previous arm")
	}
}

func TestRuntimeArtifactFromEnvironment(t *testing.T) {
	runtimeDir := os.Getenv("RAYLINE_ARC_TEST_RUNTIME_DIR")
	if runtimeDir == "" {
		t.Skip("RAYLINE_ARC_TEST_RUNTIME_DIR is not set")
	}
	if _, err := LoadRuntime(runtimeDir); err != nil {
		t.Fatalf("LoadRuntime(%q) error = %v", runtimeDir, err)
	}
}

func TestRuntimeFailsClosedOnArtifactMutation(t *testing.T) {
	t.Parallel()
	runtimeDir := writeSyntheticRuntime(t, nil)
	path := filepath.Join(runtimeDir, "head.safetensors")
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write([]byte{0}); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRuntime(runtimeDir); err == nil ||
		!strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Fatalf("LoadRuntime() error = %v, want SHA256 mismatch", err)
	}
}

func TestRuntimeRejectsEscapingSymlink(t *testing.T) {
	t.Parallel()
	runtimeDir := writeSyntheticRuntime(t, nil)
	outsideDir := t.TempDir()
	outside := filepath.Join(outsideDir, "outside.safetensors")
	weights, err := os.ReadFile(filepath.Join(runtimeDir, "head.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(outside, weights, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(runtimeDir, "head.safetensors")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(runtimeDir, "head.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRuntime(runtimeDir); err == nil ||
		!strings.Contains(err.Error(), "escapes") {
		t.Fatalf("LoadRuntime() error = %v, want path escape", err)
	}
}

func TestRuntimeRejectsInvalidTensorShape(t *testing.T) {
	t.Parallel()
	runtimeDir := writeSyntheticRuntime(t, func(
		_ *Manifest,
		tensors map[string]syntheticTensor,
	) {
		tensors["q_network.head.bias"] = syntheticTensor{
			shape:  []uint64{2},
			values: []float32{0.1, 0.1},
		}
	})
	if _, err := LoadRuntime(runtimeDir); err == nil ||
		!strings.Contains(err.Error(), "q_network.head.bias") {
		t.Fatalf("LoadRuntime() error = %v, want head-bias shape error", err)
	}
}

func TestRuntimeRejectsMutableDecisionPrices(t *testing.T) {
	t.Parallel()
	runtimeDir := writeSyntheticRuntime(t, func(
		manifest *Manifest,
		_ map[string]syntheticTensor,
	) {
		manifest.PricingSnapshot.MutableLivePricesAffectDecisions = true
	})
	if _, err := LoadRuntime(runtimeDir); err == nil ||
		!strings.Contains(err.Error(), "mutable live prices") {
		t.Fatalf("LoadRuntime() error = %v, want mutable-price error", err)
	}
}

func TestSyntheticTieHeadSelectsFirstArm(t *testing.T) {
	t.Parallel()
	tensors := syntheticTensors()
	tensors["q_network.head.weight"] = syntheticTensor{
		shape:  []uint64{1, 2},
		values: []float32{0, 0},
	}
	tensors["q_network.head.bias"] = syntheticTensor{
		shape:  []uint64{1},
		values: []float32{1},
	}
	head, err := loadHead(
		encodeSafeTensors(t, tensors),
		func() *Manifest {
			manifest := syntheticManifest()
			return &manifest
		}(),
	)
	if err != nil {
		t.Fatal(err)
	}
	scores, err := head.Scores([]float32{3, 4}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if scores[0] != scores[1] {
		t.Fatalf("tie-head scores = %v, want an exact tie", scores)
	}
	if selected, ok := ArgmaxFirst(scores); !ok || selected != 0 {
		t.Fatalf("ArgmaxFirst(tie head) = %d, %v; want 0, true", selected, ok)
	}
}

func TestDecodeSafeTensorsRejectsMalformedInput(t *testing.T) {
	t.Parallel()
	valid := encodeSafeTensors(t, map[string]syntheticTensor{
		"value": {shape: []uint64{1}, values: []float32{1}},
	})
	if _, err := decodeSafeTensors(valid, []string{"value"}); err != nil {
		t.Fatalf("decodeSafeTensors(valid) error = %v", err)
	}

	duplicateHeader := `{"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},` +
		`"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	nonfinite := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint32(
		nonfinite[len(nonfinite)-4:],
		math.Float32bits(float32(math.Inf(1))),
	)
	tests := map[string]struct {
		data     []byte
		expected []string
	}{
		"short prefix": {
			data:     []byte{1},
			expected: []string{"value"},
		},
		"duplicate header": {
			data:     safeTensorWithRawHeader(duplicateHeader, []byte{0, 0, 0, 0}),
			expected: []string{"value"},
		},
		"missing tensor": {
			data:     valid,
			expected: []string{"missing"},
		},
		"unexpected tensor": {
			data:     valid,
			expected: nil,
		},
		"nonfinite value": {
			data:     nonfinite,
			expected: []string{"value"},
		},
		"wrong dtype": {
			data: safeTensorWithRawHeader(
				`{"value":{"dtype":"F16","shape":[1],"data_offsets":[0,4]}}`,
				[]byte{0, 0, 0, 0},
			),
			expected: []string{"value"},
		},
		"trailing data": {
			data:     append(append([]byte(nil), valid...), 0),
			expected: []string{"value"},
		},
		"overlapping tensors": {
			data: safeTensorWithRawHeader(
				`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},`+
					`"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
				[]byte{0, 0, 0, 0},
			),
			expected: []string{"a", "b"},
		},
		"overflowing shape": {
			data: safeTensorWithRawHeader(
				`{"value":{"dtype":"F32","shape":[18446744073709551615,2],`+
					`"data_offsets":[0,4]}}`,
				[]byte{0, 0, 0, 0},
			),
			expected: []string{"value"},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if _, err := decodeSafeTensors(test.data, test.expected); err == nil {
				t.Fatal("decodeSafeTensors() unexpectedly succeeded")
			}
		})
	}
}

func TestArgmaxFirstUsesStableFloat32TotalOrder(t *testing.T) {
	t.Parallel()
	if index, ok := ArgmaxFirst([]float32{1, 2, 2}); !ok || index != 1 {
		t.Fatalf("ArgmaxFirst(tie) = %d, %v; want 1, true", index, ok)
	}
	negativeZero := float32(math.Copysign(0, -1))
	if index, ok := ArgmaxFirst([]float32{negativeZero, 0}); !ok || index != 1 {
		t.Fatalf("ArgmaxFirst(signed zero) = %d, %v; want 1, true", index, ok)
	}
	if _, ok := ArgmaxFirst([]float32{float32(math.NaN())}); ok {
		t.Fatal("ArgmaxFirst() accepted NaN")
	}
	if _, ok := ArgmaxFirst(nil); ok {
		t.Fatal("ArgmaxFirst() accepted an empty slice")
	}
}

func writeSyntheticRuntime(
	t *testing.T,
	mutate func(*Manifest, map[string]syntheticTensor),
) string {
	t.Helper()
	runtimeDir := t.TempDir()
	tensors := syntheticTensors()
	manifest := syntheticManifest()
	if mutate != nil {
		mutate(&manifest, tensors)
	}
	weights := encodeSafeTensors(t, tensors)
	weightsPath := filepath.Join(runtimeDir, manifest.Weights.File)
	if err := os.WriteFile(weightsPath, weights, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest.Weights.SHA256 = sha256Hex(weights)

	expected := syntheticExpectedScores()
	golden := headGolden{
		SchemaVersion:    HeadGoldenSchema,
		CheckpointSHA256: manifest.Source.Checkpoint.SHA256,
		ScoreTolerance:   1e-5,
		Pool:             append([]string(nil), manifest.Architecture.Pool...),
		Cases: []headGoldenCase{
			{
				ID:                 "cold-start",
				Embedding:          []float32{3, 4},
				PreviousModelIndex: -1,
				RouteCallIndex:     0,
				Scores:             expected,
				SelectedIndex:      1,
			},
			{
				ID:                 "warm-return",
				Embedding:          []float32{3, 4},
				PreviousModelIndex: 1,
				RouteCallIndex:     9,
				Scores:             expected,
				SelectedIndex:      1,
			},
		},
	}
	goldenBytes, err := json.MarshalIndent(golden, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	err = os.WriteFile(
		filepath.Join(runtimeDir, manifest.Golden.Head.File),
		goldenBytes,
		0o600,
	)
	if err != nil {
		t.Fatal(err)
	}
	manifest.Golden.Head.SHA256 = sha256Hex(goldenBytes)
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(runtimeDir, "manifest.json"),
		manifestBytes,
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	return runtimeDir
}

func syntheticManifest() Manifest {
	checkpoint := sha256.Sum256([]byte("synthetic checkpoint"))
	return Manifest{
		SchemaVersion:  ManifestSchema,
		ArtifactID:     "public-synthetic-arc",
		CreatedAt:      "2026-01-01T00:00:00Z",
		ExporterCommit: "synthetic-exporter-commit",
		Weights: WeightsManifest{
			File:  "head.safetensors",
			DType: "F32",
		},
		Source: SourceManifest{
			Checkpoint: SourceCheckpoint{
				File:        "synthetic-checkpoint.pt",
				SHA256:      hex.EncodeToString(checkpoint[:]),
				ModalVolume: "synthetic-volume",
				ModalPath:   "/synthetic/checkpoint.pt",
			},
			ModelMeta: DigestManifest{
				SHA256: strings.Repeat("1", 64),
			},
			ReliabilityTable: DigestManifest{
				SHA256: strings.Repeat("2", 64),
			},
			TrainReport: DigestManifest{
				SHA256: strings.Repeat("3", 64),
			},
		},
		Encoder: EncoderManifest{
			Model:                   "synthetic/encoder",
			Revision:                "synthetic-revision",
			Dimension:               2,
			MaxTokens:               1024,
			MinRecentTurns:          1,
			MinRecentTokens:         1,
			Serialization:           SerializationName,
			Pooling:                 "masked_mean",
			NormalizeEmbeddings:     true,
			AttentionImplementation: "sdpa",
			DType:                   "BF16",
			IncrementalDefault:      true,
			KVChunkTokens:           32,
			KVSessionBudgetTokens:   2048,
			KVProcessBudgetTokens:   4096,
			KVIdleTTLSeconds:        300,
			Native:                  json.RawMessage(`{"runtime":"synthetic"}`),
		},
		Architecture: ArchitectureManifest{
			Name:                  "switch_aware",
			HistoryDimension:      2,
			ArmEmbeddingDimension: 2,
			JointInputDimension:   8,
			HiddenDimensions:      []int{2, 2},
			Dropout:               0.1,
			Pool:                  []string{"worker-a", "worker-b"},
		},
		Policy: PolicyManifest{
			PreviousWorkerStayMargin: 0.05,
			ColdSwitchMarginPerUSD:   1,
			ColdSwitchUpgradeExempt:  true,
			StayMarginUpgradeExempt:  false,
			ReferenceWorker:          "worker-a",
			ReferenceMargin:          0,
		},
		Workers: []WorkerManifest{
			syntheticWorker("worker-a", 0.00001, 0.000001),
			syntheticWorker("worker-b", 0.00002, 0.000002),
		},
		PricingSnapshot: PricingSnapshot{
			ConfigPath:   "synthetic/pricing.yaml",
			ConfigCommit: "synthetic-pricing-commit",
		},
		Golden: GoldenManifest{
			Head: FileGoldenManifest{
				File:           "head_golden.json",
				ScoreTolerance: 1e-5,
			},
			AdjustedTopTwoGapTolerance: 1e-5,
			RequiredSelectionParity:    1,
		},
	}
}

func syntheticWorker(
	id string,
	inputCost float64,
	cacheReadCost float64,
) WorkerManifest {
	return WorkerManifest{
		ID:                              id,
		Model:                           "synthetic/" + id,
		APIKeyEnv:                       "SYNTHETIC_API_KEY",
		EstimatedInputCostPerToken:      inputCost,
		EstimatedCacheReadCostPerToken:  cacheReadCost,
		EstimatedCacheWriteCostPerToken: inputCost,
		EstimatedOutputCostPerToken:     inputCost,
		LatencyMS:                       100,
		CapabilityTags:                  []string{"synthetic"},
		OpenRouterProviderSlug:          "synthetic",
		OpenRouterProviderName:          "Synthetic",
		OpenRouterProviderOrder:         []string{"synthetic"},
		OpenRouterRequireParameters:     true,
		OpenRouterPricingSource:         "synthetic",
		ThinkingMode:                    "off",
		ExtraBody:                       json.RawMessage(`{}`),
		OpenRouterMaxRetries:            3,
		OpenRouterRetryBaseSeconds:      2,
		OpenRouterRetryCapSeconds:       30,
	}
}

func syntheticTensors() map[string]syntheticTensor {
	return map[string]syntheticTensor{
		"model_encoder.meta_mean": {
			shape: []uint64{1}, values: []float32{0},
		},
		"model_encoder.meta_std": {
			shape: []uint64{1}, values: []float32{1},
		},
		"model_encoder.all_metas": {
			shape: []uint64{2, 1}, values: []float32{0, 1},
		},
		"model_encoder.meta_mlp.0.weight": {
			shape: []uint64{1, 1}, values: []float32{1},
		},
		"model_encoder.meta_mlp.0.bias": {
			shape: []uint64{1}, values: []float32{0},
		},
		"model_encoder.meta_mlp.2.weight": {
			shape: []uint64{1, 1}, values: []float32{1},
		},
		"model_encoder.meta_mlp.2.bias": {
			shape: []uint64{1}, values: []float32{0},
		},
		"model_encoder.residual_embed.weight": {
			shape: []uint64{2, 1}, values: []float32{0, 0},
		},
		"model_encoder.output_proj.0.weight": {
			shape: []uint64{2, 2}, values: []float32{1, 0, 0, 1},
		},
		"model_encoder.output_proj.0.bias": {
			shape: []uint64{2}, values: []float32{0, 0},
		},
		"model_encoder.output_proj.1.weight": {
			shape: []uint64{2}, values: []float32{1, 1},
		},
		"model_encoder.output_proj.1.bias": {
			shape: []uint64{2}, values: []float32{0, 0},
		},
		"q_network.backbone.0.weight": {
			shape: []uint64{2, 8},
			values: []float32{
				0, 0, 1, 0, 0, 0, 0, 0,
				1, 0, 0, 0, 0, 0, 0, 0,
			},
		},
		"q_network.backbone.0.bias": {
			shape: []uint64{2}, values: []float32{0, 0},
		},
		"q_network.backbone.3.weight": {
			shape: []uint64{2, 2}, values: []float32{1, 0, 0, 1},
		},
		"q_network.backbone.3.bias": {
			shape: []uint64{2}, values: []float32{0, 0},
		},
		"q_network.head.weight": {
			shape: []uint64{1, 2}, values: []float32{1, 0.25},
		},
		"q_network.head.bias": {
			shape: []uint64{1}, values: []float32{0.1},
		},
	}
}

func syntheticExpectedScores() []float32 {
	armOne := float32(0.5 / math.Sqrt(0.25001))
	return []float32{0.25, float32(0.1) + armOne + float32(0.15)}
}

func encodeSafeTensors(
	t *testing.T,
	tensors map[string]syntheticTensor,
) []byte {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	descriptors := make(map[string]tensorDescriptor, len(names))
	var raw []byte
	for _, name := range names {
		value := tensors[name]
		start := uint64(len(raw))
		for _, element := range value.values {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(element))
			raw = append(raw, encoded[:]...)
		}
		descriptors[name] = tensorDescriptor{
			DType:       "F32",
			Shape:       value.shape,
			DataOffsets: []uint64{start, uint64(len(raw))},
		}
	}
	header, err := json.Marshal(descriptors)
	if err != nil {
		t.Fatal(err)
	}
	return safeTensorWithRawHeader(string(header), raw)
}

func safeTensorWithRawHeader(header string, raw []byte) []byte {
	result := make([]byte, 8, 8+len(header)+len(raw))
	binary.LittleEndian.PutUint64(result, uint64(len(header)))
	result = append(result, header...)
	result = append(result, raw...)
	return result
}

func sha256Hex(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func assertScoresClose(
	t *testing.T,
	actual []float32,
	expected []float32,
	tolerance float32,
) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("score count = %d, want %d", len(actual), len(expected))
	}
	for index := range actual {
		if abs32(actual[index]-expected[index]) > tolerance {
			t.Fatalf(
				"score[%d] = %.8f, want %.8f (tolerance %.8f)",
				index,
				actual[index],
				expected[index],
				tolerance,
			)
		}
	}
}
