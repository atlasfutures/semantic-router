//go:build !windows && cgo

package extproc

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A readiness-passing ARC artifact, generated rather than committed.
//
// The router refuses to arm ARC without a real artifact on disk, so a test of
// the startup path needs one. Every byte here is computed: the weights are
// sparse and chosen so the head's forward pass has a closed form, which is
// what lets the golden scores be written down instead of measured, and no
// opaque binary or private weight is needed.
//
// Shapes follow the generator the hermetic acceptance stack already uses, with
// the two free dimensions shrunk to the smallest the closed form allows.
const (
	fixtureArtifactID = "public-arc-port-fixture-v1"
	// The readiness contract pins the encoder identity, so these three are not
	// free: see raylineARCEncoderContractSatisfied.
	fixtureEncoderModel    = "Qwen/Qwen3.5-0.8B"
	fixtureEncoderRevision = "2fc06364715b967f1860aea9cf38778875588b17"
	fixtureSerializer      = "mtrouter-token-blocks-v2"

	// history is pinned to 1024 by the readiness contract; the rest are free.
	fixtureHistoryDimension = 1024
	fixtureArmDimension     = 8
	fixtureHiddenDimension  = 8
	fixtureMetaDimension    = 8
	fixtureMetaHidden       = 32
	fixtureResidualDim      = 16
	// joint = history + candidate arm + previous arm + same-arm flag + turn.
	fixtureJointDimension = fixtureHistoryDimension + 2*fixtureArmDimension + 2
	fixtureRoutingAxis    = 252

	fixtureAPIKeyEnv = "ARC_FIXTURE_API_KEY"

	// The encoder response contract the client checks byte for byte.
	fixtureTokenizerSHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
	fixtureEOSTokenID      = 248046
	fixtureEngineBuildID   = "vllm@public-arc-port-fixture-build"
	fixtureIOPluginVersion = "rayline-arc-io@0.1.0"

	fixtureWorkerA = "worker-a"
	fixtureWorkerB = "worker-b"
)

type fixtureTensor struct {
	shape  []int
	fill   float32
	sparse map[int]float32
}

// writeARCArtifact writes the three files LoadRuntime reads — manifest.json,
// head.safetensors and head_golden.json — into a fresh directory and returns
// it. The two SHA256 digests in the manifest are back-filled from the bytes
// actually written, so the fixture cannot drift out of agreement with itself.
func writeARCArtifact(t *testing.T, providerBaseURL string) string {
	t.Helper()
	dir := t.TempDir()

	weights := encodeFixtureSafeTensors(t, fixtureTensors())
	if err := os.WriteFile(filepath.Join(dir, "head.safetensors"), weights, 0o600); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	checkpointDigest := sha256Hex([]byte("public arc port fixture checkpoint"))
	golden, err := json.MarshalIndent(fixtureGolden(checkpointDigest), "", "  ")
	if err != nil {
		t.Fatalf("encode golden: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "head_golden.json"), golden, 0o600); err != nil {
		t.Fatalf("write golden: %v", err)
	}

	manifest := fixtureManifest(checkpointDigest, providerBaseURL)
	manifest.Weights.SHA256 = sha256Hex(weights)
	manifest.Golden.Head.SHA256 = sha256Hex(golden)
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatalf("encode manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), manifestBytes, 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return dir
}

// fixtureArmAxis is the value one arm embedding lands on after layer
// normalization, given a pre-norm vector of one 1 and the rest zeros. The
// golden scores are written in terms of it.
func fixtureArmAxis() float64 {
	mean := 1.0 / float64(fixtureArmDimension)
	variance := ((1-mean)*(1-mean) + float64(fixtureArmDimension-1)*mean*mean) /
		float64(fixtureArmDimension)
	return (1 - mean) / math.Sqrt(variance+1e-5)
}

// The head reduces to score = |history[routingAxis] + candidateArmAxis|, so a
// unit embedding on the routing axis separates the two arms by exactly 2.
func fixtureTensors() map[string]fixtureTensor {
	candidateOffset := fixtureHistoryDimension
	return map[string]fixtureTensor{
		"model_encoder.meta_mean":         {shape: []int{fixtureMetaDimension}},
		"model_encoder.meta_std":          {shape: []int{fixtureMetaDimension}, fill: 1},
		"model_encoder.all_metas":         {shape: []int{2, fixtureMetaDimension}},
		"model_encoder.meta_mlp.0.weight": {shape: []int{fixtureMetaHidden, fixtureMetaDimension}},
		"model_encoder.meta_mlp.0.bias":   {shape: []int{fixtureMetaHidden}},
		"model_encoder.meta_mlp.2.weight": {shape: []int{fixtureMetaHidden, fixtureMetaHidden}},
		"model_encoder.meta_mlp.2.bias":   {shape: []int{fixtureMetaHidden}},
		// Arm A gets +1 on residual axis 0, arm B gets -1.
		"model_encoder.residual_embed.weight": {
			shape:  []int{2, fixtureResidualDim},
			sparse: map[int]float32{0: 1, fixtureResidualDim: -1},
		},
		// Row 0 reads the first residual axis, which sits after the meta half.
		"model_encoder.output_proj.0.weight": {
			shape:  []int{fixtureArmDimension, fixtureMetaHidden + fixtureResidualDim},
			sparse: map[int]float32{fixtureMetaHidden: 1},
		},
		"model_encoder.output_proj.0.bias": {shape: []int{fixtureArmDimension}},
		"model_encoder.output_proj.1.weight": {
			shape: []int{fixtureArmDimension}, fill: 1,
		},
		"model_encoder.output_proj.1.bias": {shape: []int{fixtureArmDimension}},
		// Row 0 sums the routing axis and the candidate arm axis; row 1 negates
		// the same sum, so the two ReLU rows recover its absolute value.
		"q_network.backbone.0.weight": {
			shape: []int{fixtureHiddenDimension, fixtureJointDimension},
			sparse: map[int]float32{
				fixtureRoutingAxis: 1,
				candidateOffset:    1,
				fixtureJointDimension + fixtureRoutingAxis: -1,
				fixtureJointDimension + candidateOffset:    -1,
			},
		},
		"q_network.backbone.0.bias": {shape: []int{fixtureHiddenDimension}},
		"q_network.backbone.3.weight": {
			shape:  []int{fixtureHiddenDimension, fixtureHiddenDimension},
			sparse: map[int]float32{0: 1, fixtureHiddenDimension + 1: 1},
		},
		"q_network.backbone.3.bias": {shape: []int{fixtureHiddenDimension}},
		"q_network.head.weight": {
			shape:  []int{1, fixtureHiddenDimension},
			sparse: map[int]float32{0: 1, 1: 1},
		},
		"q_network.head.bias": {shape: []int{1}},
	}
}

type fixtureGoldenCase struct {
	ID                 string    `json:"id"`
	Embedding          []float32 `json:"embedding"`
	PreviousModelIndex int64     `json:"previous_model_index"`
	RouteCallIndex     uint64    `json:"route_call_index"`
	Scores             []float32 `json:"scores"`
	SelectedIndex      int       `json:"selected_index"`
}

type fixtureGoldenFile struct {
	SchemaVersion    string              `json:"schema_version"`
	CheckpointSHA256 string              `json:"checkpoint_sha256"`
	ScoreTolerance   float32             `json:"score_tolerance"`
	Pool             []string            `json:"pool"`
	Cases            []fixtureGoldenCase `json:"cases"`
}

func fixtureGolden(checkpointDigest string) fixtureGoldenFile {
	arm := float32(fixtureArmAxis())
	return fixtureGoldenFile{
		SchemaVersion:    "rayline.mtrouter-head-golden.v1",
		CheckpointSHA256: checkpointDigest,
		ScoreTolerance:   0.001,
		Pool:             []string{fixtureWorkerA, fixtureWorkerB},
		Cases: []fixtureGoldenCase{
			{
				ID:                 "route-a",
				Embedding:          fixtureEmbedding(1),
				PreviousModelIndex: -1,
				RouteCallIndex:     0,
				Scores:             []float32{arm + 1, arm - 1},
				SelectedIndex:      0,
			},
			{
				ID:                 "route-b",
				Embedding:          fixtureEmbedding(-1),
				PreviousModelIndex: -1,
				RouteCallIndex:     0,
				Scores:             []float32{arm - 1, arm + 1},
				SelectedIndex:      1,
			},
		},
	}
}

func fixtureEmbedding(sign float32) []float32 {
	embedding := make([]float32, fixtureHistoryDimension)
	embedding[fixtureRoutingAxis] = sign
	return embedding
}

func encodeFixtureSafeTensors(t *testing.T, tensors map[string]fixtureTensor) []byte {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	type descriptor struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	descriptors := make(map[string]descriptor, len(names))
	var raw []byte
	for _, name := range names {
		tensor := tensors[name]
		start := len(raw)
		raw = append(raw, encodeFixtureTensor(tensor)...)
		descriptors[name] = descriptor{
			DType: "F32", Shape: tensor.shape, DataOffsets: []int{start, len(raw)},
		}
	}
	header, err := json.Marshal(descriptors)
	if err != nil {
		t.Fatalf("encode safetensors header: %v", err)
	}
	out := make([]byte, 8, 8+len(header)+len(raw))
	binary.LittleEndian.PutUint64(out, uint64(len(header)))
	out = append(out, header...)
	return append(out, raw...)
}

func encodeFixtureTensor(tensor fixtureTensor) []byte {
	count := 1
	for _, dimension := range tensor.shape {
		count *= dimension
	}
	values := make([]float32, count)
	if tensor.fill != 0 {
		for index := range values {
			values[index] = tensor.fill
		}
	}
	for index, value := range tensor.sparse {
		values[index] = value
	}
	raw := make([]byte, 4*count)
	for index, value := range values {
		binary.LittleEndian.PutUint32(raw[4*index:], math.Float32bits(value))
	}
	return raw
}

func sha256Hex(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func fixtureManifest(checkpointDigest string, providerBaseURL string) raylinearc.Manifest {
	return raylinearc.Manifest{
		SchemaVersion:  raylinearc.ManifestSchema,
		ArtifactID:     fixtureArtifactID,
		CreatedAt:      "2026-01-01T00:00:00Z",
		ExporterCommit: "public-arc-port-fixture-exporter",
		Weights: raylinearc.WeightsManifest{
			File: "head.safetensors", DType: "F32",
		},
		Source: raylinearc.SourceManifest{
			Checkpoint: raylinearc.SourceCheckpoint{
				File:        "synthetic-checkpoint.pt",
				SHA256:      checkpointDigest,
				ModalVolume: "public-synthetic",
				ModalPath:   "/public/synthetic-checkpoint.pt",
			},
			ModelMeta:        raylinearc.DigestManifest{SHA256: repeatHex("1")},
			ReliabilityTable: raylinearc.DigestManifest{SHA256: repeatHex("2")},
			TrainReport:      raylinearc.DigestManifest{SHA256: repeatHex("3")},
		},
		Encoder: raylinearc.EncoderManifest{
			Model:                   fixtureEncoderModel,
			Revision:                fixtureEncoderRevision,
			Dimension:               fixtureHistoryDimension,
			MaxTokens:               262144,
			MinRecentTurns:          1,
			MinRecentTokens:         64,
			Serialization:           fixtureSerializer,
			Pooling:                 "masked_mean",
			NormalizeEmbeddings:     true,
			AttentionImplementation: "sdpa",
			DType:                   "BF16",
			IncrementalDefault:      true,
			KVChunkTokens:           8192,
			KVSessionBudgetTokens:   262144,
			KVProcessBudgetTokens:   524288,
			KVIdleTTLSeconds:        900,
			Native:                  json.RawMessage(`{"runtime":"public-arc-port-fixture"}`),
		},
		Architecture: raylinearc.ArchitectureManifest{
			Name:                  "switch_aware",
			HistoryDimension:      fixtureHistoryDimension,
			ArmEmbeddingDimension: fixtureArmDimension,
			JointInputDimension:   fixtureJointDimension,
			HiddenDimensions:      []int{fixtureHiddenDimension, fixtureHiddenDimension},
			Dropout:               0.1,
			Pool:                  []string{fixtureWorkerA, fixtureWorkerB},
		},
		Policy: raylinearc.PolicyManifest{
			PreviousWorkerStayMargin: 0.05,
			ColdSwitchMarginPerUSD:   1,
			ColdSwitchUpgradeExempt:  true,
			ReferenceWorker:          fixtureWorkerA,
		},
		Workers: []raylinearc.WorkerManifest{
			fixtureWorker(fixtureWorkerA, providerBaseURL, false),
			fixtureWorker(fixtureWorkerB, providerBaseURL, true),
		},
		PricingSnapshot: raylinearc.PricingSnapshot{
			ConfigPath:   "public/synthetic-pricing.yaml",
			ConfigCommit: "public-arc-port-fixture-pricing",
		},
		Golden: raylinearc.GoldenManifest{
			Head: raylinearc.FileGoldenManifest{
				File: "head_golden.json", ScoreTolerance: 0.001,
			},
			AdjustedTopTwoGapTolerance: 0.001,
			RequiredSelectionParity:    1,
		},
	}
}

func repeatHex(digit string) string {
	out := make([]byte, 0, 64)
	for len(out) < 64 {
		out = append(out, digit[0])
	}
	return string(out)
}

// Per-token costs are round numbers so the per-1M prices the config repeats
// are exact: the readiness contract compares the two and rejects any drift.
func fixtureWorker(id string, providerBaseURL string, thinking bool) raylinearc.WorkerManifest {
	inputCost := 0.000001
	cacheReadCost := 0.0000005
	cacheWriteCost := 0.0000015
	outputCost := 0.000002
	thinkingMode := "off"
	reasoningBudget := uint64(0)
	temperature := 0.2
	if thinking {
		inputCost, cacheReadCost = 0.000002, 0.000001
		cacheWriteCost, outputCost = 0.0000025, 0.000004
		thinkingMode = "on"
		reasoningBudget = 64
		temperature = 0.3
	}
	maxCompletionTokens := uint64(128)
	attemptDeadlineSeconds := 30.0
	extraBody := `{"chat_template_kwargs":{"enable_thinking":false}}`
	if thinking {
		extraBody = `{"chat_template_kwargs":{"enable_thinking":true}}`
	}
	return raylinearc.WorkerManifest{
		ID:                              id,
		Model:                           "synthetic/" + id,
		APIKeyEnv:                       fixtureAPIKeyEnv,
		DispatchBackend:                 raylinearc.DispatchOpenAICompat,
		ProviderBaseURL:                 providerBaseURL,
		EstimatedInputCostPerToken:      inputCost,
		EstimatedCacheReadCostPerToken:  cacheReadCost,
		EstimatedCacheWriteCostPerToken: cacheWriteCost,
		EstimatedOutputCostPerToken:     outputCost,
		LatencyMS:                       10,
		CapabilityTags:                  []string{"public-synthetic"},
		ThinkingMode:                    thinkingMode,
		ReasoningBudgetTokens:           reasoningBudget,
		MinimumCompletionTokens:         32,
		MaxCompletionTokens:             &maxCompletionTokens,
		Temperature:                     &temperature,
		ExtraBody:                       json.RawMessage(extraBody),
		OpenRouterMaxRetries:            1,
		OpenRouterRetryBaseSeconds:      2,
		OpenRouterRetryCapSeconds:       30,
		AttemptDeadlineSeconds:          &attemptDeadlineSeconds,
	}
}
