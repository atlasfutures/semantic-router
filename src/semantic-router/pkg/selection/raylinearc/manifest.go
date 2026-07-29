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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	maxManifestBytes = 2 << 20
	maxGoldenBytes   = 16 << 20
)

func loadManifest(runtimeDir string) (*Manifest, error) {
	path, err := artifactPath(runtimeDir, "manifest.json")
	if err != nil {
		return nil, err
	}
	data, err := readBoundedFile(path, maxManifestBytes)
	if err != nil {
		return nil, fmt.Errorf("read ARC manifest: %w", err)
	}
	var manifest Manifest
	if err := decodeStrictJSON(data, &manifest); err != nil {
		return nil, fmt.Errorf("parse ARC manifest: %w", err)
	}
	if err := manifest.validate(); err != nil {
		return nil, fmt.Errorf("validate ARC manifest: %w", err)
	}
	return &manifest, nil
}

func (manifest *Manifest) validate() error {
	checks := []func() error{
		manifest.validateEnvelope,
		manifest.validateSource,
		manifest.validateEncoder,
		manifest.validateArchitecture,
		manifest.validateWorkers,
		manifest.validatePolicy,
		manifest.validatePricingAndGolden,
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

func (manifest *Manifest) validateEnvelope() error {
	if manifest.SchemaVersion != ManifestSchema {
		return fmt.Errorf(
			"unsupported schema %q; expected %q",
			manifest.SchemaVersion,
			ManifestSchema,
		)
	}
	if strings.TrimSpace(manifest.ArtifactID) == "" {
		return errors.New("artifact_id is required")
	}
	if _, err := time.Parse(time.RFC3339, manifest.CreatedAt); err != nil {
		return errors.New("created_at must be an RFC3339 timestamp")
	}
	if strings.TrimSpace(manifest.ExporterCommit) == "" {
		return errors.New("exporter_commit is required")
	}
	if manifest.Weights.DType != "F32" {
		return fmt.Errorf("weights dtype must be F32, got %q", manifest.Weights.DType)
	}
	if err := validateFileManifest(
		manifest.Weights.File,
		manifest.Weights.SHA256,
		"weights",
	); err != nil {
		return err
	}
	return nil
}

func (manifest *Manifest) validateSource() error {
	if err := validateSHA256(manifest.Source.Checkpoint.SHA256); err != nil {
		return fmt.Errorf("source checkpoint: %w", err)
	}
	if strings.TrimSpace(manifest.Source.Checkpoint.File) == "" ||
		strings.TrimSpace(manifest.Source.Checkpoint.ModalVolume) == "" ||
		strings.TrimSpace(manifest.Source.Checkpoint.ModalPath) == "" {
		return errors.New("source checkpoint provenance is incomplete")
	}
	for label, digest := range map[string]string{
		"model meta":        manifest.Source.ModelMeta.SHA256,
		"reliability table": manifest.Source.ReliabilityTable.SHA256,
		"train report":      manifest.Source.TrainReport.SHA256,
	} {
		if err := validateSHA256(digest); err != nil {
			return fmt.Errorf("source %s: %w", label, err)
		}
	}
	return nil
}

func (manifest *Manifest) validatePricingAndGolden() error {
	if manifest.PricingSnapshot.MutableLivePricesAffectDecisions {
		return errors.New("mutable live prices must not affect ARC decisions")
	}
	if strings.TrimSpace(manifest.PricingSnapshot.ConfigPath) == "" ||
		strings.TrimSpace(manifest.PricingSnapshot.ConfigCommit) == "" {
		return errors.New("pricing snapshot path and commit are required")
	}
	if err := validateFileManifest(
		manifest.Golden.Head.File,
		manifest.Golden.Head.SHA256,
		"head golden",
	); err != nil {
		return err
	}
	if !finitePositive32(manifest.Golden.Head.ScoreTolerance) {
		return errors.New("head golden score_tolerance must be finite and positive")
	}
	if !finitePositive32(manifest.Golden.AdjustedTopTwoGapTolerance) {
		return errors.New(
			"adjusted_top_two_gap_tolerance must be finite and positive",
		)
	}
	if !finite32(manifest.Golden.RequiredSelectionParity) ||
		manifest.Golden.RequiredSelectionParity <= 0 ||
		manifest.Golden.RequiredSelectionParity > 1 {
		return errors.New("required_selection_parity must be in (0, 1]")
	}
	return nil
}

func (manifest *Manifest) validateEncoder() error {
	encoder := &manifest.Encoder
	checks := []func() error{
		func() error {
			return validateEncoderIdentity(
				encoder,
				manifest.Architecture.HistoryDimension,
			)
		},
		func() error {
			return validateEncoderLimits(encoder)
		},
		func() error {
			return validateEncoderExecution(encoder)
		},
		func() error {
			return validateEncoderGolden(encoder)
		},
		func() error {
			return validateEncoderNative(encoder)
		},
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

func validateEncoderIdentity(
	encoder *EncoderManifest,
	historyDimension int,
) error {
	if strings.TrimSpace(encoder.Model) == "" ||
		strings.TrimSpace(encoder.Revision) == "" {
		return errors.New("encoder model and revision are required")
	}
	if encoder.Dimension <= 0 || encoder.Dimension != historyDimension {
		return errors.New("encoder dimension must equal architecture history dimension")
	}
	return nil
}

func validateEncoderLimits(encoder *EncoderManifest) error {
	if encoder.MaxTokens <= 0 ||
		encoder.MinRecentTurns < 0 ||
		encoder.MinRecentTokens < 0 ||
		encoder.KVChunkTokens <= 0 ||
		encoder.KVSessionBudgetTokens <= 0 ||
		encoder.KVProcessBudgetTokens <= 0 ||
		math.IsNaN(encoder.KVIdleTTLSeconds) ||
		math.IsInf(encoder.KVIdleTTLSeconds, 0) ||
		encoder.KVIdleTTLSeconds <= 0 {
		return errors.New("encoder token limits are invalid")
	}
	return nil
}

func validateEncoderExecution(encoder *EncoderManifest) error {
	if encoder.Serialization != SerializationName {
		return fmt.Errorf(
			"encoder serialization must be %q, got %q",
			SerializationName,
			encoder.Serialization,
		)
	}
	if encoder.Pooling != "masked_mean" || !encoder.NormalizeEmbeddings {
		return errors.New("encoder must use normalized masked_mean pooling")
	}
	if strings.TrimSpace(encoder.AttentionImplementation) == "" ||
		strings.TrimSpace(encoder.DType) == "" {
		return errors.New("encoder attention implementation and dtype are required")
	}
	return nil
}

func validateEncoderGolden(encoder *EncoderManifest) error {
	if encoder.Golden != nil {
		if err := validateFileManifest(
			encoder.Golden.File,
			encoder.Golden.SHA256,
			"encoder golden",
		); err != nil {
			return err
		}
	}
	return nil
}

func validateEncoderNative(encoder *EncoderManifest) error {
	if len(bytes.TrimSpace(encoder.Native)) == 0 ||
		bytes.Equal(bytes.TrimSpace(encoder.Native), []byte("null")) {
		return errors.New("encoder native contract is required")
	}
	var native map[string]json.RawMessage
	if err := json.Unmarshal(encoder.Native, &native); err != nil || len(native) == 0 {
		return errors.New("encoder native contract must be a nonempty object")
	}
	return nil
}

func (manifest *Manifest) validateArchitecture() error {
	architecture := &manifest.Architecture
	if architecture.Name != "switch_aware" {
		return fmt.Errorf(
			"unsupported architecture %q; expected switch_aware",
			architecture.Name,
		)
	}
	if architecture.HistoryDimension <= 0 ||
		architecture.ArmEmbeddingDimension <= 0 {
		return errors.New("architecture dimensions must be positive")
	}
	expectedJoint := architecture.HistoryDimension +
		2*architecture.ArmEmbeddingDimension + 2
	if architecture.JointInputDimension != expectedJoint {
		return fmt.Errorf(
			"joint input dimension is %d; expected %d",
			architecture.JointInputDimension,
			expectedJoint,
		)
	}
	if len(architecture.HiddenDimensions) != 2 ||
		architecture.HiddenDimensions[0] <= 0 ||
		architecture.HiddenDimensions[1] <= 0 {
		return errors.New("architecture requires two positive hidden dimensions")
	}
	if math.IsNaN(architecture.Dropout) ||
		math.IsInf(architecture.Dropout, 0) ||
		architecture.Dropout < 0 ||
		architecture.Dropout >= 1 {
		return errors.New("architecture dropout must be finite and in [0, 1)")
	}
	return nil
}

func (manifest *Manifest) validateWorkers() error {
	if len(manifest.Workers) == 0 {
		return errors.New("at least one worker is required")
	}
	if len(manifest.Architecture.Pool) != len(manifest.Workers) {
		return errors.New("architecture pool length must equal worker count")
	}
	seen := make(map[string]struct{}, len(manifest.Workers))
	for index := range manifest.Workers {
		worker := &manifest.Workers[index]
		if err := validateWorker(
			worker,
			index,
			manifest.Architecture.Pool[index],
			seen,
		); err != nil {
			return err
		}
	}
	return nil
}

func validateWorker(
	worker *WorkerManifest,
	index int,
	poolID string,
	seen map[string]struct{},
) error {
	if strings.TrimSpace(worker.ID) == "" ||
		strings.TrimSpace(worker.Model) == "" {
		return fmt.Errorf("worker %d must have id and model", index)
	}
	if _, exists := seen[worker.ID]; exists {
		return fmt.Errorf("duplicate worker id %q", worker.ID)
	}
	seen[worker.ID] = struct{}{}
	if poolID != worker.ID {
		return fmt.Errorf(
			"pool[%d] %q does not match worker id %q",
			index,
			poolID,
			worker.ID,
		)
	}
	if err := validateWorkerCosts(worker); err != nil {
		return err
	}
	if err := validateWorkerDispatch(worker); err != nil {
		return err
	}
	if len(worker.CapabilityTags) == 0 {
		return fmt.Errorf("worker %q capability tags are required", worker.ID)
	}
	return nil
}

func validateWorkerCosts(worker *WorkerManifest) error {
	for label, value := range map[string]float64{
		"input":       worker.EstimatedInputCostPerToken,
		"cache_read":  worker.EstimatedCacheReadCostPerToken,
		"cache_write": worker.EstimatedCacheWriteCostPerToken,
		"output":      worker.EstimatedOutputCostPerToken,
		"latency":     worker.LatencyMS,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return fmt.Errorf(
				"worker %q %s price is invalid",
				worker.ID,
				label,
			)
		}
	}
	return nil
}

func validateWorkerDispatch(worker *WorkerManifest) error {
	checks := []func(*WorkerManifest) error{
		validateWorkerProviderContract,
		validateWorkerExecutionContract,
		validateWorkerRetryContract,
		validateWorkerExtraBody,
	}
	for _, check := range checks {
		if err := check(worker); err != nil {
			return err
		}
	}
	return nil
}

func validateWorkerProviderContract(worker *WorkerManifest) error {
	required := map[string]string{
		"api_key_env":             worker.APIKeyEnv,
		"provider slug":           worker.OpenRouterProviderSlug,
		"provider name":           worker.OpenRouterProviderName,
		"provider pricing source": worker.OpenRouterPricingSource,
	}
	for label, value := range required {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("worker %q %s is required", worker.ID, label)
		}
	}
	if len(worker.OpenRouterProviderOrder) != 1 ||
		worker.OpenRouterProviderOrder[0] != worker.OpenRouterProviderSlug {
		return fmt.Errorf(
			"worker %q provider order must contain only its provider slug",
			worker.ID,
		)
	}
	if worker.OpenRouterAllowFallbacks {
		return fmt.Errorf("worker %q must disable provider fallbacks", worker.ID)
	}
	if !worker.OpenRouterRequireParameters {
		return fmt.Errorf("worker %q must require provider parameters", worker.ID)
	}
	return nil
}

func validateWorkerExecutionContract(worker *WorkerManifest) error {
	if err := validateWorkerThinkingContract(worker); err != nil {
		return err
	}
	return validateWorkerGenerationLimits(worker)
}

func validateWorkerThinkingContract(worker *WorkerManifest) error {
	if worker.ThinkingMode != "on" && worker.ThinkingMode != "off" {
		return fmt.Errorf("worker %q thinking mode must be on or off", worker.ID)
	}
	if worker.ThinkingMode == "on" && worker.ReasoningBudgetTokens == 0 {
		return fmt.Errorf("worker %q thinking-on budget must be positive", worker.ID)
	}
	if worker.ThinkingMode == "off" && worker.ReasoningBudgetTokens != 0 {
		return fmt.Errorf("worker %q thinking-off budget must be zero", worker.ID)
	}
	return nil
}

func validateWorkerGenerationLimits(worker *WorkerManifest) error {
	if worker.MaxCompletionTokens != nil &&
		*worker.MaxCompletionTokens < worker.MinimumCompletionTokens {
		return fmt.Errorf("worker %q completion limits are incompatible", worker.ID)
	}
	if worker.Temperature != nil &&
		(math.IsNaN(*worker.Temperature) ||
			math.IsInf(*worker.Temperature, 0) ||
			*worker.Temperature < 0 ||
			*worker.Temperature > 2) {
		return fmt.Errorf("worker %q temperature is invalid", worker.ID)
	}
	return nil
}

func validateWorkerRetryContract(worker *WorkerManifest) error {
	if worker.OpenRouterMaxRetries == 0 ||
		!finitePositive64(worker.OpenRouterRetryBaseSeconds) ||
		!finitePositive64(worker.OpenRouterRetryCapSeconds) ||
		worker.OpenRouterRetryCapSeconds < worker.OpenRouterRetryBaseSeconds {
		return fmt.Errorf("worker %q retry contract is invalid", worker.ID)
	}
	if worker.AttemptDeadlineSeconds != nil &&
		!finitePositive64(*worker.AttemptDeadlineSeconds) {
		return fmt.Errorf("worker %q attempt deadline is invalid", worker.ID)
	}
	return nil
}

func validateWorkerExtraBody(worker *WorkerManifest) error {
	var extra map[string]json.RawMessage
	if err := decodeStrictJSON(worker.ExtraBody, &extra); err != nil {
		return fmt.Errorf("worker %q extra_body must be a strict object", worker.ID)
	}
	for _, reserved := range []string{
		"model",
		"provider",
		"messages",
		"tools",
		"max_completion_tokens",
		"temperature",
	} {
		if _, exists := extra[reserved]; exists {
			return fmt.Errorf(
				"worker %q extra_body cannot override %s",
				worker.ID,
				reserved,
			)
		}
	}
	if err := validateWorkerReasoningBody(worker, extra); err != nil {
		return err
	}
	if err := validateWorkerReasoningEffort(worker, extra); err != nil {
		return err
	}
	return validateWorkerExtraMaxTokens(worker, extra)
}

func validateWorkerReasoningBody(
	worker *WorkerManifest,
	extra map[string]json.RawMessage,
) error {
	rawReasoning, exists := extra["reasoning"]
	if !exists {
		return fmt.Errorf("worker %q reasoning contract is required", worker.ID)
	}
	var reasoning struct {
		Enabled   bool   `json:"enabled"`
		MaxTokens uint64 `json:"max_tokens,omitempty"`
		Effort    string `json:"effort,omitempty"`
	}
	if err := decodeStrictJSON(rawReasoning, &reasoning); err != nil {
		return fmt.Errorf("worker %q reasoning contract is invalid", worker.ID)
	}
	if worker.ThinkingMode == "on" {
		if !reasoning.Enabled ||
			reasoning.MaxTokens != worker.ReasoningBudgetTokens ||
			reasoning.Effort == "none" {
			return fmt.Errorf(
				"worker %q thinking-on reasoning contract does not match its budget",
				worker.ID,
			)
		}
	} else if reasoning.Enabled || reasoning.MaxTokens != 0 ||
		(reasoning.Effort != "" && reasoning.Effort != "none") {
		return fmt.Errorf(
			"worker %q thinking-off reasoning contract must be disabled",
			worker.ID,
		)
	}
	return nil
}

func validateWorkerReasoningEffort(
	worker *WorkerManifest,
	extra map[string]json.RawMessage,
) error {
	if rawEffort, exists := extra["reasoning_effort"]; exists {
		var effort string
		if err := json.Unmarshal(rawEffort, &effort); err != nil ||
			(worker.ThinkingMode == "off" && effort != "none") ||
			(worker.ThinkingMode == "on" && effort == "none") {
			return fmt.Errorf(
				"worker %q reasoning_effort conflicts with thinking mode",
				worker.ID,
			)
		}
	}
	return nil
}

func validateWorkerExtraMaxTokens(
	worker *WorkerManifest,
	extra map[string]json.RawMessage,
) error {
	if rawMaxTokens, exists := extra["max_tokens"]; exists {
		var maxTokens uint64
		if err := json.Unmarshal(rawMaxTokens, &maxTokens); err != nil ||
			maxTokens == 0 {
			return fmt.Errorf(
				"worker %q extra_body max_tokens is invalid",
				worker.ID,
			)
		}
	}
	return nil
}

func finitePositive64(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0
}

func (manifest *Manifest) validatePolicy() error {
	policy := &manifest.Policy
	if !finite32(policy.PreviousWorkerStayMargin) ||
		policy.PreviousWorkerStayMargin < 0 {
		return errors.New("previous worker stay margin must be finite and nonnegative")
	}
	if !finite32(policy.ColdSwitchMarginPerUSD) ||
		policy.ColdSwitchMarginPerUSD < 0 {
		return errors.New("cold switch margin per USD must be finite and nonnegative")
	}
	if !finite32(policy.ReferenceMargin) {
		return errors.New("reference margin must be finite")
	}
	for index := range manifest.Workers {
		if manifest.Workers[index].ID == policy.ReferenceWorker {
			return nil
		}
	}
	return fmt.Errorf("reference worker %q is not in the worker pool", policy.ReferenceWorker)
}

func validateFileManifest(path, hash, label string) error {
	if err := validateRelativePath(path); err != nil {
		return fmt.Errorf("%s file: %w", label, err)
	}
	if err := validateSHA256(hash); err != nil {
		return fmt.Errorf("%s: %w", label, err)
	}
	return nil
}

func validateRelativePath(path string) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("path is required")
	}
	if filepath.IsAbs(path) {
		return errors.New("absolute paths are not allowed")
	}
	clean := filepath.Clean(path)
	if clean == "." || clean == ".." ||
		strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return errors.New("path must remain inside the artifact directory")
	}
	return nil
}

func artifactPath(runtimeDir, relative string) (string, error) {
	if err := validateRelativePath(relative); err != nil {
		return "", err
	}
	root, err := filepath.Abs(runtimeDir)
	if err != nil {
		return "", fmt.Errorf("resolve artifact directory: %w", err)
	}
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return "", fmt.Errorf("resolve artifact directory symlinks: %w", err)
	}
	path := filepath.Join(root, filepath.Clean(relative))
	path, err = filepath.EvalSymlinks(path)
	if err != nil {
		return "", fmt.Errorf("resolve artifact entry symlinks: %w", err)
	}
	rel, err := filepath.Rel(root, path)
	if err != nil ||
		rel == ".." ||
		strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", errors.New("artifact path escapes the runtime directory")
	}
	return path, nil
}

func readVerifiedFile(
	path string,
	expected string,
	maxBytes int64,
) ([]byte, error) {
	if err := validateSHA256(expected); err != nil {
		return nil, err
	}
	data, err := readBoundedFile(path, maxBytes)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(data)
	actual := hex.EncodeToString(digest[:])
	if !strings.EqualFold(actual, expected) {
		// The expected value is a private artifact pin and the actual value
		// fingerprints private content; report only the bounded failure class.
		return nil, errors.New("artifact file SHA256 mismatch")
	}
	return data, nil
}

func validateSHA256(value string) error {
	if len(value) != sha256.Size*2 {
		return errors.New("SHA256 must contain 64 hexadecimal characters")
	}
	decoded, err := hex.DecodeString(value)
	if err != nil || len(decoded) != sha256.Size {
		return errors.New("SHA256 must contain 64 hexadecimal characters")
	}
	return nil
}

func readBoundedFile(path string, maxBytes int64) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("artifact entry is not a regular file")
	}
	if info.Size() > maxBytes {
		return nil, fmt.Errorf("file exceeds %d bytes", maxBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("file exceeds %d bytes", maxBytes)
	}
	return data, nil
}

func decodeStrictJSON(data []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("unexpected trailing JSON value")
		}
		return err
	}
	return nil
}

func finite32(value float32) bool {
	return !float32IsNaNOrInf(value)
}

func finitePositive32(value float32) bool {
	return finite32(value) && value > 0
}

func float32IsNaNOrInf(value float32) bool {
	return math.IsNaN(float64(value)) || math.IsInf(float64(value), 0)
}
