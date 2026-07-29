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

// Package raylinearc implements the artifact-owned numerical runtime for
// Rayline ARC selectors.
package raylinearc

import "encoding/json"

const (
	ManifestSchema    = "rayline.mtrouter-runtime.v3"
	HeadGoldenSchema  = "rayline.mtrouter-head-golden.v1"
	SerializationName = "mtrouter-token-blocks-v2"
)

type Manifest struct {
	SchemaVersion   string               `json:"schema_version"`
	ArtifactID      string               `json:"artifact_id"`
	CreatedAt       string               `json:"created_at"`
	ExporterCommit  string               `json:"exporter_commit"`
	Weights         WeightsManifest      `json:"weights"`
	Source          SourceManifest       `json:"source"`
	Encoder         EncoderManifest      `json:"encoder"`
	Architecture    ArchitectureManifest `json:"architecture"`
	Policy          PolicyManifest       `json:"policy"`
	Workers         []WorkerManifest     `json:"workers"`
	PricingSnapshot PricingSnapshot      `json:"pricing_snapshot"`
	Golden          GoldenManifest       `json:"golden"`
}

type WeightsManifest struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
	DType  string `json:"dtype"`
}

type SourceManifest struct {
	Checkpoint       SourceCheckpoint `json:"checkpoint"`
	ModelMeta        DigestManifest   `json:"model_meta"`
	ReliabilityTable DigestManifest   `json:"reliability_table"`
	TrainReport      DigestManifest   `json:"train_report"`
}

type SourceCheckpoint struct {
	File        string `json:"file"`
	SHA256      string `json:"sha256"`
	ModalVolume string `json:"modal_volume"`
	ModalPath   string `json:"modal_path"`
}

type DigestManifest struct {
	SHA256 string `json:"sha256"`
}

type EncoderManifest struct {
	Model                   string          `json:"model"`
	Revision                string          `json:"revision"`
	Dimension               int             `json:"dimension"`
	MaxTokens               int             `json:"max_tokens"`
	MinRecentTurns          int             `json:"min_recent_turns"`
	MinRecentTokens         int             `json:"min_recent_tokens"`
	Serialization           string          `json:"serialization"`
	Pooling                 string          `json:"pooling"`
	NormalizeEmbeddings     bool            `json:"normalize_embeddings"`
	AttentionImplementation string          `json:"attention_implementation"`
	DType                   string          `json:"dtype"`
	IncrementalDefault      bool            `json:"incremental_default"`
	KVChunkTokens           int             `json:"kv_chunk_tokens"`
	KVSessionBudgetTokens   int             `json:"kv_session_budget_tokens"`
	KVProcessBudgetTokens   int             `json:"kv_process_budget_tokens"`
	KVIdleTTLSeconds        float64         `json:"kv_idle_ttl_seconds"`
	Golden                  *FileManifest   `json:"golden,omitempty"`
	Native                  json.RawMessage `json:"native"`
}

type FileManifest struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
}

type ArchitectureManifest struct {
	Name                  string   `json:"name"`
	HistoryDimension      int      `json:"history_dimension"`
	ArmEmbeddingDimension int      `json:"arm_embedding_dimension"`
	JointInputDimension   int      `json:"joint_input_dimension"`
	HiddenDimensions      []int    `json:"hidden_dimensions"`
	Dropout               float64  `json:"dropout"`
	Pool                  []string `json:"pool"`
}

type PolicyManifest struct {
	PreviousWorkerStayMargin float32 `json:"previous_worker_stay_margin"`
	ColdSwitchMarginPerUSD   float32 `json:"cold_switch_margin_per_usd"`
	ColdSwitchUpgradeExempt  bool    `json:"cold_switch_upgrade_exempt"`
	StayMarginUpgradeExempt  bool    `json:"stay_margin_upgrade_exempt"`
	ReferenceWorker          string  `json:"reference_worker"`
	ReferenceMargin          float32 `json:"reference_margin"`
}

type WorkerManifest struct {
	ID                              string          `json:"id"`
	Model                           string          `json:"model"`
	APIKeyEnv                       string          `json:"api_key_env"`
	EstimatedInputCostPerToken      float64         `json:"estimated_input_cost_per_token"`
	EstimatedCacheReadCostPerToken  float64         `json:"estimated_cache_read_cost_per_token"`
	EstimatedCacheWriteCostPerToken float64         `json:"estimated_cache_write_cost_per_token"`
	EstimatedOutputCostPerToken     float64         `json:"estimated_output_cost_per_token"`
	LatencyMS                       float64         `json:"latency_ms"`
	CapabilityTags                  []string        `json:"capability_tags"`
	OpenRouterProviderSlug          string          `json:"openrouter_provider_slug"`
	OpenRouterProviderName          string          `json:"openrouter_provider_name"`
	OpenRouterProviderOrder         []string        `json:"openrouter_provider_order"`
	OpenRouterAllowFallbacks        bool            `json:"openrouter_allow_fallbacks"`
	OpenRouterRequireParameters     bool            `json:"openrouter_require_parameters"`
	OpenRouterPricingSource         string          `json:"openrouter_pricing_source"`
	ThinkingMode                    string          `json:"thinking_mode"`
	ReasoningBudgetTokens           uint64          `json:"reasoning_budget_tokens"`
	MinimumCompletionTokens         uint64          `json:"minimum_completion_tokens"`
	MaxCompletionTokens             *uint64         `json:"max_completion_tokens"`
	Temperature                     *float64        `json:"temperature"`
	SupportsOutputEffort            bool            `json:"supports_output_effort"`
	ExtraBody                       json.RawMessage `json:"extra_body"`
	OpenRouterMaxRetries            uint64          `json:"openrouter_max_retries"`
	OpenRouterRetryBaseSeconds      float64         `json:"openrouter_retry_base_seconds"`
	OpenRouterRetryCapSeconds       float64         `json:"openrouter_retry_cap_seconds"`
	AttemptDeadlineSeconds          *float64        `json:"attempt_deadline_seconds"`
}

type PricingSnapshot struct {
	ConfigPath                       string `json:"config_path"`
	ConfigCommit                     string `json:"config_commit"`
	MutableLivePricesAffectDecisions bool   `json:"mutable_live_prices_affect_decisions"`
}

type GoldenManifest struct {
	Head                       FileGoldenManifest `json:"head"`
	AdjustedTopTwoGapTolerance float32            `json:"adjusted_top_two_gap_tolerance"`
	RequiredSelectionParity    float32            `json:"required_selection_parity"`
}

type FileGoldenManifest struct {
	File           string  `json:"file"`
	SHA256         string  `json:"sha256"`
	ScoreTolerance float32 `json:"score_tolerance"`
}

type HeadParityReport struct {
	Cases           int
	MaxScoreDrift   float32
	SelectionParity float32
	Tolerance       float32
}
