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

// Wire types for the Rayline ARC policy service. The normative contract is
// pathfinder's docs/arc_serving_contract.md; testdata/policy_service holds
// byte-identical copies of its fixtures. Nullable fields are pointers and are
// always emitted, as the Python side emits them.

import (
	"bytes"
	"encoding/json"
	"fmt"
)

const (
	PolicyPackageSchema          = "rayline.arc-policy-package.v4"
	PolicyDecisionRequestSchema  = "rayline.arc.policy-decision-request.v1"
	PolicyDecisionResponseSchema = "rayline.arc.policy-decision-response.v1"
	PolicyPackagesSchema         = "rayline.arc.policy-packages.v1"
)

type PolicyPackageRef struct {
	Alias         string `json:"alias"`
	PackageSHA256 string `json:"package_sha256"`
}

type PolicyAttribution struct {
	Message  int    `json:"message"`
	ActionID string `json:"action_id"`
}

type PolicyOperatingPoint struct {
	Name           string   `json:"name"`
	CostWeight     *float64 `json:"cost_weight"`
	Reference      *string  `json:"reference"`
	MaxQualityDrop *float64 `json:"max_quality_drop"`
	QualityMargin  *float64 `json:"quality_margin"`
}

type PolicySelection struct {
	AvailableActionIDs  []string             `json:"available_action_ids"`
	HeldActionID        *string              `json:"held_action_id"`
	Reference           *string              `json:"reference"`
	OperatingPoint      PolicyOperatingPoint `json:"operating_point"`
	PreferenceDimension string               `json:"preference_dimension"`
}

type PolicyEvaluation struct {
	Benchmark string `json:"benchmark"`
	TaskID    string `json:"task_id"`
}

// PolicyClientRequest carries the client's system, tools and messages exactly
// as received; the service projects them with pathfinder's training intake.
type PolicyClientRequest struct {
	System   json.RawMessage `json:"system"`
	Tools    json.RawMessage `json:"tools"`
	Messages json.RawMessage `json:"messages"`
}

type PolicyDecisionRequest struct {
	SchemaVersion string              `json:"schema_version"`
	Package       PolicyPackageRef    `json:"package"`
	EpisodeIDHash string              `json:"episode_id_hash"`
	ContextEpoch  string              `json:"context_epoch"`
	RequestFormat string              `json:"request_format"`
	Request       PolicyClientRequest `json:"request"`
	Attribution   []PolicyAttribution `json:"attribution"`
	Selection     PolicySelection     `json:"selection"`
	Evaluation    *PolicyEvaluation   `json:"evaluation"`
	Shadow        []PolicyPackageRef  `json:"shadow"`
}

type PolicyPredictedUsage struct {
	UncachedInput float64 `json:"uncached_input"`
	CacheRead     float64 `json:"cache_read"`
	CacheWrite    float64 `json:"cache_write"`
	Output        float64 `json:"output"`
}

type PolicyActionScore struct {
	ActionID         string                `json:"action_id"`
	Available        bool                  `json:"available"`
	Supported        bool                  `json:"supported"`
	Selected         bool                  `json:"selected"`
	Score            float64               `json:"score"`
	PredictedQuality *float64              `json:"predicted_quality"`
	PredictedUsage   *PolicyPredictedUsage `json:"predicted_usage"`
	PredictedCostUSD *float64              `json:"predicted_cost_usd"`
}

type PolicyResolvedOperatingPoint struct {
	Name       string  `json:"name"`
	PointID    string  `json:"point_id"`
	CostWeight float64 `json:"cost_weight"`
}

type PolicyDecision struct {
	SelectedActionID string                       `json:"selected_action_id"`
	Reason           string                       `json:"reason"`
	OperatingPoint   PolicyResolvedOperatingPoint `json:"operating_point"`
}

type PolicyScoredPackage struct {
	Package  PolicyPackageRef    `json:"package"`
	PolicyID string              `json:"policy_id"`
	Decision PolicyDecision      `json:"decision"`
	Actions  []PolicyActionScore `json:"actions"`
}

type PolicyShadowFailure struct {
	Package PolicyPackageRef `json:"package"`
	Error   string           `json:"error"`
}

// PolicyShadowResult is one shadow package: scored, or failed with a code.
type PolicyShadowResult struct {
	Scored  *PolicyScoredPackage
	Failure *PolicyShadowFailure
}

func (result *PolicyShadowResult) UnmarshalJSON(data []byte) error {
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(data, &keys); err != nil {
		return err
	}
	if _, failed := keys["error"]; failed {
		result.Failure = &PolicyShadowFailure{}
		return decodeStrict(data, result.Failure)
	}
	result.Scored = &PolicyScoredPackage{}
	return decodeStrict(data, result.Scored)
}

func (result PolicyShadowResult) MarshalJSON() ([]byte, error) {
	if result.Failure != nil {
		return json.Marshal(result.Failure)
	}
	return json.Marshal(result.Scored)
}

type PolicyEncoding struct {
	ProfileID        string `json:"profile_id"`
	TokenizationID   string `json:"tokenization_id"`
	RepresentationID string `json:"representation_id"`
	TokenCount       int    `json:"token_count"`
	SessionAction    string `json:"session_action"`
	SessionRevision  int    `json:"session_revision"`
	EngineBuildID    string `json:"engine_build_id"`
}

type PolicyTiming struct {
	Encode float64 `json:"encode"`
	Score  float64 `json:"score"`
	Total  float64 `json:"total"`
}

type PolicyPricing struct {
	ScenarioID      string `json:"scenario_id"`
	PricingIdentity string `json:"pricing_identity"`
}

type PolicyDecisionResponse struct {
	Package                  PolicyPackageRef     `json:"package"`
	PolicyID                 string               `json:"policy_id"`
	Decision                 PolicyDecision       `json:"decision"`
	Actions                  []PolicyActionScore  `json:"actions"`
	SchemaVersion            string               `json:"schema_version"`
	Pricing                  PolicyPricing        `json:"pricing"`
	Encoding                 PolicyEncoding       `json:"encoding"`
	TimingMillis             PolicyTiming         `json:"timing_ms"`
	Shadow                   []PolicyShadowResult `json:"shadow"`
	QualityParityEstablished bool                 `json:"quality_parity_established"`
}

type PolicyErrorResponse struct {
	Error  string         `json:"error"`
	Detail map[string]any `json:"detail"`
}

type PolicyInferencePackage struct {
	Name         string `json:"name"`
	SourceDigest string `json:"source_digest"`
}

type PolicyLoadedPackage struct {
	Alias         string `json:"alias"`
	PackageSHA256 string `json:"package_sha256"`
	PackageID     string `json:"package_id"`
	PolicyID      string `json:"policy_id"`
	ProfileID     string `json:"profile_id"`
	DecisionMode  string `json:"decision_mode"`
	State         string `json:"state"`
}

type PolicyPackagesResponse struct {
	SchemaVersion    string                 `json:"schema_version"`
	EngineBuildID    string                 `json:"engine_build_id"`
	InferencePackage PolicyInferencePackage `json:"inference_package"`
	Packages         []PolicyLoadedPackage  `json:"packages"`
}

// PolicyArm is the pathfinder arm an action was trained on. Its provider and
// route are provenance: VSR binds each action to a route of its own choosing.
type PolicyArm struct {
	Model              string  `json:"model"`
	Provider           *string `json:"provider"`
	ProviderRoute      *string `json:"provider_route"`
	Effort             *string `json:"effort"`
	Control            *string `json:"control"`
	ReasoningMaxTokens *int    `json:"reasoning_max_tokens"`
	CatalogID          *string `json:"catalog_id"`
}

type PolicyCatalogAction struct {
	ActionID   string    `json:"action_id"`
	TrainedArm PolicyArm `json:"trained_arm"`
}

type PolicyPackageManifest struct {
	SchemaVersion string `json:"schema_version"`
	PackageID     string `json:"package_id"`
	ServingPolicy struct {
		SchemaVersion string            `json:"schema_version"`
		PolicyID      string            `json:"policy_id"`
		Files         map[string]string `json:"files"`
	} `json:"serving_policy"`
	Head struct {
		File              string   `json:"file"`
		SHA256            string   `json:"sha256"`
		Format            string   `json:"format"`
		TorchVersion      string   `json:"torch_version"`
		DynamicDimensions []string `json:"dynamic_dimensions"`
	} `json:"head"`
	EncodingProfile struct {
		Serializer      string `json:"serializer"`
		Projection      string `json:"projection"`
		ToolDefinitions string `json:"tool_definitions"`
		EncoderModel    string `json:"encoder_model"`
		EncoderRevision string `json:"encoder_revision"`
		DType           string `json:"dtype"`
		MaxTokens       int    `json:"max_tokens"`
		Readout         string `json:"readout"`
		Dimension       int    `json:"dimension"`
	} `json:"encoding_profile"`
	InferencePackage PolicyInferencePackage `json:"inference_package"`
	Decision         struct {
		Mode               string               `json:"mode"`
		Selection          string               `json:"selection"`
		Horizon            string               `json:"horizon"`
		FallbackActionID   string               `json:"fallback_action_id"`
		ReferenceActionIDs map[string]string    `json:"reference_action_ids"`
		OperatingBounds    map[string][]float64 `json:"operating_bounds"`
		ModelSchedule      *string              `json:"model_schedule"`
	} `json:"decision"`
	Actions []PolicyCatalogAction `json:"actions"`
	Pricing struct {
		ScenarioID                string `json:"scenario_id"`
		PricingIdentity           string `json:"pricing_identity"`
		LivePricesAffectDecisions bool   `json:"live_prices_affect_decisions"`
	} `json:"pricing"`
	Calibration json.RawMessage `json:"calibration"`
	Goldens     struct {
		File                    string  `json:"file"`
		SHA256                  string  `json:"sha256"`
		Cases                   int     `json:"cases"`
		RequiredSelectionParity float64 `json:"required_selection_parity"`
		ScoreTolerance          float64 `json:"score_tolerance"`
	} `json:"goldens"`
	Provenance struct {
		RegistryID string `json:"registry_id"`
		HFRepo     string `json:"hf_repo"`
		HFPath     string `json:"hf_path"`
		HFRevision string `json:"hf_revision"`
		DatasetID  string `json:"dataset_id"`
		Fold       *int   `json:"fold"`
	} `json:"provenance"`
}

// DecodePolicyDecisionResponse refuses unknown fields, trailing bytes, a
// different schema, and a primary or shadow result that does not mark exactly
// its nonempty decided action.
func DecodePolicyDecisionResponse(body []byte) (*PolicyDecisionResponse, error) {
	var response PolicyDecisionResponse
	if err := decodeStrict(body, &response); err != nil {
		return nil, err
	}
	if response.SchemaVersion != PolicyDecisionResponseSchema {
		return nil, fmt.Errorf("policy decision schema %q", response.SchemaVersion)
	}
	if err := requireOneSelected(response.Decision, response.Actions); err != nil {
		return nil, err
	}
	for _, shadow := range response.Shadow {
		if shadow.Scored == nil {
			continue
		}
		if err := requireOneSelected(shadow.Scored.Decision, shadow.Scored.Actions); err != nil {
			return nil, fmt.Errorf("shadow %s: %w", shadow.Scored.Package.Alias, err)
		}
	}
	return &response, nil
}

func requireOneSelected(decision PolicyDecision, actions []PolicyActionScore) error {
	if decision.SelectedActionID == "" {
		return fmt.Errorf("policy decision names no action")
	}
	selected := 0
	for _, action := range actions {
		if action.Selected {
			selected++
			if action.ActionID != decision.SelectedActionID {
				return fmt.Errorf("selected action differs from the decision")
			}
		}
	}
	if selected != 1 {
		return fmt.Errorf("policy decision marks %d selected actions", selected)
	}
	return nil
}

// DecodePolicyPackagesResponse refuses unknown fields and another schema.
func DecodePolicyPackagesResponse(body []byte) (*PolicyPackagesResponse, error) {
	var response PolicyPackagesResponse
	if err := decodeStrict(body, &response); err != nil {
		return nil, err
	}
	if response.SchemaVersion != PolicyPackagesSchema {
		return nil, fmt.Errorf("policy packages schema %q", response.SchemaVersion)
	}
	return &response, nil
}

// DecodePolicyPackageManifest refuses unknown fields and another schema.
func DecodePolicyPackageManifest(body []byte) (*PolicyPackageManifest, error) {
	var manifest PolicyPackageManifest
	if err := decodeStrict(body, &manifest); err != nil {
		return nil, err
	}
	if manifest.SchemaVersion != PolicyPackageSchema {
		return nil, fmt.Errorf("policy package schema %q", manifest.SchemaVersion)
	}
	if manifest.Pricing.LivePricesAffectDecisions {
		return nil, fmt.Errorf("policy package lets live prices affect decisions")
	}
	return &manifest, nil
}

func decodeStrict(body []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}
