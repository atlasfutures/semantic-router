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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
)

const (
	PolicyPackageSchema          = "rayline.arc-policy-package.v4"
	PolicyPackageSchemaV5        = "rayline.arc-policy-package.v5"
	PolicyDecisionRequestSchema  = "rayline.arc.policy-decision-request.v1"
	PolicyDecisionResponseSchema = "rayline.arc.policy-decision-response.v1"
	PolicyPackagesSchema         = "rayline.arc.policy-packages.v1"
)

type PolicyPackageRef struct {
	Alias         string `json:"alias"`
	PackageSHA256 string `json:"package_sha256"`
}

type PolicyAttribution struct {
	Message  int     `json:"message"`
	ActionID string  `json:"action_id"`
	ArmID    *string `json:"arm_id"`
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

// PolicyClientRequest carries the client's request as the service projects
// it with pathfinder's training intake. For anthropic_messages and
// openai_chat it is the client's system, tools and messages exactly as
// received. For openai_responses it is Input, the fully materialized item
// history (stored history resolved from previous_response_id, then this
// turn's input), and Instructions; the other fields are absent.
type PolicyClientRequest struct {
	System   json.RawMessage `json:"system"`
	Tools    json.RawMessage `json:"tools"`
	Messages json.RawMessage `json:"messages"`

	Input        []json.RawMessage `json:"-"`
	Instructions *string           `json:"-"`
}

type policyChatRequestWire struct {
	System   json.RawMessage `json:"system"`
	Tools    json.RawMessage `json:"tools"`
	Messages json.RawMessage `json:"messages"`
}

type policyResponsesRequestWire struct {
	Input        []json.RawMessage `json:"input"`
	Instructions *string           `json:"instructions"`
}

// MarshalJSON writes the shape the request format carries: input items for
// Responses, messages otherwise.
func (request PolicyClientRequest) MarshalJSON() ([]byte, error) {
	if request.Input != nil {
		return json.Marshal(policyResponsesRequestWire{Input: request.Input, Instructions: request.Instructions})
	}
	return json.Marshal(policyChatRequestWire{System: request.System, Tools: request.Tools, Messages: request.Messages})
}

// UnmarshalJSON reads either shape strictly: a request with input carries
// only input and instructions.
func (request *PolicyClientRequest) UnmarshalJSON(data []byte) error {
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(data, &keys); err != nil {
		return err
	}
	if _, responses := keys["input"]; responses {
		var wire policyResponsesRequestWire
		if err := decodeStrict(data, &wire); err != nil {
			return err
		}
		*request = PolicyClientRequest{Input: wire.Input, Instructions: wire.Instructions}
		return nil
	}
	var wire policyChatRequestWire
	if err := decodeStrict(data, &wire); err != nil {
		return err
	}
	*request = PolicyClientRequest{System: wire.System, Tools: wire.Tools, Messages: wire.Messages}
	return nil
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
	// EpisodeMode is "relaxed" for a relaxed cell: the service takes no lock
	// and keeps no session, so concurrent calls on one episode are all
	// served. Strict is the field's absence, the form every caller sent
	// before it existed (pathfinder#3068).
	EpisodeMode string `json:"episode_mode,omitempty"`
}

// PolicyEpisodeModeRelaxed is the decide request's episode_mode for a relaxed
// cell.
const PolicyEpisodeModeRelaxed = "relaxed"

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
	SelectedArmID    string                       `json:"selected_arm_id"`
	Reason           string                       `json:"reason"`
	OperatingPoint   PolicyResolvedOperatingPoint `json:"operating_point"`
}

type PolicyScoredPackage struct {
	Package             PolicyPackageRef    `json:"package"`
	PolicyID            string              `json:"policy_id"`
	Decision            PolicyDecision      `json:"decision"`
	Actions             []PolicyActionScore `json:"actions"`
	PackageVerification *string             `json:"package_verification,omitempty"`
}

// Package verification, as a policy service reports it (pathfinder#3120,
// #3138): "verified" when the package's goldens reproduced on the replica at
// load, "unverified" when it carries none. A service that predates
// verification omits the field, which means unknown.
const (
	PolicyPackageVerified   = "verified"
	PolicyPackageUnverified = "unverified"
)

func validPolicyVerification(value *string) bool {
	return value == nil || *value == PolicyPackageVerified || *value == PolicyPackageUnverified
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
	// SessionRevision is null for a relaxed call, which advances no session.
	SessionRevision *int   `json:"session_revision"`
	EngineBuildID   string `json:"engine_build_id"`
}

type PolicyTiming struct {
	Encode *float64 `json:"encode"`
	Score  *float64 `json:"score"`
	Total  float64  `json:"total"`
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
	PackageVerification      *string              `json:"package_verification,omitempty"`
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
	// Verification is the package's verification; absent means unknown.
	Verification *string `json:"verification,omitempty"`
}

type PolicyPackagesResponse struct {
	SchemaVersion    string                 `json:"schema_version"`
	EngineBuildID    string                 `json:"engine_build_id"`
	InferencePackage PolicyInferencePackage `json:"inference_package"`
	Packages         []PolicyLoadedPackage  `json:"packages"`
}

// PolicyCatalogAction is one dispatchable action: model, native effort,
// budget and steering bytes. VSR binds it to a worker; TrainedArmIDs are
// opaque to VSR.
type PolicyCatalogAction struct {
	ActionID           string   `json:"action_id"`
	Model              string   `json:"model"`
	Effort             *string  `json:"effort"`
	ReasoningMaxTokens *int     `json:"reasoning_max_tokens"`
	Level              *string  `json:"level"`
	SteeringSuffix     string   `json:"steering_suffix"`
	TrainedArmIDs      []string `json:"trained_arm_ids"`
}

// PolicyPackageManifest is a v4 manifest.
type PolicyPackageManifest struct {
	PolicyPackageCommon
	Actions []PolicyCatalogAction `json:"actions"`
}

// PolicyCatalogActionV5 is one v5 action: a model and a thinking control
// (ADR 0109). Control keeps the manifest's bytes, so its control_id is
// recomputed over exactly what the package names; ActionID and TrainedArmIDs
// are opaque to VSR.
type PolicyCatalogActionV5 struct {
	ActionID      string          `json:"action_id"`
	Model         string          `json:"model"`
	Control       json.RawMessage `json:"control"`
	ControlID     string          `json:"control_id"`
	Level         *string         `json:"level"`
	TrainedArmIDs []string        `json:"trained_arm_ids"`
}

// PolicyPackageManifestV5 is a v5 manifest. ThinkingControlsSHA256 names the
// compiled registry the actions were resolved from; it is informational.
type PolicyPackageManifestV5 struct {
	PolicyPackageCommon
	ThinkingControlsSHA256 string                  `json:"thinking_controls_sha256"`
	Actions                []PolicyCatalogActionV5 `json:"actions"`
}

// The encoding profile's optional members and the one value each may state;
// absent selects the recorded-form projection and keeps harness injections.
const (
	PolicyConversationCanonicalV1          = "canonical_v1"
	PolicyHarnessInjectionsStripClaudeCode = "strip_claude_code_2_1_v1"
)

func (common *PolicyPackageCommon) checkConversation() error {
	profile := common.EncodingProfile
	for name, member := range map[string]struct {
		raw     json.RawMessage
		allowed string
	}{
		"conversation":       {profile.Conversation, PolicyConversationCanonicalV1},
		"harness_injections": {profile.HarnessInjections, PolicyHarnessInjectionsStripClaudeCode},
	} {
		if member.raw == nil {
			continue
		}
		var value string
		if err := json.Unmarshal(member.raw, &value); err != nil || value != member.allowed {
			return fmt.Errorf("policy package encoding_profile.%s %s is not one the contract defines", name, member.raw)
		}
	}
	return nil
}

// PolicyPackageCommon is every manifest section v4 and v5 share; v5 differs
// only in its actions and thinking_controls_sha256.
type PolicyPackageCommon struct {
	SchemaVersion string `json:"schema_version"`
	PackageID     string `json:"package_id"`
	ServingPolicy struct {
		SchemaVersion string            `json:"schema_version"`
		PolicyID      string            `json:"policy_id"`
		Files         map[string]string `json:"files"`
	} `json:"serving_policy"`
	Head *struct {
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
		// Conversation and HarnessInjections name how the policy service
		// encodes the request: conversation absent is the recorded-form
		// transcript projection and canonical_v1 the bundle recipe's
		// canonical projection; harness_injections absent keeps harness
		// injections and strip_claude_code_2_1_v1 strips Claude Code's. The
		// service applies them; VSR only requires values the contract
		// defines.
		// Raw, so an explicit null is told apart from an omitted member.
		Conversation      json.RawMessage `json:"conversation,omitempty"`
		HarnessInjections json.RawMessage `json:"harness_injections,omitempty"`
		EncoderModel      string          `json:"encoder_model"`
		EncoderRevision   string          `json:"encoder_revision"`
		DType             string          `json:"dtype"`
		MaxTokens         int             `json:"max_tokens"`
		Readout           string          `json:"readout"`
		Dimension         int             `json:"dimension"`
	} `json:"encoding_profile"`
	InferencePackage PolicyInferencePackage `json:"inference_package"`
	Decision         struct {
		Mode               string               `json:"mode"`
		Selection          string               `json:"selection"`
		Horizon            string               `json:"horizon"`
		CostWeight         float64              `json:"cost_weight"`
		FallbackActionID   string               `json:"fallback_action_id"`
		ReferenceActionIDs map[string]string    `json:"reference_action_ids"`
		OperatingBounds    map[string][]float64 `json:"operating_bounds"`
		ModelSchedule      *string              `json:"model_schedule"`
	} `json:"decision"`
	Pricing struct {
		ScenarioID                string `json:"scenario_id"`
		PricingIdentity           string `json:"pricing_identity"`
		LivePricesAffectDecisions bool   `json:"live_prices_affect_decisions"`
	} `json:"pricing"`
	Calibration json.RawMessage `json:"calibration"`
	Goldens     *struct {
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
	if !validPolicyVerification(response.PackageVerification) {
		return nil, fmt.Errorf("policy package_verification %q", *response.PackageVerification)
	}
	for _, shadow := range response.Shadow {
		if shadow.Scored == nil {
			continue
		}
		if !validPolicyVerification(shadow.Scored.PackageVerification) {
			return nil, fmt.Errorf("shadow %s: package_verification %q", shadow.Scored.Package.Alias, *shadow.Scored.PackageVerification)
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
	for _, loaded := range response.Packages {
		if !validPolicyVerification(loaded.Verification) {
			return nil, fmt.Errorf("policy package %s verification %q", loaded.Alias, *loaded.Verification)
		}
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
	if err := manifest.checkConversation(); err != nil {
		return nil, err
	}
	return &manifest, nil
}

// DecodePolicyPackageManifestV5 refuses unknown fields, another schema, live
// prices, and an action whose control is not a thinking control, whose
// control_id does not recompute from it (ADR 0107 decision 8), or whose level
// is not its instruction's. Whether each control is in the loaded registry
// and admitted for its worker is the binding's check.
func DecodePolicyPackageManifestV5(body []byte) (*PolicyPackageManifestV5, error) {
	var manifest PolicyPackageManifestV5
	if err := decodeStrict(body, &manifest); err != nil {
		return nil, err
	}
	if manifest.SchemaVersion != PolicyPackageSchemaV5 {
		return nil, fmt.Errorf("policy package schema %q", manifest.SchemaVersion)
	}
	if manifest.Pricing.LivePricesAffectDecisions {
		return nil, fmt.Errorf("policy package lets live prices affect decisions")
	}
	if err := manifest.checkConversation(); err != nil {
		return nil, err
	}
	if len(manifest.Actions) == 0 {
		return nil, fmt.Errorf("policy package has no actions")
	}
	seen := make(map[string]bool, len(manifest.Actions))
	for _, action := range manifest.Actions {
		if action.ActionID == "" || seen[action.ActionID] {
			return nil, fmt.Errorf("action %q is empty or repeated", action.ActionID)
		}
		seen[action.ActionID] = true
		if action.Model == "" {
			return nil, fmt.Errorf("action %s names no model", action.ActionID)
		}
		control, err := thinkingcontrol.ParseControl(action.Control)
		if err != nil {
			return nil, fmt.Errorf("action %s: %w", action.ActionID, err)
		}
		id, err := thinkingcontrol.ControlIDOf(action.Control)
		if err != nil || id != action.ControlID || control.ID() != action.ControlID {
			return nil, fmt.Errorf("action %s: control_id %s does not recompute from its control", action.ActionID, action.ControlID)
		}
		if (control.Instruction == nil) != (action.Level == nil) ||
			(control.Instruction != nil && *action.Level != control.Instruction.Level) {
			return nil, fmt.Errorf("action %s: level is not its control's instruction level", action.ActionID)
		}
	}
	return &manifest, nil
}

// PolicyPackageSchemaOf reads a manifest's schema_version.
func PolicyPackageSchemaOf(body []byte) (string, error) {
	var head struct {
		SchemaVersion string `json:"schema_version"`
	}
	if err := json.Unmarshal(body, &head); err != nil {
		return "", err
	}
	return head.SchemaVersion, nil
}

func decodeStrict(body []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}
