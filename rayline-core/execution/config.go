package execution

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

type Config struct {
	ContractVersion           int             `json:"contract_version"`
	PackageSHA256             string          `json:"package_sha256"`
	PackagePath               string          `json:"package_path"`
	RegistryPath              string          `json:"registry_path,omitempty"`
	RegistrySHA256            string          `json:"registry_sha256"`
	AllowExperimentalControls bool            `json:"allow_experimental_controls"`
	Decisions                 DecisionsConfig `json:"decisions"`
	Deployments               []Deployment    `json:"deployments"`
	Bindings                  []ActionBinding `json:"bindings"`
	// DerivedActions replaces package_path and bindings for a package-less
	// execution_context_v2 profile: actions are the registry's admitted
	// controls for each deployment, with ARC's v5 action identity.
	DerivedActions        []DerivedActions `json:"derived_actions,omitempty"`
	TestAlias             *TestAlias       `json:"test_alias,omitempty"`
	MaxGenerationAttempts int64            `json:"max_generation_attempts,omitempty"`
	Actions               []Action         `json:"-"`
	PackageID             string           `json:"-"`
}

// DerivedActions binds every registry-admitted control of one model on one
// deployment. Bases optionally limits the controls to these native bases.
type DerivedActions struct {
	Model        string   `json:"model"`
	Worker       string   `json:"worker"`
	DeploymentID string   `json:"deployment_id"`
	Bases        []string `json:"bases,omitempty"`
}

type ActionBinding struct {
	ActionID      string `json:"action_id"`
	Worker        string `json:"worker"`
	DeploymentID  string `json:"deployment_id"`
	ReportedModel string `json:"reported_model,omitempty"`
}

type DecisionsConfig struct {
	APIMode     string          `json:"api_mode,omitempty"`
	URL         string          `json:"url"`
	CellTag     string          `json:"cell_tag,omitempty"`
	AuthEnv     string          `json:"auth_env"`
	Model       string          `json:"model"`
	Result      string          `json:"result"`
	Routing     json.RawMessage `json:"routing"`
	PriceSHA256 string          `json:"price_sha256,omitempty"`
	OfferSHA256 string          `json:"offer_sha256"`
}

const (
	DecisionsCurrent            = "current"
	DecisionsExecutionContextV1 = "execution_context_v1"
)

func (config DecisionsConfig) apiMode() string {
	if config.APIMode == "" {
		return DecisionsCurrent
	}
	return config.APIMode
}

// TestAlias is an explicit test-only substitution. The selected package action
// remains recorded separately; only the named, independently admitted local
// action executes. An omitted alias never substitutes a model.
type TestAlias struct {
	SelectedModel     string `json:"selected_model"`
	ExecutionActionID string `json:"execution_action_id"`
}

// LoadConfig refuses unknown fields and mismatched registry pins. A sidecar
// must not look ready while silently using another version's control rules.
func LoadConfig(path string) (Config, *thinkingcontrol.Registry, error) {
	var config Config
	raw, err := os.ReadFile(path)
	if err != nil {
		return config, nil, fmt.Errorf("read execution config: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		return config, nil, fmt.Errorf("decode execution config: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return config, nil, fmt.Errorf("execution config must contain exactly one JSON object")
	}
	if config.ContractVersion != ContractVersion {
		return config, nil, fmt.Errorf("unsupported execution contract version %d", config.ContractVersion)
	}
	derived := len(config.DerivedActions) > 0
	if derived {
		if config.PackagePath != "" || config.PackageSHA256 != "" || len(config.Bindings) > 0 || config.Decisions.apiMode() != DecisionsExecutionContextV2 {
			return config, nil, fmt.Errorf("derived_actions replaces package_path, package_sha256 and bindings and requires execution_context_v2")
		}
	} else if !validSHA256(config.PackageSHA256) {
		return config, nil, fmt.Errorf("package_sha256 must pin the decision package")
	}
	var registry *thinkingcontrol.Registry
	if config.RegistryPath == "" {
		registry, err = thinkingcontrol.Embedded()
	} else {
		registryPath := config.RegistryPath
		if !filepath.IsAbs(registryPath) {
			registryPath = filepath.Join(filepath.Dir(path), registryPath)
		}
		var registryBytes []byte
		registryBytes, err = os.ReadFile(registryPath)
		if err == nil {
			registry, err = thinkingcontrol.Load(registryBytes)
		}
	}
	if err != nil {
		return config, nil, fmt.Errorf("load thinking registry: %w", err)
	}
	if derived && config.RegistryPath == "" && config.RegistrySHA256 == "" {
		// The binary's own embedded registry; a derived profile names no package.
		config.RegistrySHA256 = registry.SHA256
	}
	// v2 without an explicit registry_path executes with the embedded
	// registry, whatever registry the serving package names: controls are
	// content-hashed, every bound action's control must be admitted by the
	// embedded cells (NewServer), and the selected control id is checked
	// against the bound action. registry_sha256 then pins the serving side
	// (expect, response identity); omitted, it is the package's registry.
	// An explicit registry_path, and current/v1, keep the exact pin.
	looseRegistry := config.Decisions.apiMode() == DecisionsExecutionContextV2 && config.RegistryPath == ""
	if !looseRegistry && config.RegistrySHA256 != registry.SHA256 {
		return config, nil, fmt.Errorf("thinking registry SHA256 does not match the execution config")
	}
	if derived {
		if err := config.deriveActions(registry); err != nil {
			return config, nil, err
		}
		return config, registry, nil
	}
	if err := config.loadActions(filepath.Dir(path)); err != nil {
		return config, nil, err
	}
	return config, registry, nil
}

func (config *Config) loadActions(directory string) error {
	path := config.PackagePath
	if path == "" {
		return fmt.Errorf("package_path is required")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(directory, path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read policy package: %w", err)
	}
	if fingerprint(raw) != config.PackageSHA256 {
		return fmt.Errorf("policy package SHA256 does not match the execution config")
	}
	var manifest struct {
		Schema         string   `json:"schema_version"`
		PackageID      string   `json:"package_id"`
		ControlsSHA256 string   `json:"thinking_controls_sha256"`
		Actions        []Action `json:"actions"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return fmt.Errorf("decode policy package: %w", err)
	}
	if config.RegistrySHA256 == "" && config.RegistryPath == "" && config.Decisions.apiMode() == DecisionsExecutionContextV2 {
		config.RegistrySHA256 = manifest.ControlsSHA256
	}
	if manifest.Schema != "rayline.arc-policy-package.v6" || !validSHA256(manifest.PackageID) || manifest.ControlsSHA256 != config.RegistrySHA256 {
		return fmt.Errorf("policy package version, identity or thinking registry does not match")
	}
	config.PackageID = manifest.PackageID
	actions := make(map[string]Action)
	for _, action := range manifest.Actions {
		if _, exists := actions[action.ID]; exists {
			return fmt.Errorf("duplicate action in policy package")
		}
		actions[action.ID] = action
	}
	for _, binding := range config.Bindings {
		action, ok := actions[binding.ActionID]
		if !ok {
			return fmt.Errorf("binding action is absent from pinned policy package")
		}
		action.Worker = binding.Worker
		action.DeploymentID = binding.DeploymentID
		action.ReportedModel = binding.ReportedModel
		config.Actions = append(config.Actions, action)
	}
	return nil
}

// deriveActions lists each deployment's registry-admitted controls as actions
// with ARC's v5 identity, sha256 of JCS {"control_id","model"}.
func (config *Config) deriveActions(registry *thinkingcontrol.Registry) error {
	deployments := make(map[string]Deployment, len(config.Deployments))
	for _, deployment := range config.Deployments {
		deployments[deployment.ID] = deployment
	}
	seen := map[string]bool{}
	for _, entry := range config.DerivedActions {
		deployment, ok := deployments[entry.DeploymentID]
		if !ok || entry.Model == "" || entry.Worker == "" {
			return fmt.Errorf("derived_actions entry needs model, worker and a known deployment_id")
		}
		format, err := registryFormat(deployment.Protocol)
		if err != nil {
			return err
		}
		cell, err := registry.Cell(deployment.RegistryModel, deployment.RegistryProvider, format)
		if err != nil {
			return fmt.Errorf("derived_actions %s: %w", entry.Model, err)
		}
		bases := map[string]bool{}
		for _, base := range entry.Bases {
			bases[base] = true
		}
		count := 0
		for _, controlID := range cell.ControlIDs() {
			control, ok := registry.Control(controlID)
			if !ok || (len(bases) > 0 && !bases[control.Native]) {
				continue
			}
			if _, err := registry.Admit(deployment.RegistryModel, deployment.RegistryProvider, format, control, config.AllowExperimentalControls); err != nil {
				continue
			}
			raw, err := control.Canonical()
			if err != nil {
				return err
			}
			identity, err := json.Marshal(struct {
				ControlID string `json:"control_id"`
				Model     string `json:"model"`
			}{controlID, entry.Model})
			if err != nil {
				return err
			}
			action := Action{ID: fingerprint(identity), Worker: entry.Worker, Model: entry.Model, DeploymentID: entry.DeploymentID, ControlID: controlID, Control: raw}
			if seen[action.ID] {
				return fmt.Errorf("derived_actions bind action %s twice", action.ID)
			}
			seen[action.ID] = true
			config.Actions = append(config.Actions, action)
			count++
		}
		if count == 0 {
			return fmt.Errorf("derived_actions %s: no admitted control on its deployment", entry.Model)
		}
	}
	return nil
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && hex.EncodeToString(decoded) == value
}

func validateEndpoint(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.Fragment != "" || u.RawQuery != "" {
		return fmt.Errorf("endpoint must be an absolute HTTP(S) URL without credentials, query or fragment")
	}
	return nil
}

func registryFormat(format llmprotocol.WireFormat) (string, error) {
	switch format {
	case llmprotocol.AnthropicMessagesV1:
		return thinkingcontrol.FormatMessages, nil
	case llmprotocol.OpenAIChatV1:
		return thinkingcontrol.FormatChat, nil
	case llmprotocol.OpenAIResponsesV1:
		return thinkingcontrol.FormatResponses, nil
	default:
		return "", fmt.Errorf("unsupported wire protocol %q", format)
	}
}

func routeFormat(path string) (llmprotocol.WireFormat, bool) {
	switch path {
	case "/v1/messages":
		return llmprotocol.AnthropicMessagesV1, true
	case "/v1/chat/completions":
		return llmprotocol.OpenAIChatV1, true
	case "/v1/responses":
		return llmprotocol.OpenAIResponsesV1, true
	default:
		return "", false
	}
}
