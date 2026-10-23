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
	Actions                   []Action        `json:"-"`
	PackageID                 string          `json:"-"`
}

type ActionBinding struct {
	ActionID     string `json:"action_id"`
	Worker       string `json:"worker"`
	DeploymentID string `json:"deployment_id"`
}

type DecisionsConfig struct {
	URL         string          `json:"url"`
	AuthEnv     string          `json:"auth_env"`
	Model       string          `json:"model"`
	Result      string          `json:"result"`
	Routing     json.RawMessage `json:"routing"`
	PriceSHA256 string          `json:"price_sha256,omitempty"`
	OfferSHA256 string          `json:"offer_sha256"`
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
	if !validSHA256(config.PackageSHA256) {
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
	if config.RegistrySHA256 != registry.SHA256 {
		return config, nil, fmt.Errorf("thinking registry SHA256 does not match the execution config")
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
		config.Actions = append(config.Actions, action)
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
