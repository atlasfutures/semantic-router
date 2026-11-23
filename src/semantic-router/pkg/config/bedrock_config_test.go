package config

import (
	"strings"
	"testing"
)

// A keyless Amazon Bedrock config: a Claude card on Bedrock's Anthropic
// Messages endpoint, an open model on its OpenAI-compatible endpoint, and the
// bedrock-short-term-key authz provider in place of static keys. Each card
// keeps its provider_model_id and names the id Bedrock serves it as (an
// inference profile) in external_model_ids.
const bedrockKeylessConfig = `
version: v0.3
providers:
  defaults:
    model: claude-worker
  models:
    - name: claude-worker
      provider_model_id: anthropic/claude-x
      external_model_ids:
        bedrock: us.anthropic.claude-x
      api_format: anthropic
      backend_refs:
        - name: bedrock-anthropic
          base_url: https://bedrock-runtime.us-east-1.amazonaws.com/anthropic
          provider: bedrock
          extra_headers:
            anthropic-version: "2023-06-01"
    - name: open-worker
      provider_model_id: open/open-model
      external_model_ids:
        bedrock: us.open.open-model
      api_format: openai
      backend_refs:
        - name: bedrock-openai
          base_url: https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1
          provider: bedrock
routing:
  modelCards:
    - name: claude-worker
    - name: open-worker
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: claude-worker
global:
  services:
    authz:
      fail_open: false
      providers:
        - type: bedrock-short-term-key
`

func TestBedrockKeylessConfigLoads(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(bedrockKeylessConfig))
	if err != nil {
		t.Fatalf("Bedrock config did not load: %v", err)
	}
	if len(cfg.Authz.Providers) != 1 || cfg.Authz.Providers[0].Type != "bedrock-short-term-key" || cfg.Authz.FailOpen {
		t.Fatalf("authz = %+v", cfg.Authz)
	}
	want := map[string]struct {
		protocol         string
		model            string
		path             string
		registryProvider string
		registryModel    string
	}{
		"claude-worker": {"anthropic/messages@1", "us.anthropic.claude-x", "/anthropic/v1/messages",
			"openrouter", "anthropic/claude-x"},
		"open-worker": {"openai/chat-completions@1", "us.open.open-model", "/openai/v1/chat/completions",
			"openai", "open/open-model"},
	}
	for model, w := range want {
		address, backend, ok, err := cfg.ResolvePrimaryBackendForModel(model)
		if err != nil || !ok || address != "bedrock-runtime.us-east-1.amazonaws.com:443" {
			t.Fatalf("%s: backend = %q %q %v %v", model, address, backend, ok, err)
		}
		profile, err := cfg.GetProviderProfileForEndpoint(backend)
		if err != nil || profile == nil || profile.Type != "bedrock" {
			t.Fatalf("%s: profile = %+v, %v", model, profile, err)
		}
		upstream := cfg.ResolveExternalModelID(model, backend)
		if upstream != w.model {
			t.Fatalf("%s: upstream model = %q, want %q", model, upstream, w.model)
		}
		// Bedrock names the model in the body, not the path, and streams on
		// the same path.
		for _, stream := range []bool{false, true} {
			path, err := profile.ResolveRequestPath(w.protocol, upstream, stream)
			if err != nil || path != w.path {
				t.Fatalf("%s (stream %v): path = %q, %v; want %q", model, stream, path, err, w.path)
			}
		}
		provider, err := RaylineARCRegistryProvider(cfg, model)
		if err != nil || provider != w.registryProvider {
			t.Fatalf("%s: registry provider = %q, %v; want %q", model, provider, err, w.registryProvider)
		}
		served, err := RaylineARCRegistryModel(cfg, model)
		if err != nil || served != w.registryModel {
			t.Fatalf("%s: registry model = %q, %v; want %q", model, served, err, w.registryModel)
		}
	}
	_, backend, _, _ := cfg.ResolvePrimaryBackendForModel("claude-worker")
	profile, err := cfg.GetProviderProfileForEndpoint(backend)
	if err != nil || profile.ExtraHeaders["anthropic-version"] != "2023-06-01" {
		t.Fatalf("the Messages profile does not carry anthropic-version: %+v, %v", profile, err)
	}
}

// A Bedrock base URL serves one API (/anthropic or /openai/v1), so a Bedrock
// model accepts one format.
func TestBedrockBackendAcceptsOneFormat(t *testing.T) {
	both := strings.Replace(bedrockKeylessConfig, "      api_format: anthropic\n",
		"      api_format: anthropic\n      accepted_formats: [anthropic, openai]\n", 1)
	if both == bedrockKeylessConfig {
		t.Fatal("the fixture no longer has the Claude worker's api_format")
	}
	_, err := ParseYAMLBytes([]byte(both))
	if err == nil || !strings.Contains(err.Error(), "Amazon Bedrock, whose base URL serves one API") {
		t.Fatalf("a Bedrock model accepting two formats loaded: %v", err)
	}
}

// A Bedrock backend keeps Authorization: Bearer; an explicit default is fine.
func TestBedrockBackendKeepsItsAuthHeader(t *testing.T) {
	const anchor = "          provider: bedrock\n          extra_headers:\n"
	if !strings.Contains(bedrockKeylessConfig, anchor) {
		t.Fatal("the fixture no longer has the Claude worker's backend")
	}
	for _, override := range []string{
		"          auth_header: x-credential-carrier\n",
		"          auth_prefix: \"\"\n",
		"          auth_prefix: Token\n",
	} {
		_, err := ParseYAMLBytes([]byte(strings.Replace(bedrockKeylessConfig, anchor, "          provider: bedrock\n"+override+"          extra_headers:\n", 1)))
		if err == nil || !strings.Contains(err.Error(), "reads its key only from Authorization: Bearer") {
			t.Fatalf("a Bedrock backend with %q loaded: %v", strings.TrimSpace(override), err)
		}
	}
	explicit := "          provider: bedrock\n          auth_header: authorization\n          auth_prefix: Bearer\n          extra_headers:\n"
	if _, err := ParseYAMLBytes([]byte(strings.Replace(bedrockKeylessConfig, anchor, explicit, 1))); err != nil {
		t.Fatalf("a Bedrock backend stating the default auth was refused: %v", err)
	}
}
