package config

import (
	"strings"
	"testing"
)

// A keyless Vertex AI config: a Claude card on the Messages protocol, an open
// model on the OpenAI-compatible endpoint, and the google-access-token authz
// provider in place of static keys. Each card keeps its provider_model_id and
// names the id Vertex AI serves it as in external_model_ids.
const vertexKeylessConfig = `
version: v0.3
providers:
  defaults:
    model: claude-worker
  models:
    - name: claude-worker
      provider_model_id: anthropic/claude-x
      external_model_ids:
        vertex-ai: claude-x@v1
      api_format: anthropic
      backend_refs:
        - name: vertex
          base_url: https://aiplatform.googleapis.com/v1/projects/p/locations/global
          provider: vertex-ai
    - name: open-worker
      provider_model_id: open/open-model
      external_model_ids:
        vertex-ai: publisher/open-model
      api_format: openai
      backend_refs:
        - name: vertex-openapi
          base_url: https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi
          provider: vertex-ai
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
        - type: google-access-token
`

func TestVertexKeylessConfigLoads(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(vertexKeylessConfig))
	if err != nil {
		t.Fatalf("Vertex AI config did not load: %v", err)
	}
	if got := cfg.GetModelAPIFormat("claude-worker"); got != APIFormatAnthropic {
		t.Fatalf("claude-worker api format = %q", got)
	}
	if len(cfg.Authz.Providers) != 1 || cfg.Authz.Providers[0].Type != "google-access-token" || cfg.Authz.FailOpen {
		t.Fatalf("authz = %+v", cfg.Authz)
	}
	want := map[string]struct {
		protocol         string
		model            string
		path             string
		registryProvider string
		registryModel    string
	}{
		"claude-worker": {"anthropic/messages@1", "claude-x@v1",
			"/v1/projects/p/locations/global/publishers/anthropic/models/claude-x@v1:streamRawPredict",
			"openrouter", "anthropic/claude-x"},
		"open-worker": {"openai/chat-completions@1", "publisher/open-model",
			"/v1/projects/p/locations/global/endpoints/openapi/chat/completions",
			"openai", "open/open-model"},
	}
	for model, w := range want {
		address, backend, ok, err := cfg.ResolvePrimaryBackendForModel(model)
		if err != nil || !ok || address != "aiplatform.googleapis.com:443" {
			t.Fatalf("%s: backend = %q %q %v %v", model, address, backend, ok, err)
		}
		profile, err := cfg.GetProviderProfileForEndpoint(backend)
		if err != nil || profile == nil || profile.Type != "vertex-ai" {
			t.Fatalf("%s: profile = %+v, %v", model, profile, err)
		}
		upstream := cfg.ResolveExternalModelID(model, backend)
		if upstream != w.model {
			t.Fatalf("%s: upstream model = %q, want %q", model, upstream, w.model)
		}
		path, err := profile.ResolveRequestPath(w.protocol, upstream, true)
		if err != nil || path != w.path {
			t.Fatalf("%s: path = %q, %v; want %q", model, path, err, w.path)
		}
		// The thinking-control registry cell, named by the card's
		// provider_model_id, not by the Vertex AI id: Messages on the
		// OpenRouter Messages evidence, chat on the openai cell.
		provider, err := RaylineARCRegistryProvider(cfg, model)
		if err != nil || provider != w.registryProvider {
			t.Fatalf("%s: registry provider = %q, %v; want %q", model, provider, err, w.registryProvider)
		}
		served, err := RaylineARCRegistryModel(cfg, model)
		if err != nil || served != w.registryModel {
			t.Fatalf("%s: registry model = %q, %v; want %q", model, served, err, w.registryModel)
		}
	}
}

// A Vertex AI backend's base URL names one API, so a model on it accepts one
// format: listing both would resolve one of them under the wrong base path.
func TestVertexBackendAcceptsOneFormat(t *testing.T) {
	both := strings.Replace(vertexKeylessConfig, "      api_format: anthropic\n",
		"      api_format: anthropic\n      accepted_formats: [anthropic, openai]\n", 1)
	if both == vertexKeylessConfig {
		t.Fatal("the fixture no longer has the Claude worker's api_format")
	}
	_, err := ParseYAMLBytes([]byte(both))
	if err == nil || !strings.Contains(err.Error(), "Vertex AI, whose base URL serves one API") {
		t.Fatalf("a Vertex AI model accepting two formats loaded: %v", err)
	}
}
