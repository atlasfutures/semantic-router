package config

import (
	"reflect"
	"strings"
	"testing"
)

func acceptedFormatsConfig(model string) string {
	return `
version: v0.3
providers:
  defaults:
    model: worker
  models:
` + model + `
routing:
  modelCards:
    - name: worker
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: worker
`
}

const openRouterAcceptedFormatsModel = `    - name: worker
      provider_model_id: anthropic/claude-opus-5
      accepted_formats: [anthropic, openai]
      backend_refs:
        - name: openrouter
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: ACCEPTED_FORMATS_TEST_KEY`

// A model's accepted formats choose the format per request: the client's own
// when listed, otherwise the first, which is also the format its backends are
// bound in.
func TestAcceptedFormatsChooseTheTargetPerRequest(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(acceptedFormatsConfig(openRouterAcceptedFormatsModel)))
	if err != nil {
		t.Fatalf("accepted_formats did not load: %v", err)
	}
	if got := cfg.GetModelAPIFormat("worker"); got != APIFormatAnthropic {
		t.Fatalf("api format = %q, want the first accepted format", got)
	}
	if got := cfg.GetModelAcceptedFormats("worker"); !reflect.DeepEqual(got, []string{APIFormatAnthropic, APIFormatOpenAI}) {
		t.Fatalf("accepted formats = %v", got)
	}
	for client, want := range map[string]string{
		APIFormatOpenAI:    APIFormatOpenAI,
		APIFormatAnthropic: APIFormatAnthropic,
		APIFormatResponses: APIFormatAnthropic,
		"":                 APIFormatAnthropic,
	} {
		if got := cfg.ResolveModelTargetAPIFormat("worker", client); got != want {
			t.Errorf("a %q client targets %q, want %q", client, got, want)
		}
	}
}

// Without accepted_formats a model keeps its single api_format for every
// client, as before.
func TestNoAcceptedFormatsKeepsTheSingleAPIFormat(t *testing.T) {
	model := strings.Replace(openRouterAcceptedFormatsModel, "accepted_formats: [anthropic, openai]", "api_format: anthropic", 1)
	cfg, err := ParseYAMLBytes([]byte(acceptedFormatsConfig(model)))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.GetModelAcceptedFormats("worker"); !reflect.DeepEqual(got, []string{APIFormatAnthropic}) {
		t.Fatalf("accepted formats = %v", got)
	}
	if got := cfg.ResolveModelTargetAPIFormat("worker", APIFormatOpenAI); got != APIFormatAnthropic {
		t.Fatalf("a Chat client targets %q, want anthropic", got)
	}
	if params := cfg.ModelConfig["worker"]; params.AcceptedFormats != nil {
		t.Fatalf("accepted formats were materialized: %v", params.AcceptedFormats)
	}
}

func TestAcceptedFormatsRefusals(t *testing.T) {
	for name, test := range map[string]struct{ from, to, want string }{
		"an unknown format": {"[anthropic, openai]", "[anthropic, gemini]", `"gemini" is unsupported`},
		"a repeated format": {"[anthropic, openai]", "[anthropic, anthropic]", `lists "anthropic" twice`},
		"an api_format that is not first": {
			"accepted_formats: [anthropic, openai]",
			"api_format: openai\n      accepted_formats: [anthropic, openai]", "must be the first of accepted_formats",
		},
		// Anthropic serves only Messages, so it cannot take a Chat request.
		"a backend that does not serve a format": {
			"base_url: https://openrouter.ai/api/v1\n          provider: openrouter",
			"base_url: https://api.anthropic.com\n          provider: anthropic", "does not serve openai/chat-completions@1",
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := strings.Replace(openRouterAcceptedFormatsModel, test.from, test.to, 1)
			if model == openRouterAcceptedFormatsModel {
				t.Fatal("the fixture did not change")
			}
			_, err := ParseYAMLBytes([]byte(acceptedFormatsConfig(model)))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

// A backend with a catalog mapping must list the format there: OpenRouter's
// Kimi mapping lists only Chat, so a Messages request has no native model.
func TestAcceptedFormatsNeedTheCatalogMapping(t *testing.T) {
	model := `    - name: worker
      catalog: moonshot/kimi-k2.7-code
      accepted_formats: [openai, anthropic]
      backend_refs:
        - name: openrouter
          provider: openrouter
          api_key_env: ACCEPTED_FORMATS_TEST_KEY`
	config := strings.Replace(acceptedFormatsConfig(model), "  modelCards:\n    - name: worker\n", "", 1)
	_, err := ParseYAMLBytes([]byte(config))
	if err == nil || !strings.Contains(err.Error(), "catalog mapping of \"moonshotai/kimi-k2.7-code\" does not list anthropic/messages@1") {
		t.Fatalf("error = %v, want the catalog mapping refusal", err)
	}
}

// The format-bound ARC checks hold for every format a worker may be sent in:
// a base on OpenRouter's Chat reasoning object is refused on a worker that
// also accepts Messages, since a Messages client's turn would drop it.
func TestRaylineARCWorkerThinkingNeedsChatForEveryAcceptedFormat(t *testing.T) {
	cfg := &RouterConfig{BackendModels: BackendModels{
		ModelConfig: map[string]ModelParams{"worker": {
			PreferredEndpoints: []string{"backend"}, APIFormat: APIFormatOpenAI,
			AcceptedFormats: []string{APIFormatOpenAI, APIFormatAnthropic},
		}},
		VLLMEndpoints:    []VLLMEndpoint{{Name: "backend", Address: "provider", Port: 443, ProviderProfileName: "profile"}},
		ProviderProfiles: map[string]ProviderProfile{"profile": {Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}},
	}}
	arc := &RaylineARCAlgorithmConfig{WorkerThinking: map[string]RaylineARCWorkerThinkingConfig{
		"worker": {Level: "high", Wire: RaylineARCWorkerThinkingEffort, Effort: "high"},
	}}
	if err := validateRaylineARCWorkerThinkingTransports(cfg, arc); err == nil || !strings.Contains(err.Error(), "openai, anthropic") {
		t.Fatalf("error = %v, want a refusal naming both formats", err)
	}
	params := cfg.ModelConfig["worker"]
	params.AcceptedFormats = []string{APIFormatOpenAI}
	cfg.ModelConfig["worker"] = params
	if err := validateRaylineARCWorkerThinkingTransports(cfg, arc); err != nil {
		t.Fatalf("a Chat-only worker refused: %v", err)
	}
}
