package config

import (
	"strings"
	"testing"
)

// hosted_tools admits a provider-run tool per model card, beside and apart
// from capabilities; absent admits none.
func TestModelCardHostedTools(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: searcher
routing:
  modelCards:
    - name: searcher
      hosted_tools: [web_search]
    - name: plain
`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !cfg.ModelConfig["searcher"].AdmitsHostedTool("web_search") {
		t.Fatal("hosted_tools: [web_search] was not admitted")
	}
	if cfg.ModelConfig["plain"].AdmitsHostedTool("web_search") {
		t.Fatal("a card without hosted_tools admitted web_search")
	}
	if cfg.ModelConfig["searcher"].SupportsCapability("web_search") {
		t.Fatal("hosted_tools leaked into capabilities")
	}
}

func TestUnknownHostedToolIsRefused(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: searcher
routing:
  modelCards:
    - name: searcher
      hosted_tools: [web_serch]
`))
	if err == nil || !strings.Contains(err.Error(), "hosted_tools") {
		t.Fatalf("a misspelt hosted tool returned %v", err)
	}
}
