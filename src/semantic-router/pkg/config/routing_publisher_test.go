package config

import "testing"

// A card's publisher survives a routing-only fragment, as the DSL compiles
// and decompiles it: the tool-loop family hold reads it, and without it a
// deployment-namespaced id (local/..., accounts/...) is all that is left.
func TestRoutingFragmentKeepsTheCardPublisher(t *testing.T) {
	cfg, err := ParseRoutingYAMLBytes([]byte(`
routing:
  modelCards:
    - name: local/qwen3
      publisher: Alibaba
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.ModelConfig["local/qwen3"].Publisher; got != "Alibaba" {
		t.Fatalf("parsed publisher = %q, want Alibaba", got)
	}
	exported := routingModelsFromRuntimeConfig(cfg)
	if len(exported) != 1 || exported[0].Publisher != "Alibaba" {
		t.Fatalf("exported models = %+v, want publisher Alibaba", exported)
	}
}
