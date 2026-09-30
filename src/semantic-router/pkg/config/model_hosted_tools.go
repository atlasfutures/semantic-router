package config

import "fmt"

// Fork seam: model-card hosted_tools. The validator joins upstream's global
// contract list from here so validator_dispatch.go stays upstream's.
func init() {
	globalConfigContractValidators = append(globalConfigContractValidators, validateHostedToolContracts)
}

// AdmitsHostedTool reports whether this model may be sent a provider-run tool.
func (params ModelParams) AdmitsHostedTool(name string) bool {
	for _, tool := range params.HostedTools {
		if tool == name {
			return true
		}
	}
	return false
}

// knownHostedTools are the provider-run tools a model card may admit.
var knownHostedTools = map[string]bool{"web_search": true}

// validateHostedToolContracts refuses a hosted tool no router path knows, so
// a misspelt entry fails at load instead of silently admitting nothing.
func validateHostedToolContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for name, params := range cfg.ModelConfig {
		for _, tool := range params.HostedTools {
			if !knownHostedTools[tool] {
				return fmt.Errorf("model %q hosted_tools names %q; known hosted tools: web_search", name, tool)
			}
		}
	}
	return nil
}
