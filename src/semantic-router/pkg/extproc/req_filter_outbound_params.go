package extproc

import (
	"encoding/json"
	"strings"
)

// dispatchedOutboundParams are the request parameters the provider-bound body
// actually carries, read back off the final bytes after every mutation, so
// the routing record names what travelled rather than what was intended.
// Ids, enums, flags and numbers only: never message, tool or system content.
type dispatchedOutboundParams struct {
	Model    string
	Protocol string
	Provider string
	Backend  string
	// ProviderOnly is OpenRouter's provider.only pin, beside the order the
	// record already names.
	ProviderOnly []string

	OutputConfigEffort    string
	ReasoningObjectEffort string
	TopLevelEffort        string
	ThinkingType          string
	ThinkingBudgetTokens  *int64
	ReasoningMaxTokens    *int64

	MaxTokens   *int64
	Temperature *float64
	Stream      *bool
	ToolCount   int
	// CacheControlMarkers counts cache_control members anywhere in the body
	// (top-level automatic caching and per-block breakpoints).
	CacheControlMarkers int
	PromptCacheKey      bool
}

// readDispatchedOutboundParams reads the outbound parameters off the final
// provider body. A body that is not a JSON object yields only the dispatch's
// own facts.
func readDispatchedOutboundParams(body []byte, dispatch *providerDispatch) *dispatchedOutboundParams {
	params := &dispatchedOutboundParams{}
	if dispatch != nil {
		params.Protocol = string(dispatch.targetFormat)
		params.Backend = dispatch.backendName
		if dispatch.profile != nil {
			params.Provider = strings.ToLower(strings.TrimSpace(dispatch.profile.Type))
		}
		if providerIsOpenRouter(dispatch.profile) {
			params.Provider = "openrouter"
		}
	}
	var wire struct {
		Model        string `json:"model"`
		OutputConfig struct {
			Effort string `json:"effort"`
		} `json:"output_config"`
		Reasoning struct {
			Effort    string `json:"effort"`
			MaxTokens *int64 `json:"max_tokens"`
		} `json:"reasoning"`
		ReasoningEffort string `json:"reasoning_effort"`
		Thinking        struct {
			Type         string `json:"type"`
			BudgetTokens *int64 `json:"budget_tokens"`
		} `json:"thinking"`
		MaxTokens           *int64            `json:"max_tokens"`
		MaxCompletionTokens *int64            `json:"max_completion_tokens"`
		MaxOutputTokens     *int64            `json:"max_output_tokens"`
		Temperature         *float64          `json:"temperature"`
		Stream              *bool             `json:"stream"`
		Tools               []json.RawMessage `json:"tools"`
		PromptCacheKey      *string           `json:"prompt_cache_key"`
		Provider            struct {
			Only []string `json:"only"`
		} `json:"provider"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return params
	}
	params.Model = wire.Model
	params.ProviderOnly = wire.Provider.Only
	params.OutputConfigEffort = wire.OutputConfig.Effort
	params.ReasoningObjectEffort = wire.Reasoning.Effort
	params.ReasoningMaxTokens = wire.Reasoning.MaxTokens
	params.TopLevelEffort = wire.ReasoningEffort
	params.ThinkingType = wire.Thinking.Type
	params.ThinkingBudgetTokens = wire.Thinking.BudgetTokens
	params.MaxTokens = wire.MaxTokens
	if params.MaxTokens == nil {
		params.MaxTokens = wire.MaxCompletionTokens
	}
	if params.MaxTokens == nil {
		params.MaxTokens = wire.MaxOutputTokens
	}
	params.Temperature = wire.Temperature
	params.Stream = wire.Stream
	params.ToolCount = len(wire.Tools)
	params.PromptCacheKey = wire.PromptCacheKey != nil && *wire.PromptCacheKey != ""
	var tree interface{}
	if json.Unmarshal(body, &tree) == nil {
		params.CacheControlMarkers = countMembers(tree, "cache_control")
	}
	return params
}

// countMembers counts object members named key at any depth.
func countMembers(node interface{}, key string) int {
	count := 0
	switch typed := node.(type) {
	case map[string]interface{}:
		for name, child := range typed {
			if name == key {
				count++
			}
			count += countMembers(child, key)
		}
	case []interface{}:
		for _, child := range typed {
			count += countMembers(child, key)
		}
	}
	return count
}

// appendOutboundParamFields writes the outbound parameters on the routing
// record. A parameter the body does not carry is left off, never zeroed:
// absent max_tokens or temperature means the provider's default applied.
func appendOutboundParamFields(record map[string]interface{}, params *dispatchedOutboundParams) {
	if params == nil {
		return
	}
	record["outbound_model"] = params.Model
	record["outbound_protocol"] = params.Protocol
	record["outbound_provider"] = params.Provider
	record["outbound_backend"] = params.Backend
	if len(params.ProviderOnly) > 0 {
		record["outbound_provider_only"] = params.ProviderOnly
	}
	optionalString := map[string]string{
		"outbound_output_config_effort": params.OutputConfigEffort,
		"outbound_reasoning_effort":     params.ReasoningObjectEffort,
		"outbound_top_level_effort":     params.TopLevelEffort,
		"outbound_thinking_type":        params.ThinkingType,
	}
	for key, value := range optionalString {
		if value != "" {
			record[key] = value
		}
	}
	if params.ThinkingBudgetTokens != nil {
		record["outbound_thinking_budget_tokens"] = *params.ThinkingBudgetTokens
	}
	if params.ReasoningMaxTokens != nil {
		record["outbound_reasoning_max_tokens"] = *params.ReasoningMaxTokens
	}
	if params.MaxTokens != nil {
		record["outbound_max_tokens"] = *params.MaxTokens
	}
	if params.Temperature != nil {
		record["outbound_temperature"] = *params.Temperature
	}
	if params.Stream != nil {
		record["outbound_stream"] = *params.Stream
	}
	record["outbound_tool_count"] = params.ToolCount
	record["outbound_cache_control_markers"] = params.CacheControlMarkers
	record["outbound_prompt_cache_key"] = params.PromptCacheKey
}
