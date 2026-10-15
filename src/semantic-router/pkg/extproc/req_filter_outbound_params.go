package extproc

import (
	"encoding/json"
	"sort"
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
	// ReasoningBudgetTokens is Chat's top-level reasoning_budget_tokens; the
	// two template fields are vLLM's chat_template_kwargs controls.
	ReasoningBudgetTokens  *int64
	TemplateEffort         string
	TemplateEnableThinking *bool

	MaxTokens         *int64
	Temperature       *float64
	TopP              *float64
	TopK              *int64
	Seed              *int64
	FrequencyPenalty  *float64
	PresencePenalty   *float64
	N                 *int64
	ParallelToolCalls *bool
	Stream            *bool
	ToolCount         int
	// CacheControlMarkers counts cache_control directives where the protocol
	// defines them (cacheLocations).
	CacheControlMarkers int
	// CachePolicies are the directives' distinct type/ttl/scope values.
	CachePolicies  []string
	PromptCacheKey bool
}

// readDispatchedOutboundParams reads the outbound parameters off the final
// provider body in one typed decode: content members are never materialized.
// A body that is not a JSON object yields only the dispatch's own facts.
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
		ReasoningEffort       string `json:"reasoning_effort"`
		ReasoningBudgetTokens *int64 `json:"reasoning_budget_tokens"`
		ChatTemplateKwargs    struct {
			ReasoningEffort string `json:"reasoning_effort"`
			EnableThinking  *bool  `json:"enable_thinking"`
		} `json:"chat_template_kwargs"`
		Thinking struct {
			Type         string `json:"type"`
			BudgetTokens *int64 `json:"budget_tokens"`
		} `json:"thinking"`
		MaxTokens           *int64   `json:"max_tokens"`
		MaxCompletionTokens *int64   `json:"max_completion_tokens"`
		MaxOutputTokens     *int64   `json:"max_output_tokens"`
		Temperature         *float64 `json:"temperature"`
		TopP                *float64 `json:"top_p"`
		TopK                *int64   `json:"top_k"`
		Seed                *int64   `json:"seed"`
		FrequencyPenalty    *float64 `json:"frequency_penalty"`
		PresencePenalty     *float64 `json:"presence_penalty"`
		N                   *int64   `json:"n"`
		ParallelToolCalls   *bool    `json:"parallel_tool_calls"`
		// Messages spells the same control inverted, inside tool_choice.
		ToolChoice     toolChoiceObject `json:"tool_choice"`
		Stream         *bool            `json:"stream"`
		PromptCacheKey *string          `json:"prompt_cache_key"`
		Provider       struct {
			Only []string `json:"only"`
		} `json:"provider"`
		cacheLocations
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
	params.ReasoningBudgetTokens = wire.ReasoningBudgetTokens
	params.TemplateEffort = wire.ChatTemplateKwargs.ReasoningEffort
	params.TemplateEnableThinking = wire.ChatTemplateKwargs.EnableThinking
	params.MaxTokens = wire.MaxTokens
	if params.MaxTokens == nil {
		params.MaxTokens = wire.MaxCompletionTokens
	}
	if params.MaxTokens == nil {
		params.MaxTokens = wire.MaxOutputTokens
	}
	params.Temperature = wire.Temperature
	params.TopP, params.TopK, params.Seed = wire.TopP, wire.TopK, wire.Seed
	params.FrequencyPenalty, params.PresencePenalty = wire.FrequencyPenalty, wire.PresencePenalty
	params.N, params.ParallelToolCalls = wire.N, wire.ParallelToolCalls
	if params.ParallelToolCalls == nil && wire.ToolChoice.DisableParallelToolUse != nil {
		parallel := !*wire.ToolChoice.DisableParallelToolUse
		params.ParallelToolCalls = &parallel
	}
	params.Stream = wire.Stream
	params.ToolCount = len(wire.Tools)
	params.PromptCacheKey = wire.PromptCacheKey != nil && *wire.PromptCacheKey != ""
	params.CacheControlMarkers, params.CachePolicies = wire.cacheLocations.markers()
	return params
}

// toolChoiceObject is tool_choice's object form; its scalar form ("auto",
// "none", "required") carries no parallel flag and decodes to nothing rather
// than failing the whole read.
type toolChoiceObject struct {
	DisableParallelToolUse *bool
}

func (choice *toolChoiceObject) UnmarshalJSON(raw []byte) error {
	*choice = toolChoiceObject{}
	if len(raw) == 0 || raw[0] != '{' {
		return nil
	}
	var object struct {
		DisableParallelToolUse *bool `json:"disable_parallel_tool_use"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return err
	}
	choice.DisableParallelToolUse = object.DisableParallelToolUse
	return nil
}

// cacheLocations are the places a cache_control directive is defined: the
// top level (automatic caching), each tool, each system block, and each
// message, its content blocks and their nested blocks (a tool_result's). A member of that name anywhere else (a
// tool's input schema, metadata) is data, not a directive.
type cacheLocations struct {
	CacheControl json.RawMessage `json:"cache_control"`
	Tools        []cacheMarked   `json:"tools"`
	System       cacheBlocks     `json:"system"`
	Messages     []struct {
		CacheControl json.RawMessage `json:"cache_control"`
		Content      cacheBlocks     `json:"content"`
	} `json:"messages"`
}

// cacheMarked is a tool or content block. A block's own content (a Messages
// tool_result's blocks) can carry directives too.
type cacheMarked struct {
	CacheControl json.RawMessage `json:"cache_control"`
	Content      cacheBlocks     `json:"content"`
}

// cacheBlocks is a content or system member: a block list, whose blocks may
// carry cache_control, or a string, which cannot.
type cacheBlocks []cacheMarked

func (blocks *cacheBlocks) UnmarshalJSON(raw []byte) error {
	if len(raw) == 0 || raw[0] != '[' {
		*blocks = nil
		return nil
	}
	var list []cacheMarked
	if err := json.Unmarshal(raw, &list); err != nil {
		return err
	}
	*blocks = list
	return nil
}

// markers counts the directives and names their distinct policies, each as
// type/ttl/scope with an empty member left empty (provider default).
func (locations cacheLocations) markers() (int, []string) {
	count := 0
	policies := map[string]bool{}
	marked := func(raw json.RawMessage) {
		if len(raw) == 0 || string(raw) == "null" {
			return
		}
		count++
		var directive struct {
			Type  string `json:"type"`
			TTL   string `json:"ttl"`
			Scope string `json:"scope"`
		}
		if json.Unmarshal(raw, &directive) == nil {
			policies[directive.Type+"/"+directive.TTL+"/"+directive.Scope] = true
		}
	}
	var blocks func(list cacheBlocks)
	blocks = func(list cacheBlocks) {
		for _, block := range list {
			marked(block.CacheControl)
			blocks(block.Content)
		}
	}
	marked(locations.CacheControl)
	for _, tool := range locations.Tools {
		marked(tool.CacheControl)
	}
	blocks(locations.System)
	for _, message := range locations.Messages {
		marked(message.CacheControl)
		blocks(message.Content)
	}
	names := make([]string, 0, len(policies))
	for name := range policies {
		names = append(names, name)
	}
	sort.Strings(names)
	return count, names
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
		"outbound_template_effort":      params.TemplateEffort,
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
	if params.ReasoningBudgetTokens != nil {
		record["outbound_reasoning_budget_tokens"] = *params.ReasoningBudgetTokens
	}
	if params.TemplateEnableThinking != nil {
		record["outbound_template_enable_thinking"] = *params.TemplateEnableThinking
	}
	if params.MaxTokens != nil {
		record["outbound_max_tokens"] = *params.MaxTokens
	}
	if params.Temperature != nil {
		record["outbound_temperature"] = *params.Temperature
	}
	optionalFloat := map[string]*float64{
		"outbound_top_p": params.TopP, "outbound_frequency_penalty": params.FrequencyPenalty,
		"outbound_presence_penalty": params.PresencePenalty,
	}
	for key, value := range optionalFloat {
		if value != nil {
			record[key] = *value
		}
	}
	optionalInt := map[string]*int64{"outbound_top_k": params.TopK, "outbound_seed": params.Seed, "outbound_n": params.N}
	for key, value := range optionalInt {
		if value != nil {
			record[key] = *value
		}
	}
	if params.ParallelToolCalls != nil {
		record["outbound_parallel_tool_calls"] = *params.ParallelToolCalls
	}
	if params.Stream != nil {
		record["outbound_stream"] = *params.Stream
	}
	record["outbound_tool_count"] = params.ToolCount
	record["outbound_cache_control_markers"] = params.CacheControlMarkers
	if len(params.CachePolicies) > 0 {
		record["outbound_cache_control_policies"] = params.CachePolicies
	}
	record["outbound_prompt_cache_key"] = params.PromptCacheKey
}
