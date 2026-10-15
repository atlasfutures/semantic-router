package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	// ProviderPreferences is the whole provider object, compact: every
	// routing constraint (allow_fallbacks, ignore, require_parameters,
	// data_collection, max_price, ...) that decides the eligible endpoints.
	ProviderPreferences string
	ToolChoice          string
	ToolChoiceName      string
	// StopSequences is how many stop sequences were sent, never their text.
	StopSequences *int64
	// OutputFormat is the structured-output type (json_schema, json_object,
	// text) and OutputFormatStrict its strict flag; the schema is not read.
	OutputFormat       string
	OutputFormatStrict *bool
	// PromptCacheRetention is the retention enum, "other" for any value the
	// router does not know.
	PromptCacheRetention string
	// ReasoningSummary is Responses' reasoning.summary enum, "other" for any
	// value the router does not know.
	ReasoningSummary string
	// TaskBudget is Messages' output_config.task_budget when it is a number.
	TaskBudget *int64
	// PreviousResponseID and Conversation say whether a Responses body
	// references provider-held context; the ids themselves are not recorded.
	PreviousResponseID bool
	Conversation       bool
	// Store and AutoStore are Chat/Responses' provider-side storage flags.
	Store     *bool
	AutoStore *bool
	// Truncation is Responses' context truncation mode.
	Truncation string
	// StreamIncludeUsage and StreamIncludeObfuscation are stream_options.
	StreamIncludeUsage       *bool
	StreamIncludeObfuscation *bool
	// StrictToolCount is how many tools ask for strict schema adherence
	// (tool.strict, or Chat's tool.function.strict).
	StrictToolCount int

	OutputConfigEffort    string
	ReasoningObjectEffort string
	TopLevelEffort        string
	ThinkingType          string
	ThinkingDisplay       string
	ThinkingBudgetTokens  *int64
	ReasoningMaxTokens    *int64
	// ReasoningBudgetTokens is Chat's top-level reasoning_budget_tokens; the
	// two template fields are vLLM's chat_template_kwargs controls.
	ReasoningBudgetTokens  *int64
	TemplateEffort         string
	TemplateEnableThinking *bool
	// TemplateSwitches is every scalar chat_template_kwargs member, compact
	// JSON: catalogs spell reasoning switches differently (thinking,
	// low_effort, ...). See templateKwargs for what is kept.
	TemplateSwitches string

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
			Effort     string             `json:"effort"`
			Format     outputFormatObject `json:"format"`
			TaskBudget json.RawMessage    `json:"task_budget"`
		} `json:"output_config"`
		ResponseFormat struct {
			outputFormatObject
			JSONSchema struct {
				Strict *bool `json:"strict"`
			} `json:"json_schema"`
		} `json:"response_format"`
		Text struct {
			Format outputFormatObject `json:"format"`
		} `json:"text"`
		PromptCacheRetention *string         `json:"prompt_cache_retention"`
		PreviousResponseID   *string         `json:"previous_response_id"`
		Conversation         json.RawMessage `json:"conversation"`
		Store                *bool           `json:"store"`
		Truncation           string          `json:"truncation"`
		StreamOptions        struct {
			IncludeUsage       *bool `json:"include_usage"`
			IncludeObfuscation *bool `json:"include_obfuscation"`
		} `json:"stream_options"`
		AutoStore *bool `json:"auto_store"`
		Reasoning struct {
			Effort    string          `json:"effort"`
			MaxTokens *int64          `json:"max_tokens"`
			Summary   json.RawMessage `json:"summary"`
		} `json:"reasoning"`
		ReasoningEffort       string         `json:"reasoning_effort"`
		ReasoningBudgetTokens *int64         `json:"reasoning_budget_tokens"`
		ChatTemplateKwargs    templateKwargs `json:"chat_template_kwargs"`
		Thinking              struct {
			Type         string `json:"type"`
			BudgetTokens *int64 `json:"budget_tokens"`
			Display      string `json:"display"`
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
		Stop           *stopSequences   `json:"stop"`
		StopSequences  *stopSequences   `json:"stop_sequences"`
		// Provider is OpenRouter's routing object: configuration (pins,
		// fallbacks, price caps, data policy), never content.
		Provider json.RawMessage `json:"provider"`
		cacheLocations
	}
	if json.Unmarshal(body, &wire) != nil {
		return params
	}
	params.Model = wire.Model
	if len(wire.Provider) > 0 && wire.Provider[0] == '{' {
		// Only the routing fields the router knows are recorded: a client's
		// own provider object reaches the wire unmodeled on a same-format
		// dispatch, and any other member it carries stays off the record.
		var provider config.OpenRouterProviderPreferences
		if json.Unmarshal(wire.Provider, &provider) == nil {
			redactProviderPreferences(&provider)
			params.ProviderOnly = provider.Only
			if known, err := json.Marshal(provider); err == nil && string(known) != "{}" {
				params.ProviderPreferences = string(known)
			}
		}
	}
	params.OutputFormat, params.OutputFormatStrict = wire.OutputConfig.Format.Type, wire.OutputConfig.Format.Strict
	if params.OutputFormat == "" {
		params.OutputFormat, params.OutputFormatStrict = wire.ResponseFormat.Type, wire.ResponseFormat.JSONSchema.Strict
	}
	if params.OutputFormat == "" {
		params.OutputFormat, params.OutputFormatStrict = wire.Text.Format.Type, wire.Text.Format.Strict
	}
	if wire.PromptCacheRetention != nil {
		params.PromptCacheRetention = "other"
		if known := *wire.PromptCacheRetention; known == "in_memory" || known == "24h" {
			params.PromptCacheRetention = known
		}
	}
	if len(wire.Reasoning.Summary) > 0 && string(wire.Reasoning.Summary) != "null" {
		params.ReasoningSummary = "other"
		var summary string
		if json.Unmarshal(wire.Reasoning.Summary, &summary) == nil &&
			(summary == "auto" || summary == "concise" || summary == "detailed" || summary == "none") {
			params.ReasoningSummary = summary
		}
	}
	params.PreviousResponseID = wire.PreviousResponseID != nil && *wire.PreviousResponseID != ""
	params.Conversation = len(wire.Conversation) > 0 && string(wire.Conversation) != "null" && string(wire.Conversation) != `""`
	params.Store, params.AutoStore = wire.Store, wire.AutoStore
	params.Truncation = knownOrDigest(wire.Truncation, knownTruncations)
	params.StreamIncludeUsage = wire.StreamOptions.IncludeUsage
	params.StreamIncludeObfuscation = wire.StreamOptions.IncludeObfuscation
	for _, tool := range wire.Tools {
		if string(tool.Strict) == "true" || string(tool.Function.Strict) == "true" {
			params.StrictToolCount++
		}
	}
	var taskBudget int64
	if json.Unmarshal(wire.OutputConfig.TaskBudget, &taskBudget) == nil && string(wire.OutputConfig.TaskBudget) != "null" {
		params.TaskBudget = &taskBudget
	}
	// A forced tool's name is the client's declaration: recorded only as a
	// digest, so turns forcing the same tool still correlate.
	params.ToolChoice = knownOrDigest(wire.ToolChoice.Mode, knownToolChoices)
	if wire.ToolChoice.Name != "" {
		params.ToolChoiceName = outboundDigest(wire.ToolChoice.Name)
	}
	switch {
	case wire.Stop != nil:
		params.StopSequences = llmprotocol.Int64(int64(*wire.Stop))
	case wire.StopSequences != nil:
		params.StopSequences = llmprotocol.Int64(int64(*wire.StopSequences))
	}
	params.OutputFormat = knownOrDigest(params.OutputFormat, knownOutputFormats)
	params.OutputConfigEffort = knownOrDigest(wire.OutputConfig.Effort, knownEfforts)
	params.ReasoningObjectEffort = knownOrDigest(wire.Reasoning.Effort, knownEfforts)
	params.ReasoningMaxTokens = wire.Reasoning.MaxTokens
	params.TopLevelEffort = knownOrDigest(wire.ReasoningEffort, knownEfforts)
	params.ThinkingType = knownOrDigest(wire.Thinking.Type, knownThinkingTypes)
	params.ThinkingDisplay = knownOrDigest(wire.Thinking.Display, knownThinkingDisplay)
	params.ThinkingBudgetTokens = wire.Thinking.BudgetTokens
	params.ReasoningBudgetTokens = wire.ReasoningBudgetTokens
	params.TemplateEffort = knownOrDigest(wire.ChatTemplateKwargs.ReasoningEffort, knownEfforts)
	params.TemplateEnableThinking = wire.ChatTemplateKwargs.EnableThinking
	params.TemplateSwitches = wire.ChatTemplateKwargs.Switches
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

// providerSlug is the shape of an OpenRouter provider slug, with an optional
// variant (deepinfra/fp8). A value outside it (a hostname, a credential pasted
// into a pin) is recorded only as a digest.
var providerSlug = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9-]{0,31}(/[A-Za-z0-9][A-Za-z0-9-]{0,31})?$`)

// outboundDigest stands in for a client-controlled value the record must not
// publish: equal values still correlate, the value itself never appears.
func outboundDigest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:6])
}

// knownOrDigest keeps a value the protocols define and digests any other, so a
// client string in an enum's place (prompt text, a key, a hostname) is never
// published. An empty value stays empty.
func knownOrDigest(value string, known map[string]bool) string {
	if value == "" || known[value] {
		return value
	}
	return outboundDigest(value)
}

func enumSet(values ...string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}

// The enums the record names verbatim. Each string field below is written by
// a client on a same-format dispatch and is not validated to these values.
var (
	knownEfforts         = enumSet("none", "minimal", "low", "medium", "high", "xhigh", "max")
	knownThinkingTypes   = enumSet("enabled", "disabled", "adaptive", "auto")
	knownThinkingDisplay = enumSet("summarized", "omitted")
	knownOutputFormats   = enumSet("text", "json_object", "json_schema")
	knownToolChoices     = enumSet("auto", "none", "required", "any", "tool", "function", "allowed_tools", "custom")
	knownCacheTypes      = enumSet("ephemeral")
	knownCacheTTLs       = enumSet("5m", "1h")
	knownCacheScopes     = enumSet()
	knownTruncations     = enumSet("auto", "disabled")
)

// outboundListLimit bounds every client-sized list or map the record carries,
// so a large client object cannot make each routing line large.
const outboundListLimit = 16

// templateKey is the shape a chat_template_kwargs key is recorded verbatim in.
var templateKey = regexp.MustCompile(`^[a-z][a-z0-9_]{0,31}$`)

// templateKwargs reads chat_template_kwargs: the two named reasoning
// controls, and Switches, the scalar members as compact JSON. A bool or
// number is kept, a string only as a known effort (else a digest), a key
// outside templateKey as a digest; objects and arrays are left out, and at
// most outboundListLimit members (by sorted key) are kept.
type templateKwargs struct {
	ReasoningEffort string
	EnableThinking  *bool
	Switches        string
}

func (kwargs *templateKwargs) UnmarshalJSON(raw []byte) error {
	*kwargs = templateKwargs{}
	var members map[string]json.RawMessage
	if json.Unmarshal(raw, &members) != nil {
		return nil
	}
	keys := make([]string, 0, len(members))
	for key := range members {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	switches := map[string]interface{}{}
	for _, key := range keys {
		value := members[key]
		var text string
		var flag bool
		var number json.Number
		var recorded interface{}
		switch {
		case string(value) == "null":
			continue
		case json.Unmarshal(value, &flag) == nil:
			recorded = flag
			if key == "enable_thinking" {
				kwargs.EnableThinking = &flag
			}
		case json.Unmarshal(value, &text) == nil:
			recorded = knownOrDigest(text, knownEfforts)
			if key == "reasoning_effort" {
				kwargs.ReasoningEffort = text
			}
		case json.Unmarshal(value, &number) == nil:
			recorded = number
		default:
			continue
		}
		if len(switches) >= outboundListLimit {
			continue
		}
		if !templateKey.MatchString(key) {
			key = outboundDigest(key)
		}
		switches[key] = recorded
	}
	if len(switches) > 0 {
		if encoded, err := json.Marshal(switches); err == nil {
			kwargs.Switches = string(encoded)
		}
	}
	return nil
}

// maxPriceKinds are the price kinds OpenRouter's max_price accepts; any other
// key is recorded as a digest.
var maxPriceKinds = map[string]bool{"prompt": true, "completion": true, "request": true, "image": true}

// redactProviderPreferences keeps the provider lists' slugs and the
// data_collection enum readable and replaces any other value: a same-format
// dispatch carries the client's own provider object, whose strings are not
// validated.
func redactProviderPreferences(provider *config.OpenRouterProviderPreferences) {
	digest := outboundDigest
	redact := func(values []string) []string {
		if len(values) > outboundListLimit {
			values = append(values[:outboundListLimit:outboundListLimit], "+"+strconv.Itoa(len(values)-outboundListLimit)+" more")
		}
		for i, value := range values {
			if i == outboundListLimit {
				break
			}
			if !providerSlug.MatchString(value) {
				values[i] = digest(value)
			}
		}
		return values
	}
	provider.Order, provider.Only, provider.Ignore = redact(provider.Order), redact(provider.Only), redact(provider.Ignore)
	if provider.MaxPrice != nil {
		kinds := make([]string, 0, len(provider.MaxPrice))
		for kind := range provider.MaxPrice {
			kinds = append(kinds, kind)
		}
		sort.Strings(kinds)
		known := make(map[string]float64, len(provider.MaxPrice))
		for _, kind := range kinds {
			if len(known) >= outboundListLimit {
				break
			}
			key := kind
			if !maxPriceKinds[kind] {
				key = digest(kind)
			}
			known[key] = provider.MaxPrice[kind]
		}
		provider.MaxPrice = known
	}
	if provider.DataCollection != "" && provider.DataCollection != "allow" && provider.DataCollection != "deny" {
		provider.DataCollection = "other"
	}
}

// toolChoiceObject is tool_choice in either form: a scalar mode ("auto",
// "none", "required") or an object whose type is the mode and which may name
// one tool (Messages {type, name}, Chat {type, function: {name}}) and carry
// Messages' inverted parallel flag. A tool name is an id, not content.
type toolChoiceObject struct {
	Mode                   string
	Name                   string
	DisableParallelToolUse *bool
}

func (choice *toolChoiceObject) UnmarshalJSON(raw []byte) error {
	*choice = toolChoiceObject{}
	if len(raw) == 0 {
		return nil
	}
	if raw[0] == '"' {
		return json.Unmarshal(raw, &choice.Mode)
	}
	if raw[0] != '{' {
		return nil
	}
	var object struct {
		Type                   string `json:"type"`
		Name                   string `json:"name"`
		DisableParallelToolUse *bool  `json:"disable_parallel_tool_use"`
		Function               struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return err
	}
	choice.Mode, choice.Name = object.Type, object.Name
	if choice.Name == "" {
		choice.Name = object.Function.Name
	}
	choice.DisableParallelToolUse = object.DisableParallelToolUse
	return nil
}

// outputFormatObject is a structured-output format's type and strict flag;
// its schema is never read.
type outputFormatObject struct {
	Type   string `json:"type"`
	Strict *bool  `json:"strict"`
}

// stopSequences counts stop or stop_sequences: a string is one sequence.
type stopSequences int

func (count *stopSequences) UnmarshalJSON(raw []byte) error {
	*count = 0
	switch {
	case len(raw) > 0 && raw[0] == '"':
		*count = 1
	case len(raw) > 0 && raw[0] == '[':
		var list []json.RawMessage
		if err := json.Unmarshal(raw, &list); err != nil {
			return err
		}
		*count = stopSequences(len(list))
	}
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
	// Strict is a tool's strict flag (Chat nests it under function); raw so
	// an unexpected shape cannot fail the decode.
	Strict   json.RawMessage `json:"strict"`
	Function struct {
		Strict json.RawMessage `json:"strict"`
	} `json:"function"`
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
// type/ttl/scope with an empty member left empty (provider default), plus
// /evict_on_complete=<bool> when the directive carries that member.
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
			Evict *bool  `json:"evict_on_complete"`
		}
		if json.Unmarshal(raw, &directive) == nil {
			policy := knownOrDigest(directive.Type, knownCacheTypes) + "/" + knownOrDigest(directive.TTL, knownCacheTTLs) +
				"/" + knownOrDigest(directive.Scope, knownCacheScopes)
			if directive.Evict != nil {
				policy += "/evict_on_complete=" + strconv.FormatBool(*directive.Evict)
			}
			policies[policy] = true
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
	if params.ProviderPreferences != "" {
		record["outbound_provider_preferences"] = params.ProviderPreferences
	}
	if params.ToolChoice != "" {
		record["outbound_tool_choice"] = params.ToolChoice
	}
	if params.ToolChoiceName != "" {
		record["outbound_tool_choice_name"] = params.ToolChoiceName
	}
	if params.StopSequences != nil {
		record["outbound_stop_sequences"] = *params.StopSequences
	}
	if params.OutputFormat != "" {
		record["outbound_output_format"] = params.OutputFormat
	}
	if params.OutputFormatStrict != nil {
		record["outbound_output_format_strict"] = *params.OutputFormatStrict
	}
	if params.PromptCacheRetention != "" {
		record["outbound_prompt_cache_retention"] = params.PromptCacheRetention
	}
	if params.ReasoningSummary != "" {
		record["outbound_reasoning_summary"] = params.ReasoningSummary
	}
	if params.PreviousResponseID {
		record["outbound_previous_response_id"] = true
	}
	if params.Conversation {
		record["outbound_conversation"] = true
	}
	if params.Store != nil {
		record["outbound_store"] = *params.Store
	}
	if params.AutoStore != nil {
		record["outbound_auto_store"] = *params.AutoStore
	}
	if params.Truncation != "" {
		record["outbound_truncation"] = params.Truncation
	}
	if params.StreamIncludeUsage != nil {
		record["outbound_stream_include_usage"] = *params.StreamIncludeUsage
	}
	if params.StreamIncludeObfuscation != nil {
		record["outbound_stream_include_obfuscation"] = *params.StreamIncludeObfuscation
	}
	record["outbound_strict_tool_count"] = params.StrictToolCount
	if params.TaskBudget != nil {
		record["outbound_task_budget"] = *params.TaskBudget
	}
	optionalString := map[string]string{
		"outbound_output_config_effort": params.OutputConfigEffort,
		"outbound_reasoning_effort":     params.ReasoningObjectEffort,
		"outbound_top_level_effort":     params.TopLevelEffort,
		"outbound_thinking_type":        params.ThinkingType,
		"outbound_thinking_display":     params.ThinkingDisplay,
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
	if params.TemplateSwitches != "" {
		record["outbound_template_switches"] = params.TemplateSwitches
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
