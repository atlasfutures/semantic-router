// Package llmprotocol defines the protocol-neutral semantic contract used by
// inference ingress, routing, backend dispatch, streaming, and accounting.
// Wire JSON belongs to codecs; it must not leak into this package.
package llmprotocol

import (
	"encoding/json"
	"time"
)

// WireFormat is a stable wire contract identifier, not a provider product.
type WireFormat string

const (
	OpenAIChatV1        WireFormat = "openai.chat.v1"
	OpenAIResponsesV1   WireFormat = "openai.responses.v1"
	AnthropicMessagesV1 WireFormat = "anthropic.messages.v1"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type ContentKind string

// ReasoningScope preserves whether a provider exposed full reasoning text or
// a user-facing summary. An empty scope means the source format did not make
// that distinction; codecs must not guess that unspecified reasoning is a
// summary.
type ReasoningScope string

const (
	ContentText       ContentKind = "text"
	ContentRefusal    ContentKind = "refusal"
	ContentImage      ContentKind = "image"
	ContentAudio      ContentKind = "audio"
	ContentVideo      ContentKind = "video"
	ContentFile       ContentKind = "file"
	ContentToolCall   ContentKind = "tool_call"
	ContentToolResult ContentKind = "tool_result"
	ContentReasoning  ContentKind = "reasoning"
	// ContentGeneratedImage represents one image-generation operation and its
	// result. It is intentionally distinct from ContentImage: the latter is an
	// image supplied as model input, while this kind preserves the lifecycle of
	// a model-hosted image generation tool.
	ContentGeneratedImage ContentKind = "generated_image"
	// ContentUnmodeled is a block this contract does not name, carried whole in
	// its source bytes so that routing does not have to refuse it. It holds no
	// semantics: nothing may read it except the codec that re-emits it.
	ContentUnmodeled ContentKind = "unmodeled"
)

const (
	ReasoningScopeText    ReasoningScope = "text"
	ReasoningScopeSummary ReasoningScope = "summary"
)

// Content is one ordered semantic block. Fields are closed by Kind. Data and
// references are never fetched by a codec.
type Content struct {
	Kind      ContentKind
	Text      string
	Citations []Citation
	// CitationsRaw is the citations member an Anthropic text or document
	// block carries. Claude Code echoes the citations of a web-search or
	// document answer back in history on every later turn, and the
	// Citations API marks a document with {"enabled": true}. Neither shape
	// fits Citation, which models one URL span, so the Router carries the
	// member unread. It records where an answer came from rather than
	// changing what the model is asked, so a format with nowhere to put it
	// drops and counts it instead of refusing the turn.
	CitationsRaw   json.RawMessage
	Cache          *CacheDirective
	MediaType      string
	URL            string
	Data           string
	FileID         string
	Filename       string
	Detail         string
	ToolCall       *ToolCall
	ToolResult     *ToolResult
	GeneratedImage *GeneratedImage
	Signature      string
	Reasoning      ReasoningScope
	Unmodeled      *UnmodeledBlock
	// Extensions holds the members of this block, at any depth, that the wire
	// contract the block came from does not name. Unmodeled carries a whole
	// block the contract cannot name at all; Extensions carries the parts of
	// a block it can.
	Extensions *UnmodeledFields
}

// CacheDirective marks a request block or tool definition as an explicit
// prompt-cache boundary. It is semantic request state rather than an opaque
// provider extension, so same-format routing mutations cannot silently erase
// it. A target format without cache directives must reject the translation.
type CacheDirective struct {
	Type string
	TTL  string
	// Scope is the member Claude Code sets on its cache breakpoints once it
	// negotiates the prompt-caching-scope beta. Anthropic has not published
	// what it means, and observed values differ ("global" here on
	// 2026-09-03, "turn" in a third-party report), so the Router carries it
	// unread rather than interpreting it: refusing the turn loses the
	// conversation and rewriting the value would move a cache entry the
	// Router does not understand.
	Scope string
}

// Citation is bounded, protocol-neutral attribution attached to a text block.
// Offsets are Unicode code-point indexes into Content.Text.
type Citation struct {
	URL        string
	Title      string
	StartIndex int64
	EndIndex   int64
}

type Message struct {
	ID      string
	Role    Role
	Content []Content
	// Configuration marks a content-less system message that changes
	// generation settings from its position on. Only the router writes one;
	// no decoder produces it.
	Configuration *ConfigurationUpdate
	// WireGroup names the client wire message this message was decoded from,
	// as its 1-based position, when that wire message held tool results.
	// Anthropic carries a turn's tool results, and any text beside them, in
	// one user message that the decoder splits into one tool message per
	// result; the Messages encoder rejoins consecutive messages of one group,
	// so the provider sees the client's grouping. Zero is no group.
	WireGroup int
}

// ConfigurationUpdate is a mid-conversation settings change. Only the
// reasoning effort is modelled: providers that document it apply the new
// effort from that point on without invalidating the cached prefix, which a
// request-level effort change always does.
type ConfigurationUpdate struct {
	ReasoningEffort string
}

type InstructionBlock struct {
	Role    Role
	Content []Content
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments string
	// Caller names who issued the call. Anthropic's programmatic tool
	// calling sets it on a tool_use block, and Claude Code sends
	// {"type":"direct"}. The Router carries it unread: it identifies the
	// issuer rather than changing what the model is asked, so a format
	// that cannot express it drops and counts it instead of refusing.
	Caller json.RawMessage
	// Namespace is the tool group a Responses call names beside its function,
	// as Codex's multi_agent_v1 sub-agent tools do. See Tool.Namespace.
	Namespace string
	// Incomplete marks a call the model was cut off in the middle of: its
	// output hit the token limit while it was writing the arguments, so
	// Arguments is the prefix it wrote and need not be a JSON object. Only a
	// response that stopped at max_tokens can hold one, as its last content;
	// every encoder says the call is incomplete in its own format, so a client
	// never takes the prefix for a call to run.
	Incomplete bool
}

type ToolResult struct {
	CallID  string
	Content []Content
	IsError *bool
	// DeferredLink means the referenced call belongs to retained conversation
	// state identified by PreviousResponseID instead of this request body. It is
	// semantic validation state and is never accepted without that continuation
	// reference.
	DeferredLink bool
}

type Tool struct {
	Name        string
	Description string
	Strict      *bool
	InputSchema json.RawMessage
	Cache       *CacheDirective
	// Type is the tool's own discriminator. An empty value and "custom" both
	// mean a tool the model calls and the caller runs. A value in the
	// Anthropic-defined table names a tool the caller runs too, declared by
	// type in place of a schema. Every other value names a tool the source
	// API runs itself -- web search, the advisor -- which a target format
	// that has no such tool cannot express.
	Type string
	// Extensions holds the members of this tool that the source contract does
	// not name.
	Extensions *UnmodeledFields
	// Namespace groups the tool with others under one name, and
	// NamespaceDescription describes the group: a Responses namespace tool,
	// which Codex uses for its multi_agent_v1 sub-agent functions. Responses
	// spells the group; a format that cannot calls the function by
	// QualifiedToolName, and the router restores the namespace on the call
	// that comes back.
	Namespace            string
	NamespaceDescription string
}

// namespaceSeparator joins a namespace and a function name into the one name
// a format without namespaces calls. It uses only characters every provider
// allows in a function name, [A-Za-z0-9_-].
const namespaceSeparator = "__"

// QualifiedToolName is the function name a format without namespaces calls a
// namespaced tool by. A tool with no namespace keeps its name.
func QualifiedToolName(namespace, name string) string {
	if namespace == "" {
		return name
	}
	return namespace + namespaceSeparator + name
}

// FlattenedToolDescription is the description a namespaced tool is declared
// with where the format has no namespaces: the namespace's description, which
// may be the only place a generic function such as close says what it closes,
// followed by the function's own.
func FlattenedToolDescription(tool Tool) string {
	switch {
	case tool.NamespaceDescription == "":
		return tool.Description
	case tool.Description == "":
		return tool.NamespaceDescription
	default:
		return tool.NamespaceDescription + "\n\n" + tool.Description
	}
}

// ToolNamespaces maps each qualified name this request declares to the
// namespace and function it stands for. The router uses it to restore the
// namespace on a call a Chat or Messages provider returns under the qualified
// name. Only names the request declared are restored, so a function whose own
// name happens to contain the separator is never split.
func ToolNamespaces(tools []Tool) map[string][2]string {
	var namespaces map[string][2]string
	for _, tool := range tools {
		if tool.Namespace == "" {
			continue
		}
		if namespaces == nil {
			namespaces = map[string][2]string{}
		}
		namespaces[QualifiedToolName(tool.Namespace, tool.Name)] = [2]string{tool.Namespace, tool.Name}
	}
	return namespaces
}

// RestoreToolNamespace puts back the namespace of a call returned under a
// qualified name, and reports whether it did.
func RestoreToolNamespace(call *ToolCall, namespaces map[string][2]string) bool {
	if call == nil || call.Namespace != "" {
		return false
	}
	parts, qualified := namespaces[call.Name]
	if !qualified {
		return false
	}
	call.Namespace, call.Name = parts[0], parts[1]
	return true
}

// ServerTool reports whether the source API runs this tool itself rather than
// handing the call back to the caller. A server tool is identified by its type
// alone: it declares no schema the Router could validate and need not name a
// function, because nothing outside the source API ever invokes it.
//
// An Anthropic-defined tool is typed the same way and is not one: the caller
// runs it, and the type stands in for a schema the table can supply. Reading
// it as a server tool gated every Workshop agent turn of 2026-09-14 behind a
// capability no arm can hold.
func (tool Tool) ServerTool() bool {
	if _, defined := tool.AnthropicDefined(); defined {
		return false
	}
	return tool.Type != "" && tool.Type != "custom"
}

// Identity is what makes two declared tools the same tool. A callable tool is
// its name, qualified by its namespace; a server tool that states no name is
// its type.
func (tool Tool) Identity() string {
	if tool.Name != "" {
		return QualifiedToolName(tool.Namespace, tool.Name)
	}
	return tool.Type
}

type ToolChoiceMode string

const (
	ToolChoiceAuto            ToolChoiceMode = "auto"
	ToolChoiceNone            ToolChoiceMode = "none"
	ToolChoiceRequired        ToolChoiceMode = "required"
	ToolChoiceNamed           ToolChoiceMode = "named"
	ToolChoiceImageGeneration ToolChoiceMode = "image_generation"
)

type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

type OutputFormatKind string

const (
	OutputText       OutputFormatKind = "text"
	OutputJSONObject OutputFormatKind = "json_object"
	OutputJSONSchema OutputFormatKind = "json_schema"
)

type OutputFormat struct {
	Kind        OutputFormatKind
	Name        string
	Description string
	Strict      *bool
	Schema      json.RawMessage
}

type ReasoningMode string

const (
	ReasoningModeEnabled  ReasoningMode = "enabled"
	ReasoningModeDisabled ReasoningMode = "disabled"
	ReasoningModeAdaptive ReasoningMode = "adaptive"
)

type Sampling struct {
	Temperature      *float64
	TopP             *float64
	TopK             *int64
	MaxOutputTokens  *int64
	Seed             *int64
	FrequencyPenalty *float64
	PresencePenalty  *float64
	Stop             []string
}

// StreamOptions contains public response-stream preferences. These options
// belong to the client contract rather than model semantics: Router dispatch
// may request additional accounting data from a backend without changing what
// the public stream exposes.
type StreamOptions struct {
	IncludeUsage       *bool
	IncludeObfuscation *bool
}

// TrustedMetadata is populated by the Router after transport authentication.
// Codecs never populate trusted fields from client headers.
type TrustedMetadata struct {
	NamespaceID   string
	ActorID       string
	SubjectID     string
	SessionID     string
	AgentID       string
	TaskID        string
	TurnID        string
	CorrelationID string
	SourceFormat  WireFormat
}

type Request struct {
	Generation        uint64
	Model             string
	Instructions      []InstructionBlock
	Messages          []Message
	Tools             []Tool
	ImageGeneration   *ImageGenerationOptions
	ToolChoice        ToolChoice
	ParallelToolCalls *bool
	CandidateCount    *int64
	Sampling          Sampling
	// ClientMaxOutputTokens is the output allowance the caller stated, kept
	// only when a Router plugin raises Sampling.MaxOutputTokens above it. It
	// is the number a derived reasoning bound comes from: raising the output
	// allowance so an answer has room beside the thinking must not also let
	// the turn think longer. No wire format carries it.
	ClientMaxOutputTokens *int64
	// RouterSetMaxOutputTokens marks a Sampling.MaxOutputTokens the Router
	// set because the caller stated none: a model card's max_output_tokens,
	// or the fallback for a target that requires a limit. It is not the
	// caller's allowance, so no reasoning bound derives from it. No wire
	// format carries it.
	RouterSetMaxOutputTokens bool
	OutputFormat             OutputFormat
	ReasoningMode            ReasoningMode
	ReasoningEffort          string
	ReasoningBudgetTokens    *int64
	// ReasoningDisplay controls whether a provider returns summarized reasoning
	// content or only its signed continuation token. It is distinct from whether
	// reasoning itself is enabled.
	ReasoningDisplay string
	Stream           bool
	StreamOptions    StreamOptions
	// AutoCache asks the provider to cache the conversation without naming
	// a block: Anthropic's top-level cache_control (automatic caching), and
	// OpenRouter's top-level cache_control on Responses. A target that caches
	// only at explicit breakpoints places one where automatic caching would,
	// so translating the request does not silently lose the client's cache
	// intent. Nil asks for nothing.
	AutoCache          *CacheDirective
	Metadata           map[string]string
	EndUserID          string
	PreviousResponseID string
	ConversationID     string
	Truncation         string
	Store              *bool
	AutoStore          *bool
	Trusted            TrustedMetadata
	// Unmodeled holds source-format members this contract does not name. It is
	// opaque to the Router and survives only to the wire format it came from.
	Unmodeled *UnmodeledFields
	// CarriedTools holds, whole, the tool declarations of a kind this contract
	// does not model: a Responses web_search or namespace tool, which Codex
	// declares on every turn. Every target drops and counts them for now (see
	// protocolcodec's carriedResponsesToolTypes). They do not gate which arm
	// serves the turn: without them the turn still works, and the model only
	// cannot call them.
	CarriedTools []UnmodeledBlock
	// HostedTools names the provider-run tools the arm this request is
	// dispatched to is admitted with (the model card's hosted_tools). A
	// carried declaration of one reaches a target of its own format only
	// when named here. The router sets it per dispatch; no decoder does.
	HostedTools []string
	// ForwardsEncryptedReasoning lets resent Responses reasoning items that
	// hold encrypted_content reach a Responses target, unchanged. Without it
	// every target drops and counts them. The router sets it per dispatch,
	// only when that target issued every blob the request resends; no
	// decoder does.
	ForwardsEncryptedReasoning bool
}

type StopReason string

const (
	StopEndTurn       StopReason = "end_turn"
	StopMaxTokens     StopReason = "max_tokens"
	StopSequence      StopReason = "stop_sequence"
	StopToolCall      StopReason = "tool_call"
	StopContentFilter StopReason = "content_filter"
	StopPaused        StopReason = "paused"
	StopContextWindow StopReason = "context_window_exceeded"
	StopCanceled      StopReason = "canceled"
	StopError         StopReason = "error"
	StopUnknown       StopReason = "unknown"
)

type UsageProvenance string

const (
	UsageAuthoritative UsageProvenance = "authoritative"
	UsageDerived       UsageProvenance = "derived"
	UsageEstimated     UsageProvenance = "estimated"
	UsageUnknown       UsageProvenance = "unknown"
)

// UsageSourceStreamEstimate marks a count that settles a turn the Router
// ended itself. The turn never reached its usage frame, so the count is what
// the Router observed rather than what the provider billed, and a consumer
// summing settlements has to be able to leave it out.
const (
	UsageSourceStreamEstimate = "stream_estimate"
)

// TokenCount uses a pointer so absent and an authoritative zero remain
// distinguishable.
type TokenCount struct {
	Value      *int64
	Provenance UsageProvenance
}

type Usage struct {
	State           UsageState
	InputUncached   TokenCount
	InputCacheRead  TokenCount
	InputCacheWrite TokenCount
	OutputReasoning TokenCount
	OutputOther     TokenCount
	InputTotal      TokenCount
	OutputTotal     TokenCount
	Total           TokenCount
	// ProviderCost is what the upstream itself said the call cost. It is
	// never derived from the counts above.
	ProviderCost ProviderCost
}

// ProviderCost is a charge an upstream states on its usage object, as
// OpenRouter does with usage.cost (USD credits). Each member is nil when the
// upstream did not state it, so a missing charge never reads as a free call.
type ProviderCost struct {
	// Charged is what the upstream charged for the call (usage.cost).
	Charged *float64
	// UpstreamInference is what the serving provider charged the aggregator
	// (usage.cost_details.upstream_inference_cost).
	UpstreamInference *float64
	// BYOK says the call ran on the caller's own provider key (usage.is_byok),
	// in which case Charged is the aggregator's fee alone.
	BYOK *bool
}

type UsageState string

const (
	UsageAvailable   UsageState = "available"
	UsageUnavailable UsageState = "unknown"
)

// StopDetails is the structured reason a provider gives for a stop: Anthropic's
// stop_details, whose type is "refusal" and whose category names the
// classifier that refused (for example "cyber"). With ProviderStopReason (the
// stop string the serving provider sent, such as OpenRouter's
// native_finish_reason, where it differs from the upstream's own reason) it is
// Router telemetry for a refused turn's diagnostics. No codec publishes either
// to a client.
type StopDetails struct {
	Type     string
	Category string
}

type Response struct {
	Generation uint64
	ID         string
	CreatedAt  time.Time
	Model      string
	Output     []OutputItem
	// Alternatives preserves additional, ordered model choices when a source
	// format supports them. A target that cannot represent alternatives must
	// apply the configured lossy policy; it may never silently pick one.
	Alternatives        [][]OutputItem
	StopReason          StopReason
	SourceStopReason    string
	MatchedStopSequence string
	Usage               Usage
	ProviderRequestID   string
	// UpstreamProvider names the upstream that served the turn, when the
	// provider reports one. It is Router telemetry: caching, thinking-off
	// handling and empty completions differ by provider rather than by model.
	// No codec publishes it to a client.
	UpstreamProvider string
	// ProviderStopReason and StopDetails say why the provider stopped, where
	// it says so beyond the neutral reason. See StopDetails.
	ProviderStopReason string
	StopDetails        *StopDetails
	// Evidence is bounded, protocol-neutral model evidence for Router
	// algorithms. It is never usage evidence and codecs do not publish it unless
	// the target protocol explicitly represents the same semantic field.
	Evidence ResponseEvidence
	Error    *ProtocolError
}

type ResponseEvidence struct {
	TokenLogprobs []TokenLogprob
}

type TokenLogprob struct {
	Token        string
	Logprob      float64
	Alternatives []TokenLogprobAlternative
}

type TokenLogprobAlternative struct {
	Token   string
	Logprob float64
}

type OutputItem struct {
	ID      string
	Role    Role
	Content []Content
}

func Int64(value int64) *int64       { return &value }
func Bool(value bool) *bool          { return &value }
func Float64(value float64) *float64 { return &value }
