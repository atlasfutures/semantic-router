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
	Kind           ContentKind
	Text           string
	Citations      []Citation
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
	OutputFormat          OutputFormat
	ReasoningMode         ReasoningMode
	ReasoningEffort       string
	ReasoningBudgetTokens *int64
	// ReasoningDisplay controls whether a provider returns summarized reasoning
	// content or only its signed continuation token. It is distinct from whether
	// reasoning itself is enabled.
	ReasoningDisplay   string
	Stream             bool
	StreamOptions      StreamOptions
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
}

type UsageState string

const (
	UsageAvailable   UsageState = "available"
	UsageUnavailable UsageState = "unknown"
)

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
