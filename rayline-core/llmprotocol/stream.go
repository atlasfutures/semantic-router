package llmprotocol

import "context"

type EventType string

// FailureScope distinguishes a transport-level stream error from a model
// response that reached a terminal failed state. Formats that expose only one
// error event may render both scopes identically.
type FailureScope string

const (
	FailureTransport FailureScope = "transport"
	FailureResponse  FailureScope = "response"
	// FailureTruncated is a turn the Router itself ended after the upstream had
	// already produced output: its own deadline, its silence limit, or a body
	// that was cut mid-read. The generation happened and was billed to us, so
	// the terminal has to carry the count -- and every one of these turns
	// stopped the model where it stood rather than failing, which is a stop
	// each client protocol can already say in its own vocabulary. An encoder
	// renders this scope as the shape its own format defines for a response
	// that stopped early, not as a transport error; see the stop reason the
	// event carries. Messages is the exception by contract (decision 31).
	FailureTruncated FailureScope = "truncated"
)

// The causes a host may give for a turn it ended. A host states one of these
// on a FailureTruncated terminal (Event.Truncation); a codec renders it where
// its format has a place for one. The vocabulary names the mechanism rather
// than a remedy or a transport category: a caller reads TruncationDeadline and
// asks for less work in one turn, TruncationSilence and retries a model that
// stopped talking, TruncationTransport and retries a connection. A string
// rather than a boolean, so a further cause can be added without changing the
// shape.
const (
	TruncationDeadline  = "deadline"
	TruncationSilence   = "silence"
	TruncationTransport = "transport"
)

const (
	EventResponseStarted         EventType = "response.started"
	EventOutputItemStarted       EventType = "output.item.started"
	EventOutputTextDelta         EventType = "output.text.delta"
	EventReasoningDelta          EventType = "output.reasoning.delta"
	EventToolCallDelta           EventType = "tool.call.delta"
	EventImageGenerationProgress EventType = "image_generation.progress"
	EventOutputItemCompleted     EventType = "output.item.completed"
	EventUsageUpdated            EventType = "usage.updated"
	EventResponseCompleted       EventType = "response.completed"
	EventResponseFailed          EventType = "response.failed"
	EventProviderOpaque          EventType = "provider.opaque"
	// EventKeepalive says the upstream is alive and has nothing to send yet:
	// an Anthropic ping, or an SSE frame of comment lines only. It carries no
	// content, usage, stop or identity. A target writes it in its own
	// keepalive form, so a client's idle watchdog still sees bytes.
	EventKeepalive EventType = "keepalive"
)

type Event struct {
	Sequence   uint64
	Type       EventType
	ResponseID string
	Model      string
	ItemIndex  int
	// ContentIndex identifies one ordered content block within an output item.
	// It is independent from ItemIndex: Responses can stream several text or
	// refusal parts inside one message, while other wire formats may flatten
	// those blocks on the wire.
	ContentIndex   int
	ItemID         string
	Role           Role
	Delta          string
	ToolCall       *ToolCall
	Content        *Content
	GeneratedImage *GeneratedImage
	StopReason     StopReason
	// SourceStopReason is the stop string the upstream itself sent, before it
	// was mapped onto the neutral reason. A buffered response keeps the same
	// field; a stream had nowhere to carry it.
	SourceStopReason string
	// ProviderStopReason and StopDetails are the response's, carried on the
	// events from the chunk that states them. See StopDetails.
	ProviderStopReason string
	StopDetails        *StopDetails
	// UpstreamProvider names the upstream that served the turn. It is Router
	// telemetry and no encoder publishes it.
	UpstreamProvider string
	// UpstreamServiceTier is the service tier the upstream says it served
	// the turn at, carried from the chunk that states it. Router telemetry.
	UpstreamServiceTier string
	MatchedStopSequence string
	Usage               *Usage
	Error               *ProtocolError
	Failure             FailureScope
	// Truncation is why the host ended the turn, on a FailureTruncated
	// terminal only: one of the Truncation* causes, or empty when the host
	// stated none. The host that cut the turn is the one that knows why; the
	// codec only carries it.
	Truncation string
	Opaque     []byte
}

type StreamContext struct {
	Context            context.Context
	Source             WireFormat
	Target             WireFormat
	Options            StreamOptions
	PublicModel        string
	ProviderModel      string
	ResponseID         string
	PreviousResponseID string
	// ThinkingMarkerFamily, when set, has the Messages stream encoder sign a
	// thinking block that would otherwise stop unsigned with a Router
	// thinking marker of this family (semantic-router #191).
	ThinkingMarkerFamily string
	// WorkerFamily, when set, names the model family of the worker that
	// serves the turn, as the Router resolved it from its configuration
	// ("anthropic" for Claude). The Chat stream encoder streams a Claude
	// worker's thinking in OpenRouter's reasoning_details shape, a tagged
	// fragment per chunk with the signature last; without it the encoder
	// knows only the wire format and waits for the signature
	// (semantic-router #278).
	WorkerFamily string
}

type StreamDecoder interface {
	Push([]byte) ([]Event, Diagnostics, error)
	Finalize(error) ([]Event, Diagnostics, error)
}

type StreamEncoder interface {
	Push(Event) ([][]byte, Diagnostics, error)
	Finalize(error) ([][]byte, Diagnostics, error)
}
