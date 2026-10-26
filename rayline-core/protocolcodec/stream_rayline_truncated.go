package protocolcodec

import "github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

// Saying why a turn the Router ended was ended.
//
// A cut turn terminates in the client's own normal shape, so that the stream
// closes and the tokens already spent are billed (stream_truncation_usage.go).
// The stop it has to state is "length", because that is the nearest thing Chat
// Completions' closed vocabulary has -- stop, length, tool_calls,
// content_filter -- and none of them names a turn infrastructure ended. Read
// literally, "length" says the model exhausted its token budget, which is true
// of a deadline and not of a connection that broke.
//
// So the cause travels beside it, in one namespaced member, and a client that
// needs to choose between retrying and asking for less work reads that rather
// than inferring it from the stop. Namespaced so it cannot collide with a
// future OpenAI member; a string rather than a boolean so a further cause can
// be added without changing the shape (memex-desktop#7214, decision
// 2026-10-22).
//
// Whether OpenRouter forwards an unknown member on a chunk to its own users is
// open (memex-desktop#7216). If it strips it, the member still reaches
// first-party callers, and the Router's own usage row carries the same cause
// either way, so rate tracking does not depend on it.
type raylineChunkWire struct {
	Truncated string `json:"truncated,omitempty"`
}

// The causes a cut turn states. Each is the mechanism, not a remedy and not a
// transport category: a caller reads "deadline" and asks for less work in one
// turn, "silence" and retries a model that stopped talking, "transport" and
// retries a connection.
//
// "deadline" rather than "timeout" even though the Router's own error category
// is a timeout. A timeout on this path already means something else -- an
// upstream timeout, the route timer, a socket ETIMEDOUT -- and a caller reading
// it infers the model stopped responding. On a deadline cut it had not: the arm
// was producing tokens normally and a fixed wall-clock budget ran out, so a
// retry of the same long turn meets the same budget.
const (
	raylineTruncatedDeadline  = "deadline"
	raylineTruncatedSilence   = "silence"
	raylineTruncatedTransport = "transport"
)

// raylineTruncationCause maps the Router's own name for a cut onto the cause a
// client is told. Only a turn the Router itself ended reaches here; a provider
// failure is the provider's to describe and states no cause of ours.
//
// An unrecognized code states no cause at all rather than guessing one. The
// terminal still carries its count and still closes the stream, so a code added
// without a cause beside it costs a caller the distinction and nothing else.
func raylineTruncationCause(protocolError *llmprotocol.ProtocolError) string {
	if protocolError == nil {
		return ""
	}
	switch protocolError.Code {
	case "stream_truncated":
		return raylineTruncatedDeadline
	case "stream_idle_timeout", "stream_first_content_timeout":
		return raylineTruncatedSilence
	case "standard_resend_body_cut":
		return raylineTruncatedTransport
	default:
		return ""
	}
}
