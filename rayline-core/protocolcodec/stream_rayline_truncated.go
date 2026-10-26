package protocolcodec

// Saying why a turn the host ended was ended.
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
// be added without changing the shape.
//
// The cause itself is the host's statement (llmprotocol.Truncation*, set
// through StreamEngine.SetTruncationCause): the host that cut the turn is the
// one that knows why, and this codec only writes the word it was given. A
// terminal with no cause stated carries no member. Responses states its own
// reason in incomplete_details and Messages keeps its published error frame,
// so neither writes this member.
type raylineChunkWire struct {
	Truncated string `json:"truncated,omitempty"`
}
