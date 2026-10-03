package llmprotocol

// A routing capability is a request feature whose loss changes what the model
// is shown, not merely how an answer is presented.
//
// The wire capability set in capabilities.go answers a different question: can
// this format express this request at all. It fails an encode. A routing
// capability fails no encode: the target format can carry the rest of the
// request perfectly well, and dropping the feature produces a body a provider
// accepts and answers wrongly: a declared web search becomes no web search.
// The turn succeeds and the answer is built on less than the caller sent.
//
// So the decision belongs before the encode, at model selection: only an arm
// that holds the capability is eligible. When no arm holds it the cell answers
// a replayable status, never a 400, because the gateway does not replay a
// body-level 400 and the caller would lose the turn outright.
const (
	// RoutingCapabilityServerTools covers a tool the source API runs itself:
	// web search, the advisor. Nothing strips these before the cell on this
	// auth path, and a Chat arm has no shape to hold the declaration. An
	// Anthropic-defined tool the caller runs, the text editor or bash, is
	// not covered: the codec writes its schema out for any target, so a
	// Chat arm serves it without holding anything.
	RoutingCapabilityServerTools = "server_tools"
	// RoutingCapabilityToolResultImages named an image inside a tool result.
	// It is no longer derived: every encoder now carries tool-result media
	// (Messages and Responses natively, Chat by moving it to a user message
	// after the tool messages, see codec_openai_chat_tool_result_media.go), so
	// no target drops it. What such a turn still needs is image input, and the
	// vision gate already reads images nested in a tool result. The name stays
	// so a model card that declares it still loads.
	RoutingCapabilityToolResultImages = "tool_result_images"
	// RoutingCapabilityCitationsGeneration is reserved for the response side:
	// an arm that can produce the citation spans a request asked for. Nothing
	// derives it yet; the response-side table is CP9v.
	RoutingCapabilityCitationsGeneration = "citations_generation"
)

// RequiredRoutingCapabilities names what an arm must hold to serve this
// request without changing what the model sees. The order is stable so a log
// line and a metric label do not depend on map iteration.
func RequiredRoutingCapabilities(request Request) []string {
	var required []string
	if declaresServerTool(request.Tools) {
		required = append(required, RoutingCapabilityServerTools)
	}
	return required
}

func declaresServerTool(tools []Tool) bool {
	for _, tool := range tools {
		if tool.ServerTool() {
			return true
		}
	}
	return false
}
