package llmprotocol

// A routing capability is a request feature whose loss changes what the model
// is shown, not merely how an answer is presented.
//
// The wire capability set in capabilities.go answers a different question: can
// this format express this request at all. It fails an encode. A routing
// capability fails no encode: the target format can carry the rest of the
// request perfectly well, and dropping the feature produces a body a provider
// accepts and answers wrongly. A tool result that held a screenshot becomes an
// empty tool result; a declared web search becomes no web search. The turn
// succeeds and the answer is built on less than the caller sent.
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
	// RoutingCapabilityToolResultImages covers an image inside a tool result.
	// Reading a PNG produces one, and so does any MCP tool that returns an
	// image. Every dispatch format now carries it (Chat moves the media into
	// a user message after the tool messages), so what an arm's card claims
	// is whether its model takes image input at all.
	RoutingCapabilityToolResultImages = "tool_result_images"
	// RoutingCapabilityCitationsGeneration is reserved for the response side:
	// an arm that can produce the citation spans a request asked for. Nothing
	// derives it yet; the response-side table is CP9v.
	RoutingCapabilityCitationsGeneration = "citations_generation"
	// RoutingCapabilityTools states that the model takes a tool declaration
	// and can answer with a tool call. It is the catalog's own spelling, and
	// it is descriptive: nothing derives it from a request, so a card that
	// claims it is not gated by it. It exists so an arm set can state the
	// fact in one spelling and a gateway can read it off the arm listing
	// rather than hard-code it.
	RoutingCapabilityTools = "tools"
)

// RequiredRoutingCapabilities names what an arm must hold to serve this
// request without changing what the model sees. The order is stable so a log
// line and a metric label do not depend on map iteration.
func RequiredRoutingCapabilities(request Request) []string {
	var required []string
	if declaresServerTool(request.Tools) {
		required = append(required, RoutingCapabilityServerTools)
	}
	if carriesToolResultMedia(request) {
		required = append(required, RoutingCapabilityToolResultImages)
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

func carriesToolResultMedia(request Request) bool {
	for _, instruction := range request.Instructions {
		if contentCarriesToolResultMedia(instruction.Content) {
			return true
		}
	}
	for _, message := range request.Messages {
		if contentCarriesToolResultMedia(message.Content) {
			return true
		}
	}
	return false
}

// contentCarriesToolResultMedia reports whether a tool result holds a part
// that is not text. A carried block is not counted here: it belongs to the
// source contract, is re-emitted only to that contract, and appendCarriedBlockDrops
// counts it as a drop on any other target -- which it now does for a block
// inside a tool result, not only for one beside it.
func contentCarriesToolResultMedia(contents []Content) bool {
	for _, content := range contents {
		if content.Kind != ContentToolResult || content.ToolResult == nil {
			continue
		}
		for _, nested := range content.ToolResult.Content {
			if nested.Kind != ContentText && nested.Kind != ContentUnmodeled {
				return true
			}
		}
	}
	return false
}
