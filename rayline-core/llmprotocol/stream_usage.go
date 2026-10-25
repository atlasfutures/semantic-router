package llmprotocol

// WithUpstreamStreamUsage asks a Chat backend for authoritative accounting
// while preserving the caller's original public stream preference. Retiring
// the input generation prevents same-format replay from hiding the override.
func WithUpstreamStreamUsage(request Request, target WireFormat) Request {
	if target == OpenAIChatV1 && request.Stream &&
		(request.StreamOptions.IncludeUsage == nil || !*request.StreamOptions.IncludeUsage) {
		includeUsage := true
		request.StreamOptions.IncludeUsage = &includeUsage
		request.Generation++
	}
	return request
}
