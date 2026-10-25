package llmprotocol

// RestoreResponseToolNamespaces puts the namespace back on every call in a
// buffered response that came back under a qualified name.
func RestoreResponseToolNamespaces(response *Response, namespaces map[string][2]string) {
	if response == nil || namespaces == nil {
		return
	}
	for itemIndex := range response.Output {
		for contentIndex := range response.Output[itemIndex].Content {
			RestoreToolNamespace(response.Output[itemIndex].Content[contentIndex].ToolCall, namespaces)
		}
	}
}
