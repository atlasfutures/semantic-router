package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// requestToolNamespaces maps the qualified names of this request's namespaced
// tools back to their namespace and function; nil when it declares none. A
// Chat or Messages provider calls a namespaced tool by its qualified name, and
// the client that declared the namespace, Codex, needs it back on the call.
func requestToolNamespaces(ctx *RequestContext) map[string][2]string {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil
	}
	return llmprotocol.ToolNamespaces(ctx.SemanticRequest.Tools)
}

// restoreResponseToolNamespaces puts the namespace back on every call in a
// buffered response that came back under a qualified name.
func restoreResponseToolNamespaces(response *llmprotocol.Response, namespaces map[string][2]string) {
	if response == nil || namespaces == nil {
		return
	}
	for itemIndex := range response.Output {
		for contentIndex := range response.Output[itemIndex].Content {
			llmprotocol.RestoreToolNamespace(response.Output[itemIndex].Content[contentIndex].ToolCall, namespaces)
		}
	}
}
