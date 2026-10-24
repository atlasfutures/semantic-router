package extproc

import "github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

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
