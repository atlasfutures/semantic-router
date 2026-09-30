package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

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

// withResponseToolNamespaces chains namespace restoration after the client
// response mutation, so a namespaced call is returned to the client that
// declared it under its namespace whatever else the mutation does.
func withResponseToolNamespaces(
	mutation protocolcodec.ResponseMutation,
	namespaces map[string][2]string,
) protocolcodec.ResponseMutation {
	if namespaces == nil {
		return mutation
	}
	return func(response *llmprotocol.Response) error {
		if mutation != nil {
			if err := mutation(response); err != nil {
				return err
			}
		}
		restoreResponseToolNamespaces(response, namespaces)
		return nil
	}
}

// withStreamToolNamespaces is withResponseToolNamespaces for stream events.
func withStreamToolNamespaces(
	mutation protocolcodec.StreamEventMutation,
	namespaces map[string][2]string,
) protocolcodec.StreamEventMutation {
	if namespaces == nil {
		return mutation
	}
	return func(event *llmprotocol.Event) error {
		if mutation != nil {
			if err := mutation(event); err != nil {
				return err
			}
		}
		llmprotocol.RestoreToolNamespace(event.ToolCall, namespaces)
		return nil
	}
}
