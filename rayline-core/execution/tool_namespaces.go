package execution

import (
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// Codex declares some of its tools inside Responses namespace tools (its
// multi_agent_v1 sub-agent functions) and names the namespace beside the
// function on each call. Rayline Platform serves namespace tools; a user's
// own Responses server may serve function tools only, and refuses the whole
// request. So a Responses request bound for a local deployment declares each
// namespaced function as a plain function under its qualified name
// (llmprotocol.QualifiedToolName, the name Chat and Messages already call it
// by), and a call the model makes under a qualified name the request declared
// gets its namespace back before the client sees it. A call under any other
// name passes through unchanged.

// flattensToolNamespaces reports whether a deployment's requests declare
// namespaced functions as plain ones.
func flattensToolNamespaces(deployment Deployment) bool {
	return deployment.Kind == "local" && deployment.Protocol == llmprotocol.OpenAIResponsesV1
}

// flattenToolNamespaces rewrites, on fresh slices, every namespaced tool and
// every namespaced call in the history under its qualified name, and returns
// the map that restores the namespace on a returned call. Decoding already
// refused a request in which a qualified name is also another tool's name
// (duplicate_tool), so every flattened name stands for one tool.
func flattenToolNamespaces(request *llmprotocol.Request) map[string][2]string {
	namespaces := llmprotocol.ToolNamespaces(request.Tools)
	if namespaces == nil {
		return nil
	}
	tools := make([]llmprotocol.Tool, 0, len(request.Tools))
	for _, tool := range request.Tools {
		if tool.Namespace != "" {
			tool.Description = llmprotocol.FlattenedToolDescription(tool)
			tool.Name = llmprotocol.QualifiedToolName(tool.Namespace, tool.Name)
			tool.Namespace, tool.NamespaceDescription = "", ""
		}
		tools = append(tools, tool)
	}
	request.Tools = tools
	messages := make([]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		messages[index] = message
		if !holdsNamespacedCall(message) {
			continue
		}
		contents := make([]llmprotocol.Content, len(message.Content))
		for contentIndex, content := range message.Content {
			if call := content.ToolCall; call != nil && call.Namespace != "" {
				flattened := *call
				flattened.Name = llmprotocol.QualifiedToolName(call.Namespace, call.Name)
				flattened.Namespace = ""
				content.ToolCall = &flattened
			}
			contents[contentIndex] = content
		}
		messages[index].Content = contents
	}
	request.Messages = messages
	return namespaces
}

func holdsNamespacedCall(message llmprotocol.Message) bool {
	for _, content := range message.Content {
		if content.ToolCall != nil && content.ToolCall.Namespace != "" {
			return true
		}
	}
	return false
}
