package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// The Workshop agent declares its edit tool as an Anthropic-defined tool:
// a type and a name, no schema, because Claude knows the schema and the API
// fills it in. The caller runs the tool. On 2026-09-14 every such turn was
// refused by the cell as needing a server-tool arm (memex-desktop#6902). The
// Router now writes the documented schema out for a non-Anthropic backend
// under the caller's name, and carries the declaration as written to an
// Anthropic one. These two cases pin that contract at the dispatch boundary:
// what the backend actually received.
const (
	anthropicDefinedToolName = "str_replace_based_edit_tool"
	anthropicDefinedToolType = "text_editor_20250728"
)

func init() {
	pkgtestcases.Register("anthropic-defined-tool-chat-backend", pkgtestcases.TestCase{
		Description: "An Anthropic-defined tool declared by type reaches a Chat Completions backend as a function with its documented schema, buffered and streaming",
		Tags:        []string{"anthropic", "tools", "protocol-codec", "streaming"},
		Fn:          testAnthropicDefinedToolChatBackend,
	})
	pkgtestcases.Register("anthropic-defined-tool-anthropic-backend", pkgtestcases.TestCase{
		Description: "An Anthropic-defined tool declared by type reaches an Anthropic Messages backend as declared",
		Tags:        []string{"anthropic", "tools", "protocol-codec"},
		Fn:          testAnthropicDefinedToolAnthropicBackend,
	})
}

func anthropicDefinedToolRequest(model, prompt string, stream bool) map[string]any {
	return map[string]any{
		"model":      model,
		"max_tokens": 16,
		"stream":     stream,
		"tools": []any{
			map[string]any{
				"name": "read_file", "description": "Read a file",
				"input_schema": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}},
			},
			map[string]any{"name": anthropicDefinedToolName, "type": anthropicDefinedToolType},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": prompt},
		},
	}
}

func testAnthropicDefinedToolChatBackend(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	backendSession, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer backendSession.Close()

	for _, stream := range []bool{false, true} {
		mode := "buffered"
		if stream {
			mode = "streaming"
		}
		sessionID := fmt.Sprintf("anthropic-defined-tool-chat-%s-%d", mode, time.Now().UnixNano())
		body, err := sendProtocolMatrixRequestWithHeaders(
			ctx, session, "/v1/messages",
			anthropicDefinedToolRequest(chatBackendModel, "anthropic-defined tool "+mode, stream),
			stream, map[string]string{"x-vsr-test-session-id": sessionID},
		)
		if err != nil {
			return fmt.Errorf("%s: %w", mode, err)
		}
		if stream {
			err = validateAnthropicProtocolStream(body, protocolCodecChatReply)
		} else {
			err = assertAnthropicBody(body, protocolCodecChatReply)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", mode, err)
		}
		forwarded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
		if err != nil {
			return fmt.Errorf("%s: %w", mode, err)
		}
		if err := validateChatBackendMaterializedDefinedTool(forwarded); err != nil {
			return fmt.Errorf("%s dispatch: %w", mode, err)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"buffered_tool_materialized": true, "streaming_tool_materialized": true})
	}
	return nil
}

// validateChatBackendMaterializedDefinedTool reads the Chat body the mock
// backend recorded. Both tools must arrive as functions, the edit tool under
// the caller's name with the documented schema, and the type a Chat provider
// rejects must appear nowhere in the body.
func validateChatBackendMaterializedDefinedTool(body []byte) error {
	var debug struct {
		Body struct {
			Tools []struct {
				Type     string `json:"type"`
				Function struct {
					Name       string `json:"name"`
					Parameters struct {
						Required   []string                   `json:"required"`
						Properties map[string]json.RawMessage `json:"properties"`
					} `json:"parameters"`
				} `json:"function"`
			} `json:"tools"`
		} `json:"body"`
	}
	if err := json.Unmarshal(body, &debug); err != nil {
		return fmt.Errorf("decode provider request: %w", err)
	}
	if strings.Contains(string(body), anthropicDefinedToolType) {
		return fmt.Errorf("the Chat backend still received the %s type: %s", anthropicDefinedToolType, truncateString(string(body), 800))
	}
	if len(debug.Body.Tools) != 2 {
		return fmt.Errorf("the Chat backend received %d tools, want 2: %s", len(debug.Body.Tools), truncateString(string(body), 800))
	}
	edit := debug.Body.Tools[1]
	if edit.Type != "function" || edit.Function.Name != anthropicDefinedToolName {
		return fmt.Errorf("the edit tool did not arrive as a function under the caller's name: %s", truncateString(string(body), 800))
	}
	if strings.Join(edit.Function.Parameters.Required, ",") != "command,path" {
		return fmt.Errorf("the edit tool did not carry the documented schema: required = %v", edit.Function.Parameters.Required)
	}
	for _, property := range []string{"command", "path", "old_str", "new_str", "file_text", "insert_line", "view_range"} {
		if _, present := edit.Function.Parameters.Properties[property]; !present {
			return fmt.Errorf("the edit tool schema lacks %q: %s", property, truncateString(string(body), 800))
		}
	}
	return nil
}

func testAnthropicDefinedToolAnthropicBackend(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	backendSession, err := openProtocolCodecProviderSession(ctx, client, opts, "anthropic.messages.v1")
	if err != nil {
		return err
	}
	defer backendSession.Close()

	sessionID := fmt.Sprintf("anthropic-defined-tool-anthropic-%d", time.Now().UnixNano())
	body, err := sendProtocolMatrixRequestWithHeaders(
		ctx, session, "/v1/messages",
		anthropicDefinedToolRequest("MoM", protocolCodecAnthropicProbe, false),
		false, map[string]string{"x-vsr-test-session-id": sessionID},
	)
	if err != nil {
		return err
	}
	if err := assertAnthropicBody(body, protocolCodecAnthropicReply); err != nil {
		return err
	}
	forwarded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	if err := validateAnthropicBackendKeptDefinedTool(forwarded); err != nil {
		return fmt.Errorf("buffered dispatch: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"tool_carried": true})
	}
	return nil
}

// validateAnthropicBackendKeptDefinedTool reads the Messages body the
// Anthropic backend recorded. The edit tool must still be declared by type
// and name with no schema written out, so a Claude arm keeps its prompt
// cache and its own definition of the tool.
func validateAnthropicBackendKeptDefinedTool(body []byte) error {
	var debug struct {
		Body struct {
			Tools []struct {
				Name        string          `json:"name"`
				Type        string          `json:"type"`
				InputSchema json.RawMessage `json:"input_schema"`
			} `json:"tools"`
		} `json:"body"`
	}
	if err := json.Unmarshal(body, &debug); err != nil {
		return fmt.Errorf("decode provider request: %w", err)
	}
	if len(debug.Body.Tools) != 2 {
		return fmt.Errorf("the Anthropic backend received %d tools, want 2: %s", len(debug.Body.Tools), truncateString(string(body), 800))
	}
	edit := debug.Body.Tools[1]
	if edit.Name != anthropicDefinedToolName || edit.Type != anthropicDefinedToolType || len(edit.InputSchema) != 0 {
		return fmt.Errorf("the Anthropic backend did not receive the edit tool as declared: %s", truncateString(string(body), 800))
	}
	return nil
}
