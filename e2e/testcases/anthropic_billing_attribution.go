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

// Claude Code prepends one system block to every request, the billing
// attribution line, and two of its fields change on every turn. Anthropic
// reads the line as metadata and caches around it. Every other host reads it
// as prompt text, so the prompt prefix differs from about token 30 onward and
// no provider prefix cache can match. The Router drops the line on the way to
// a non-Anthropic backend and carries it to an Anthropic one. These two cases
// pin that contract at the dispatch boundary: what the backend actually
// received, not what the codec would produce in isolation.
const (
	billingAttributionLine   = "x-anthropic-billing-header: cc_version=2.1.260.ada; cc_entrypoint=sdk-cli; cch=1f3a9c2e; cc_prompt_id=7d1c2f6e-3b4a-4c5d-9e8f-0a1b2c3d4e5f;"
	billingAttributionPrefix = "x-anthropic-billing-header:"
	billingAttributionPrompt = "Reusable system context"
)

func init() {
	pkgtestcases.Register("anthropic-billing-attribution-chat-backend", pkgtestcases.TestCase{
		Description: "Claude Code's billing attribution line is dropped from buffered and streaming dispatch to a Chat Completions backend",
		Tags:        []string{"anthropic", "cache", "protocol-codec", "streaming"},
		Fn:          testBillingAttributionChatBackend,
	})
	pkgtestcases.Register("anthropic-billing-attribution-anthropic-backend", pkgtestcases.TestCase{
		Description: "Claude Code's billing attribution line reaches an Anthropic Messages backend unchanged",
		Tags:        []string{"anthropic", "cache", "protocol-codec"},
		Fn:          testBillingAttributionAnthropicBackend,
	})
}

func billingAttributionRequest(model, prompt string, stream bool) map[string]any {
	return map[string]any{
		"model":      model,
		"max_tokens": 16,
		"stream":     stream,
		"system": []any{
			map[string]any{"type": "text", "text": billingAttributionLine},
			map[string]any{"type": "text", "text": billingAttributionPrompt},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": prompt},
		},
	}
}

func testBillingAttributionChatBackend(
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
		sessionID := fmt.Sprintf("billing-attribution-chat-%s-%d", mode, time.Now().UnixNano())
		body, err := sendProtocolMatrixRequestWithHeaders(
			ctx, session, "/v1/messages",
			billingAttributionRequest(chatBackendModel, "billing attribution "+mode, stream),
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
		if err := validateChatBackendDroppedBillingAttribution(forwarded); err != nil {
			return fmt.Errorf("%s dispatch: %w", mode, err)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"buffered_line_dropped": true, "streaming_line_dropped": true})
	}
	return nil
}

// validateChatBackendDroppedBillingAttribution reads the Chat body the mock
// backend recorded. The system prompt beside the line must arrive as the
// first message, and the line must appear nowhere in the body.
func validateChatBackendDroppedBillingAttribution(body []byte) error {
	var debug struct {
		Body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal(body, &debug); err != nil {
		return fmt.Errorf("decode provider request: %w", err)
	}
	if strings.Contains(string(body), billingAttributionPrefix) {
		return fmt.Errorf("the Chat backend still received the billing attribution line: %s", truncateString(string(body), 800))
	}
	if len(debug.Body.Messages) < 2 || debug.Body.Messages[0].Role != "system" {
		return fmt.Errorf("the system prompt beside the line did not arrive first: %s", truncateString(string(body), 800))
	}
	text, err := chatMessageText(debug.Body.Messages[0].Content)
	if err != nil {
		return err
	}
	if text != billingAttributionPrompt {
		return fmt.Errorf("the system prompt beside the line changed: got %q, want %q", text, billingAttributionPrompt)
	}
	return nil
}

// chatMessageText reads a Chat message's content, which is a string when the
// message holds one plain text part and an array of parts otherwise.
func chatMessageText(content json.RawMessage) (string, error) {
	var text string
	if err := json.Unmarshal(content, &text); err == nil {
		return text, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return "", fmt.Errorf("decode Chat message content: %w", err)
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type == "text" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, ""), nil
}

func testBillingAttributionAnthropicBackend(
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

	sessionID := fmt.Sprintf("billing-attribution-anthropic-%d", time.Now().UnixNano())
	body, err := sendProtocolMatrixRequestWithHeaders(
		ctx, session, "/v1/messages",
		billingAttributionRequest("MoM", protocolCodecAnthropicProbe, false),
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
	if err := validateAnthropicBackendKeptBillingAttribution(forwarded); err != nil {
		return fmt.Errorf("buffered dispatch: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"line_carried": true})
	}
	return nil
}

// validateAnthropicBackendKeptBillingAttribution reads the Messages body the
// Anthropic backend recorded. The line must still be system[0], byte for
// byte, with the system prompt after it.
func validateAnthropicBackendKeptBillingAttribution(body []byte) error {
	var debug struct {
		Body struct {
			System []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"system"`
		} `json:"body"`
	}
	if err := json.Unmarshal(body, &debug); err != nil {
		return fmt.Errorf("decode provider request: %w", err)
	}
	if len(debug.Body.System) != 2 ||
		debug.Body.System[0].Text != billingAttributionLine ||
		debug.Body.System[1].Text != billingAttributionPrompt {
		return fmt.Errorf("the Anthropic backend did not receive the line as system[0] with the prompt after it: %s", truncateString(string(body), 800))
	}
	return nil
}
