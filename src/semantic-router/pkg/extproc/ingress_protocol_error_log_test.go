package extproc

import (
	"fmt"
	"strings"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A 400 at the protocol boundary has to leave something an operator can act
// on. On 2026-09-03 a Claude Code turn was refused with "request JSON contains
// a non-canonical field" and the cell wrote no application log line at all:
// the entire record of the rejection was an Envoy access-log row carrying a
// status and nothing else. The member name is already in the cause; it has to
// reach a log line, because the body it came from is the user's conversation
// and is never stored.
func TestIngressProtocolRejectionIsLoggedWithTheMember(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "rt_3f9f829b-029", SourceFormat: llmprotocol.AnthropicMessagesV1}
	cause := fmt.Errorf("field %q: unknown field %q", "metadata", "previous_message_id")
	recordIngressProtocolError(ctx, llmprotocol.NewError(
		llmprotocol.ErrorInvalidRequest, "invalid_json",
		"request JSON contains a non-canonical field", cause,
	))

	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("an ingress rejection wrote %d log lines, want exactly 1", len(entries))
	}
	fields := entries[0].ContextMap()
	for name, want := range map[string]string{
		"event":      "ingress_request_refused",
		"request_id": "rt_3f9f829b-029",
		"code":       "invalid_json",
		"category":   string(llmprotocol.ErrorInvalidRequest),
		"format":     string(llmprotocol.AnthropicMessagesV1),
	} {
		if got, _ := fields[name].(string); got != want {
			t.Fatalf("field %q = %q, want %q", name, got, want)
		}
	}
	detail, _ := fields["detail"].(string)
	if !strings.Contains(detail, "previous_message_id") {
		t.Fatalf("detail %q does not name the offending member", detail)
	}
}

// An unsupported-feature refusal has to name the block as well as the feature.
// On 2026-09-05 a Claude Code turn was refused with unsupported_citations and
// the line said only that, so nothing told an operator which of hundreds of
// blocks in the conversation carried the member. The refused body is the
// user's conversation and is never stored, so the block type, its index and
// the field path have to travel on the error itself.
func TestIngressUnsupportedFeatureRefusalNamesTheBlock(t *testing.T) {
	logs := captureLogs(t)
	body := []byte(`{"model":"client-model","max_tokens":32,"messages":[{"role":"user",` +
		`"content":[{"type":"text","text":"look"},` +
		`{"type":"document","source":{"type":"file","file_id":"file_1"},"title":"Reference"}]}]}`)
	_, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body)
	if err == nil {
		t.Fatal("an unsupported document title was accepted")
	}
	recordIngressProtocolError(&RequestContext{
		RequestID: "rt_ffae7c99-26e", SourceFormat: llmprotocol.AnthropicMessagesV1,
	}, err)

	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("an ingress rejection wrote %d log lines, want exactly 1", len(entries))
	}
	fields := entries[0].ContextMap()
	if code, _ := fields["code"].(string); code != "unsupported_content_title" {
		t.Fatalf("code = %q, want unsupported_content_title", code)
	}
	detail, _ := fields["detail"].(string)
	for _, want := range []string{"content block 1", `type "document"`, `"content.title"`} {
		if !strings.Contains(detail, want) {
			t.Fatalf("detail %q does not carry %s", detail, want)
		}
	}
}

// An error the protocol layer did not raise still gets a line, so a rejection
// is never silent, and it claims no cause it cannot vouch for.
func TestIngressProtocolRejectionLogsAnUntypedError(t *testing.T) {
	logs := captureLogs(t)
	recordIngressProtocolError(&RequestContext{RequestID: "rt_1"}, fmt.Errorf("boom"))

	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("an ingress rejection wrote %d log lines, want exactly 1", len(entries))
	}
	fields := entries[0].ContextMap()
	if code, _ := fields["code"].(string); code != "invalid_request" {
		t.Fatalf("code = %q, want invalid_request", code)
	}
	if _, present := fields["detail"]; present {
		t.Fatal("an untyped error must not be reported as a protocol cause")
	}
}

func captureLogs(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.DebugLevel)
	restore := zap.ReplaceGlobals(zap.New(core))
	t.Cleanup(restore)
	return logs
}
