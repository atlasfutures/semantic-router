package llmprotocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// Accept-by-default leaves a small set of refusals standing: malformed JSON,
// the union rules and the hard limits. Each of those still has to say where it
// happened, in both places anyone looks.
//
// The log line is for the operator: a refused body is never stored, and a
// Claude Code turn declares thirty-two tools, so "some tool was too large" is
// not something anyone can act on. The client body is for the client: the
// 2.1.260 client retries without a field only when the 400 body names it.

func TestToolRefusalsNameTheToolAndTheField(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	limits := DefaultPolicy().Limits
	limits.ToolDescriptionBytes = 16
	limits.ToolNameBytes = 8
	for _, test := range []struct {
		name    string
		tools   []Tool
		code    string
		message []string
	}{
		{
			name:    "callable tool without a name",
			tools:   []Tool{{InputSchema: schema}},
			code:    "invalid_tool",
			message: []string{"tool 0", `"tools.name"`},
		},
		{
			name:    "callable tool without a schema",
			tools:   []Tool{{Name: "lookup"}},
			code:    "invalid_tool",
			message: []string{`tool 0 named "lookup"`, `"tools.input_schema"`},
		},
		{
			name:    "description over the limit",
			tools:   []Tool{{Name: "lookup", InputSchema: schema, Description: strings.Repeat("x", 64)}},
			code:    "tool_text_limit",
			message: []string{`tool 0 named "lookup"`, `"tools.description"`, "64 bytes, limit 16"},
		},
		{
			name: "duplicate tool name",
			tools: []Tool{
				{Name: "lookup", InputSchema: schema},
				{Name: "lookup", InputSchema: schema},
			},
			code:    "duplicate_tool",
			message: []string{`tool 1 named "lookup"`, `"tools.name"`},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := validateRequestTools(test.tools, limits)
			assertToolRefusalNames(t, err, test.code, test.message)
		})
	}
}

func assertToolRefusalNames(t *testing.T, err error, code string, message []string) {
	t.Helper()
	protocolError, ok := err.(*ProtocolError)
	if !ok {
		t.Fatalf("validateRequestTools() error = %v, want a protocol error", err)
	}
	if protocolError.Code != code {
		t.Fatalf("code = %q, want %q", protocolError.Code, code)
	}
	for _, want := range message {
		// The client body carries the message and nothing else, so the message
		// is what a client can repair its own request from.
		if !strings.Contains(protocolError.Message, want) {
			t.Fatalf("client message %q does not carry %s", protocolError.Message, want)
		}
	}
	// The log line carries the cause, which must name the same place.
	if protocolError.Cause == nil || !strings.Contains(protocolError.Cause.Error(), "field") {
		t.Fatalf("cause = %v, want a field-naming cause", protocolError.Cause)
	}
	if protocolError.Parameter == "" {
		t.Fatal("refusal names no parameter")
	}
}

// A refusal must carry contract text alone. A field path and an index are safe
// to return; the value a client wrote is the user's conversation and is not.
func TestFieldRefusalsCarryNoRequestValues(t *testing.T) {
	secret := "the-user-wrote-this"
	limits := DefaultPolicy().Limits
	limits.ToolDescriptionBytes = 4
	_, _, err := validateRequestTools([]Tool{{
		Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), Description: secret,
	}}, limits)
	protocolError, ok := err.(*ProtocolError)
	if !ok {
		t.Fatalf("validateRequestTools() error = %v, want a protocol error", err)
	}
	if strings.Contains(protocolError.Error(), secret) {
		t.Fatalf("refusal %q carries a request value", protocolError.Error())
	}
}

func TestServerToolsNeedNeitherNameNorSchema(t *testing.T) {
	limits := DefaultPolicy().Limits
	tools := []Tool{{Type: "web_search_20250305"}, {Type: "advisor_20260301"}}
	if _, _, err := validateRequestTools(tools, limits); err != nil {
		t.Fatalf("a server tool declaration was refused: %v", err)
	}
}

// A refusal is bounded. The tool location repeats the declared name so an
// operator can find the tool among the thirty-two a turn declares, but a name
// over its own limit is the value that broke the limit: repeating it put an
// unbounded string the client wrote into the client body and into the
// ingress_request_refused line. A 5012-byte name produced a 10217-byte
// refusal. The index still finds the tool.
func TestAToolNameOverItsLimitIsNotRepeated(t *testing.T) {
	limits := DefaultPolicy().Limits
	name := strings.Repeat("n", limits.ToolNameBytes+16)
	_, _, err := validateRequestTools(
		[]Tool{{Name: name, InputSchema: json.RawMessage(`{"type":"object"}`)}}, limits,
	)
	var protocolError *ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("validateRequestTools() error = %v, want a protocol error", err)
	}
	if protocolError.Code != "tool_text_limit" {
		t.Fatalf("code = %q, want tool_text_limit", protocolError.Code)
	}
	if strings.Contains(protocolError.Error(), name) {
		t.Fatal("the refusal repeats the name that broke the limit")
	}
	if len(protocolError.Error()) > 512 {
		t.Fatalf("refusal is %d bytes, so it grows with the request", len(protocolError.Error()))
	}
	// It still says which tool, which member, and both counts.
	for _, want := range []string{"tool 0", `"tools.name"`, "1040 bytes, limit 1024"} {
		if !strings.Contains(protocolError.Message, want) {
			t.Fatalf("client message %q does not carry %s", protocolError.Message, want)
		}
	}
}

// A name within its limit is still repeated: that is what makes a refusal
// findable in a turn declaring thirty-two tools.
func TestAToolNameWithinItsLimitIsStillNamed(t *testing.T) {
	limits := DefaultPolicy().Limits
	limits.ToolDescriptionBytes = 8
	_, _, err := validateRequestTools([]Tool{{
		Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`),
		Description: strings.Repeat("d", 32),
	}}, limits)
	var protocolError *ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("validateRequestTools() error = %v, want a protocol error", err)
	}
	if !strings.Contains(protocolError.Message, `tool 0 named "lookup"`) {
		t.Fatalf("client message %q does not name the tool", protocolError.Message)
	}
}
