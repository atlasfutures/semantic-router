package llmprotocol

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// toolUsageContract builds a tool description of at least minBytes out of
// plain sentences. It has the shape a client sends when a description carries
// the tool's whole usage contract: what the tool does, when to reach for it,
// and what each argument means.
func toolUsageContract(minBytes int) string {
	sentences := []string{
		"Use this tool to read a file from the workspace before editing it.",
		"The path argument must be absolute and must name a file that exists.",
		"Prefer this tool over a shell command that prints the file contents.",
		"Do not call this tool twice for the same path within a single turn.",
		"When the file is longer than the offset window, page through it.",
	}
	var builder strings.Builder
	for index := 0; builder.Len() < minBytes; index++ {
		builder.WriteString(sentences[index%len(sentences)])
		builder.WriteByte('\n')
	}
	return builder.String()
}

// TestRealisticToolDeclarationsSurviveValidation is the ask.
//
// Coding-agent clients declare a handful of tools, and one description in that
// set carries a full usage contract that runs past 16 KiB. The default limit
// is 16 KiB (policy.go:90), and validateRequestTool (validate.go:238) answers
// one over-long description by returning tool_text_limit for the request, so
// the whole conversation is refused over a single verbose declaration.
func TestRealisticToolDeclarationsSurviveValidation(t *testing.T) {
	request := validSemanticRequest()
	request.Tools = []Tool{
		{
			Name:        "read_file",
			Description: toolUsageContract(24 << 10),
			InputSchema: json.RawMessage(`{"type":"object"}`),
		},
		{
			Name:        "list_files",
			Description: "List the files under a directory.",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		},
	}
	if err := ValidateRequest(request, DefaultPolicy().Limits); err != nil {
		t.Fatalf("a turn declaring one 24 KiB tool description was refused: %v", err)
	}
}

// TestToolDescriptionLimitIs16KiBToday pins the present behaviour so a change
// to the default, or to the refusal it produces, shows up in the diff rather
// than only in the test above.
func TestToolDescriptionLimitIs16KiBToday(t *testing.T) {
	limits := DefaultPolicy().Limits
	if limits.ToolDescriptionBytes != 16<<10 {
		t.Fatalf("default ToolDescriptionBytes = %d, want the recorded 16384", limits.ToolDescriptionBytes)
	}

	marker := "tool-contract-marker"
	description := marker + strings.Repeat("d", limits.ToolDescriptionBytes-len(marker))
	request := validSemanticRequest()
	request.Tools = []Tool{{
		Name:        "read_file",
		Description: description,
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}}
	if err := ValidateRequest(request, limits); err != nil {
		t.Fatalf("a description of exactly the limit was refused: %v", err)
	}

	request.Tools[0].Description = description + "d"
	err := ValidateRequest(request, limits)
	requireLLMProtocolErrorCode(t, err, "tool_text_limit")

	var protocolError *ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("returned %T, want a protocol error", err)
	}
	if strings.Contains(protocolError.Message, marker) {
		t.Fatal("the refusal repeats the description it rejected")
	}
	if protocolError.Parameter != "" || protocolError.Cause != nil {
		t.Fatalf(
			"the refusal names parameter %q and carries cause %v; today it names neither, so a caller cannot tell which tool was too long",
			protocolError.Parameter,
			protocolError.Cause,
		)
	}
}
