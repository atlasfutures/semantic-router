package llmprotocol

import (
	"encoding/json"
	"strings"
	"testing"
)

// The dev cell refused three Claude Code turns on 2026-09-04 with
// tool_text_limit, a 400 the proxy has no fallback for, so it reached the user
// as an API error. Nothing about those requests was malformed: the limit was
// simply smaller than a tool declaration the client legitimately sends.
//
// Captured from claude-cli 2.1.260 against a local recording stub, one
// interactive turn, no MCP servers configured:
//
//	32 tools, 140,976 B body, 75,754 B of descriptions, 48,051 B of schemas
//	largest description  Artifact        20,706 B   (schema 17,161 B)
//	next largest         Monitor          6,174 B
//	longest tool name    EnterPlanMode       16 B
//
// The same client run headless (claude -p) declares 28 tools and tops out at
// 6,174 B, which is why the -p end-to-end runs never reproduced it: the four
// interactive-only tools include Artifact.
//
// A tool declaration is client-authored text that the router only forwards. It
// is bounded already by BodyBytes (64 MiB) and by what the upstream provider
// accepts, so the router has no reason to hold a ceiling below either. These
// tests pin a real declaration as acceptable and keep the refusal reachable
// above the new ceiling.
const (
	observedArtifactToolDescriptionBytes = 20706
	observedArtifactToolSchemaBytes      = 17161
)

func toolWithDescriptionBytes(name string, descriptionBytes int) Tool {
	return Tool{
		Name:        name,
		Description: strings.Repeat("d", descriptionBytes),
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}
}

func TestDefaultLimitsAdmitRealClaudeCodeToolDeclaration(t *testing.T) {
	schemaPadding := strings.Repeat("s", observedArtifactToolSchemaBytes-len(`{"type":"object","description":""}`))
	request := validSemanticRequest()
	request.Tools = []Tool{{
		Name:        "Artifact",
		Description: strings.Repeat("d", observedArtifactToolDescriptionBytes),
		InputSchema: json.RawMessage(`{"type":"object","description":"` + schemaPadding + `"}`),
	}}

	if err := ValidateRequest(request, DefaultPolicy().Limits); err != nil {
		t.Fatalf("the largest tool claude-cli 2.1.260 declares was refused: %v", err)
	}
}

func TestDefaultLimitsAdmitAWholeClaudeCodeToolSet(t *testing.T) {
	request := validSemanticRequest()
	request.Tools = []Tool{
		toolWithDescriptionBytes("Artifact", observedArtifactToolDescriptionBytes),
		toolWithDescriptionBytes("Monitor", 6174),
		toolWithDescriptionBytes("SendMessage", 4166),
		toolWithDescriptionBytes("EnterPlanMode", 4011),
	}

	if err := ValidateRequest(request, DefaultPolicy().Limits); err != nil {
		t.Fatalf("a captured claude-cli tool set was refused: %v", err)
	}
}

func TestToolDescriptionLimitStillRefusesAboveTheCeiling(t *testing.T) {
	limits := DefaultPolicy().Limits
	request := validSemanticRequest()
	request.Tools = []Tool{toolWithDescriptionBytes("Artifact", limits.ToolDescriptionBytes+1)}

	err := ValidateRequest(request, limits)
	requireLLMProtocolErrorCode(t, err, "tool_text_limit")
}

func TestToolNameLimitStillRefusesAboveTheCeiling(t *testing.T) {
	limits := DefaultPolicy().Limits
	request := validSemanticRequest()
	request.Tools = []Tool{toolWithDescriptionBytes(strings.Repeat("n", limits.ToolNameBytes+1), 16)}

	err := ValidateRequest(request, limits)
	requireLLMProtocolErrorCode(t, err, "tool_text_limit")
}

func TestToolCountLimitAdmitsALargeMCPToolSetAndStillRefusesAboveIt(t *testing.T) {
	limits := DefaultPolicy().Limits
	request := validSemanticRequest()

	request.Tools = make([]Tool, limits.Tools)
	for i := range request.Tools {
		request.Tools[i] = toolWithDescriptionBytes("mcp__server__tool_"+itoa(i), 16)
	}
	if err := ValidateRequest(request, limits); err != nil {
		t.Fatalf("a tool set at the count limit was refused: %v", err)
	}

	request.Tools = append(request.Tools, toolWithDescriptionBytes("mcp__server__one_too_many", 16))
	requireLLMProtocolErrorCode(t, ValidateRequest(request, limits), "tools_limit")
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	digits := make([]byte, 0, 8)
	for value > 0 {
		digits = append([]byte{byte('0' + value%10)}, digits...)
		value /= 10
	}
	return string(digits)
}
