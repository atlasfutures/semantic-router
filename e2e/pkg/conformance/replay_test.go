package conformance

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseReplayScript(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "provider-response.sse"), "data: {}\n\n")
	write(t, filepath.Join(dir, "provider-response.json"), `{}`)

	const script = `schema_version: protocol-conformance-replay-v1
expect:
  method: POST
  path: /v1/messages
  headers:
    anthropic-version: "2023-06-01"
steps:
  - {kind: status, status: 200, headers: {content-type: text/event-stream}}
  - {kind: sse, file: provider-response.sse, chunk_bytes: 7}
  - {kind: delay, millis: 25}
  - {kind: disconnect}
`

	got, err := parseReplayScript([]byte(script), dir)
	if err != nil {
		t.Fatalf("parseReplayScript() error = %v", err)
	}
	if got.Expect.Headers["anthropic-version"] != "2023-06-01" {
		t.Errorf("expect.headers = %v", got.Expect.Headers)
	}
	if len(got.Steps) != 4 {
		t.Fatalf("steps = %d, want 4", len(got.Steps))
	}
	if got.Steps[1].Kind != StepSSE || got.Steps[1].ChunkBytes != 7 {
		t.Errorf("step 1 = %+v", got.Steps[1])
	}
	if got.Steps[3].Kind != StepDisconnect {
		t.Errorf("step 3 = %+v", got.Steps[3])
	}
}

func TestParseReplayScriptFailures(t *testing.T) {
	tests := []struct {
		name    string
		script  string
		wantErr string
	}{
		{
			name:    "wrong schema version",
			script:  "schema_version: replay-v2\nexpect: {method: POST, path: /x}\nsteps: [{kind: disconnect}]\n",
			wantErr: "unsupported schema_version",
		},
		{
			name:    "missing expected path",
			script:  replayHeader + "expect: {method: POST}\nsteps: [{kind: status, status: 200}]\n",
			wantErr: "expect.method and expect.path are required",
		},
		{
			name:    "no steps",
			script:  replayHeader + "expect: {method: POST, path: /x}\nsteps: []\n",
			wantErr: "at least one step is required",
		},
		{
			name:    "unknown step kind",
			script:  replayHeader + "expect: {method: POST, path: /x}\nsteps: [{kind: teleport}]\n",
			wantErr: `unknown kind "teleport"`,
		},
		{
			name:    "first step is not status",
			script:  replayHeader + "expect: {method: POST, path: /x}\nsteps: [{kind: disconnect}]\n",
			wantErr: "the first step must be",
		},
		{
			name: "second status step",
			script: replayHeader + "expect: {method: POST, path: /x}\n" +
				"steps: [{kind: status, status: 200}, {kind: status, status: 500}]\n",
			wantErr: "only the first step may be",
		},
		{
			name: "missing referenced file",
			script: replayHeader + "expect: {method: POST, path: /x}\n" +
				"steps: [{kind: status, status: 200}, {kind: body, file: absent.json}]\n",
			wantErr: `file "absent.json" is not readable`,
		},
		{
			name: "escaping file reference",
			script: replayHeader + "expect: {method: POST, path: /x}\n" +
				"steps: [{kind: status, status: 200}, {kind: body, file: ../secrets.json}]\n",
			wantErr: "must be a plain name inside the case directory",
		},
		{
			name: "chunked buffered body",
			script: replayHeader + "expect: {method: POST, path: /x}\n" +
				"steps: [{kind: status, status: 200}, {kind: body, file: provider-response.json, chunk_bytes: 4}]\n",
			wantErr: "chunk_bytes applies to sse steps only",
		},
		{
			name:    "delay without a duration",
			script:  replayHeader + "expect: {method: POST, path: /x}\nsteps: [{kind: status, status: 200}, {kind: delay}]\n",
			wantErr: "requires a positive millis",
		},
		{
			name:    "status outside the HTTP range",
			script:  replayHeader + "expect: {method: POST, path: /x}\nsteps: [{kind: status, status: 42}]\n",
			wantErr: "is not an HTTP status",
		},
		{
			name: "unknown step field",
			script: replayHeader + "expect: {method: POST, path: /x}\n" +
				"steps: [{kind: status, status: 200, jitter: 3}]\n",
			wantErr: "unknown field",
		},
	}

	dir := t.TempDir()
	write(t, filepath.Join(dir, "provider-response.json"), `{}`)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseReplayScript([]byte(tt.script), dir)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("parseReplayScript() error = %v, want it to contain %q", err, tt.wantErr)
			}
		})
	}
}

// TestLoadReplayScript pins that a case directory's replay.yaml is validated by the
// loader, so DPC-101 never has to guard against a malformed script at run time.
func TestLoadReplayScript(t *testing.T) {
	dir := writeTree(t, caseYAML(nil))
	caseDir := filepath.Join(dir, "unit-01")
	if err := os.MkdirAll(caseDir, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(caseDir, "client-request.json"), `{}`)
	write(t, filepath.Join(caseDir, "expected-provider-request.json"), `{}`)
	write(t, filepath.Join(caseDir, "provider-response.json"), `{"id":"r1"}`)
	write(t, filepath.Join(caseDir, "expected-client-response.json"), `{"id":"r1"}`)
	write(t, filepath.Join(caseDir, fileReplayScript), replayHeader+
		"expect: {method: POST, path: /v1/chat/completions}\n"+
		"steps: [{kind: status, status: 200}, {kind: body, file: provider-response.json}]\n")

	inv, err := Load(dir)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	c, _ := inv.Case("unit-01")
	if c.Fixtures.Replay == nil {
		t.Fatal("replay script was not loaded")
	}
	if c.Fixtures.Replay.Steps[1].File != "provider-response.json" {
		t.Errorf("replay step 1 = %+v", c.Fixtures.Replay.Steps[1])
	}

	write(t, filepath.Join(caseDir, fileReplayScript), replayHeader+
		"expect: {method: POST, path: /v1/chat/completions}\n"+
		"steps: [{kind: status, status: 200}, {kind: sse, file: missing.sse}]\n")
	if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), "is not readable") {
		t.Fatalf("Load() error = %v, want a dangling replay reference error", err)
	}
}

// TestLoadOptionalCompareTuning pins that compare.yaml feeds the resolved comparison.
func TestLoadOptionalCompareTuning(t *testing.T) {
	dir := writeTree(t, caseYAML(nil))
	caseDir := filepath.Join(dir, "unit-01")
	if err := os.MkdirAll(caseDir, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(caseDir, "client-request.json"), `{}`)
	write(t, filepath.Join(caseDir, "expected-provider-request.json"), `{}`)
	write(t, filepath.Join(caseDir, "provider-response.json"), `{}`)
	write(t, filepath.Join(caseDir, "expected-client-response.json"), `{}`)
	write(t, filepath.Join(caseDir, fileCompareTuning), "exclude_extra: [/id]\nvolatile: [/created]\n")

	inv, err := Load(dir)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	c, _ := inv.Case("unit-01")

	cmp, err := c.Comparison(BoundaryProviderRequest)
	if err != nil {
		t.Fatalf("Comparison() error = %v", err)
	}
	if !equalStrings(cmp.Exclude, []string{"/model", "/id"}) {
		t.Errorf("exclusions = %v, want [/model /id]", cmp.Exclude)
	}
}

const replayHeader = "schema_version: " + ReplaySchemaVersion + "\n"
