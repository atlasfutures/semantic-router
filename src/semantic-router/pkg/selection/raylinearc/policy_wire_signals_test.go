/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The fixtures under testdata/wire_signals are request bodies captured from
// the released Claude Code and codex binaries against a local stub upstream,
// sanitized (see each file's provenance). Every detector is checked against
// every captured shape: it fires on its own and on no other.

type wireFixtureBody struct {
	System   json.RawMessage   `json:"system"`
	Messages []json.RawMessage `json:"messages"`
	Input    []json.RawMessage `json:"input"`
}

func loadWireFixture(t *testing.T, name string) wireFixtureBody {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "wire_signals", name))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Body wireFixtureBody `json:"body"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture.Body
}

var (
	claudeCodeFixtures = []string{
		"claude_code_2.1.280_main_turn.json",
		"claude_code_2.1.280_title.json",
		"claude_code_2.1.280_compaction_request.json",
		"claude_code_2.1.280_post_compaction.json",
	}
	codexFixtures = []string{
		"codex_0.154.0_main_turn.json",
		"codex_0.154.0_compaction_request.json",
		"codex_0.154.0_post_compaction.json",
		"codex_0.154.0_post_compaction_next.json",
	}
)

func TestClaudeCodeWireShapesAreEachRecognisedByExactlyTheirDetector(t *testing.T) {
	for _, name := range claudeCodeFixtures {
		body := loadWireFixture(t, name)
		title := IsClaudeCodeTitleRequest(body.System)
		summarize := IsClaudeCodeCompactionRequest(body.Messages)
		continued := ClaudeCodeCompactionSummaryDigest(body.Messages) != ""
		subagent := ClaudeCodeSubagentClaim(body.System)
		want := map[string][3]bool{
			"claude_code_2.1.280_main_turn.json":          {false, false, false},
			"claude_code_2.1.280_title.json":              {true, false, false},
			"claude_code_2.1.280_compaction_request.json": {false, true, false},
			"claude_code_2.1.280_post_compaction.json":    {false, false, true},
		}[name]
		if got := [3]bool{title, summarize, continued}; got != want || subagent {
			t.Errorf("%s: title/summarize/continued = %v subagent %v, want %v", name, got, subagent, want)
		}
	}
}

func TestCodexWireShapesAreEachRecognisedByExactlyTheirDetector(t *testing.T) {
	digests := map[string]string{}
	for _, name := range codexFixtures {
		body := loadWireFixture(t, name)
		digests[name] = CodexCompactionSummaryDigest(body.Input)
		summarize := IsCodexCompactionRequest(body.Input)
		if summarize != (name == "codex_0.154.0_compaction_request.json") {
			t.Errorf("%s: compaction request = %v", name, summarize)
		}
	}
	if digests["codex_0.154.0_main_turn.json"] != "" || digests["codex_0.154.0_compaction_request.json"] != "" {
		t.Fatalf("a pre-compaction request read as compacted: %v", digests)
	}
	post, next := digests["codex_0.154.0_post_compaction.json"], digests["codex_0.154.0_post_compaction_next.json"]
	if post == "" || !isLowerHex64(post) {
		t.Fatalf("the first post-compaction request was not read as compacted: %q", post)
	}
	// Every later request of that context opens with the same summary, so
	// it is one compaction, not a new one per request.
	if next != post {
		t.Fatalf("the next request of the compacted context read a different summary: %q vs %q", next, post)
	}
}

func TestClaudeCodeDirectiveCountsOnlyAsTheLastUserBlock(t *testing.T) {
	directive := jsonString(ClaudeCodeSummarizeDirective + "\n\n- Do NOT use tools.")
	cases := []struct {
		name     string
		messages []string
		want     bool
	}{
		{"last text block", []string{`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"ok"},{"type":"text","text":` + directive + `}]}`}, true},
		{"string content", []string{`{"role":"user","content":` + directive + `}`}, true},
		{"quoted earlier in the transcript", []string{`{"role":"user","content":` + directive + `}`, `{"role":"assistant","content":"done"}`, `{"role":"user","content":"next"}`}, false},
		{"not the last block", []string{`{"role":"user","content":[{"type":"text","text":` + directive + `},{"type":"text","text":"and also"}]}`}, false},
		{"assistant message", []string{`{"role":"assistant","content":` + directive + `}`}, false},
		{"mid-text", []string{`{"role":"user","content":` + jsonString("please: "+ClaudeCodeSummarizeDirective) + `}`}, false},
	}
	for _, c := range cases {
		messages, _ := rawMessages(t, c.messages...)
		if got := IsClaudeCodeCompactionRequest(messages); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
	if IsClaudeCodeTitleRequest(json.RawMessage(`[{"type":"text","text":"be brief"}]`)) ||
		!IsClaudeCodeTitleRequest(json.RawMessage(jsonString(ClaudeCodeTitlePrompt+" Use five words."))) {
		t.Fatal("title prompt detection is wrong on a plain system or a string system")
	}
}

func TestCodexSummaryCountsOnlyInTheCompactedBase(t *testing.T) {
	summary := jsonString(CodexSummaryPrefix + "the summary")
	base := `{"type":"message","role":"user","content":[{"type":"input_text","text":` + summary + `}]}`
	reasoning := `{"type":"reasoning","summary":[],"encrypted_content":"x"}`
	assistant := `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}`
	user := `{"type":"message","role":"user","content":"go on"}`
	cases := []struct {
		name  string
		items []string
		want  bool
	}{
		{"leading summary", []string{user, base, reasoning}, true},
		{"after model output", []string{user, reasoning, base}, false},
		{"after an assistant message", []string{assistant, base}, false},
		{"two parts", []string{`{"type":"message","role":"user","content":[{"type":"input_text","text":` + summary + `},{"type":"input_text","text":"x"}]}`}, false},
		{"developer role", []string{`{"type":"message","role":"developer","content":` + summary + `}`}, false},
	}
	for _, c := range cases {
		items, _ := rawMessages(t, c.items...)
		if got := CodexCompactionSummaryDigest(items) != ""; got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
	// A later compaction keeps the earlier summary among the user messages
	// it retains; the last summary in the base is the current one.
	first := `{"type":"message","role":"user","content":` + jsonString(CodexSummaryPrefix+"first") + `}`
	second := `{"type":"message","role":"user","content":` + jsonString(CodexSummaryPrefix+"second") + `}`
	only, _ := rawMessages(t, second)
	both, _ := rawMessages(t, first, second, reasoning)
	if CodexCompactionSummaryDigest(both) != CodexCompactionSummaryDigest(only) {
		t.Fatal("the current summary is not the last one in the compacted base")
	}
}
