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
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func toolRequest(tools ...llmprotocol.Tool) *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Tools:      tools,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "rename the helper"}},
		}},
	}
}

// The flag is off by default, so a deployment that has not opted in encodes
// exactly the bytes it encoded before tools were readable at all. One byte of
// drift here voids the encoder calibration.
func TestToolNamesAreAbsentUnlessAskedFor(t *testing.T) {
	t.Parallel()
	turns, err := ProjectTurns(
		toolRequest(llmprotocol.Tool{Name: "Edit"}, llmprotocol.Tool{Name: "Bash"}),
		TurnOptions{},
	)
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	if turns[0].Text != "rename the helper" {
		t.Fatalf("turn text = %q, want the turn alone", turns[0].Text)
	}
}

// Names only, in declaration order, with the exact prefix the encoder probe
// measured. The selector has been shown this sequence and no other.
func TestToolNamesFoldIntoTheFirstUserTurn(t *testing.T) {
	t.Parallel()
	turns, err := ProjectTurns(
		toolRequest(
			llmprotocol.Tool{Name: "Edit"},
			llmprotocol.Tool{Name: "Bash"},
			llmprotocol.Tool{Name: "Grep"},
		),
		TurnOptions{IncludeToolNames: true},
	)
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	want := "[available tools] Edit, Bash, Grep\n\nrename the helper"
	if turns[0].Text != want {
		t.Fatalf("turn text = %q, want %q", turns[0].Text, want)
	}
}

// The schemas are the whole point of the measurement: they cost 3,140 tokens
// and moved 40.4 percent of decisions, level with the reject bar. Nothing but
// the name may reach the encoder.
func TestToolSchemasNeverReachTheEncoder(t *testing.T) {
	t.Parallel()
	turns, err := ProjectTurns(
		toolRequest(llmprotocol.Tool{
			Name:        "Edit",
			Description: "Replace one exact string in a file with another.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`),
		}),
		TurnOptions{IncludeToolNames: true},
	)
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	for _, leaked := range []string{"Replace one exact", "file_path", "input_schema", "properties"} {
		if strings.Contains(turns[0].Text, leaked) {
			t.Fatalf("turn text leaked %q: %s", leaked, turns[0].Text)
		}
	}
	if !strings.HasPrefix(turns[0].Text, "[available tools] Edit\n\n") {
		t.Fatalf("turn text = %q, want the name alone", turns[0].Text)
	}
}

// An empty or whitespace name carries no signal and would render as a stray
// separator, which is a sequence the selector has never been shown.
func TestToolNamesSkipUnnamedTools(t *testing.T) {
	t.Parallel()
	for name, tools := range map[string][]llmprotocol.Tool{
		"all unnamed": {{Name: ""}, {Name: "   "}},
		"no tools":    {},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			turns, err := ProjectTurns(toolRequest(tools...), TurnOptions{IncludeToolNames: true})
			if err != nil {
				t.Fatalf("ProjectTurns() error = %v", err)
			}
			if turns[0].Text != "rename the helper" {
				t.Fatalf("turn text = %q, want no tool list at all", turns[0].Text)
			}
		})
	}
	turns, err := ProjectTurns(
		toolRequest(llmprotocol.Tool{Name: ""}, llmprotocol.Tool{Name: "Bash"}),
		TurnOptions{IncludeToolNames: true},
	)
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	if turns[0].Text != "[available tools] Bash\n\nrename the helper" {
		t.Fatalf("turn text = %q, want only the named tool", turns[0].Text)
	}
}

// Tools sit at the front of the folded prefix, ahead of system text. The
// order inside that prefix is part of what the probe encoded.
func TestToolNamesPrecedeSystemText(t *testing.T) {
	t.Parallel()
	request := toolRequest(llmprotocol.Tool{Name: "Edit"})
	request.Instructions = []llmprotocol.InstructionBlock{{
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "be terse"}},
	}}
	turns, err := ProjectTurns(request, TurnOptions{
		IncludeToolNames:  true,
		IncludeSystemText: true,
	})
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	want := "[available tools] Edit\n\nbe terse\n\nrename the helper"
	if turns[0].Text != want {
		t.Fatalf("turn text = %q, want %q", turns[0].Text, want)
	}
}

// A tool the source API runs itself -- web search, the advisor -- is
// identified by its type and usually states no name at all. Keying the
// projection on Name dropped every one of them while the contract said they
// were included, so the selector could not tell a turn with web search from
// one without it.
func TestServerToolsAreEncodedByTheirIdentity(t *testing.T) {
	t.Parallel()
	turns, err := ProjectTurns(
		toolRequest(
			llmprotocol.Tool{Name: "Edit"},
			llmprotocol.Tool{Type: "web_search_20250305"},
			llmprotocol.Tool{Type: "custom", Name: "Bash"},
		),
		TurnOptions{IncludeToolNames: true},
	)
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	want := toolNamesPrefix + "Edit, web_search_20250305, Bash"
	if !strings.Contains(turns[0].Text, want) {
		t.Fatalf("turn text = %q, want it to carry %q", turns[0].Text, want)
	}
}

// A tool with neither a name nor a type identifies nothing, and rendering an
// empty entry would make two different tool sets encode identically.
func TestUnidentifiableToolsAreDropped(t *testing.T) {
	t.Parallel()
	turns, err := ProjectTurns(
		toolRequest(llmprotocol.Tool{Name: "Edit"}, llmprotocol.Tool{}),
		TurnOptions{IncludeToolNames: true},
	)
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	if want := toolNamesPrefix + "Edit\n\nrename the helper"; turns[0].Text != want {
		t.Fatalf("turn text = %q, want %q", turns[0].Text, want)
	}
}
