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
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

const (
	arcFixtureSchema     = "rayline.arc.token-blocks.v1"
	serializerSourceTag  = "serializer-src-01b692ca14003693"
	qwenModelRevision    = "2fc06364715b967f1860aea9cf38778875588b17"
	qwenVocabularyDigest = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
	qwenEOSID            = 248046
)

type tokenGoldenFixture struct {
	SchemaVersion        string                   `json:"schema_version"`
	Provenance           tokenGoldenProvenance    `json:"provenance"`
	Cases                []tokenGoldenCase        `json:"cases"`
	RecentTailBoundaries []recentTailBoundaryCase `json:"recent_tail_boundaries"`
}

type tokenGoldenProvenance struct {
	SerializerSourceTag string `json:"serializer_source_tag"`
	Serializer          string `json:"serializer"`
	Model               string `json:"model"`
	ModelRevision       string `json:"model_revision"`
	TokenizerSHA256     string `json:"tokenizer_sha256"`
	EOSToken            string `json:"eos_token"`
	EOSTokenID          int    `json:"eos_token_id"`
	AddSpecialTokens    bool   `json:"add_special_tokens"`
	ParseSpecialTokens  bool   `json:"parse_special_tokens"`
}

type tokenGoldenCase struct {
	ID       string              `json:"id"`
	Limits   tokenGoldenLimits   `json:"limits"`
	Turns    []Turn              `json:"turns"`
	Expected tokenGoldenExpected `json:"expected"`
}

type tokenGoldenLimits struct {
	MaxTokens       int `json:"max_tokens"`
	MinRecentTurns  int `json:"min_recent_turns"`
	MinRecentTokens int `json:"min_recent_tokens"`
}

type tokenGoldenExpected struct {
	InputIDs         []int              `json:"input_ids"`
	FullTokens       int                `json:"full_tokens"`
	TruncatedTokens  int                `json:"truncated_tokens"`
	TaskIDs          []int              `json:"task_ids"`
	ContextHeaderIDs []int              `json:"context_header_ids"`
	HistoryBlocks    []tokenGoldenBlock `json:"history_blocks"`
}

type tokenGoldenBlock struct {
	HeaderIDs  []int `json:"header_ids"`
	ContentIDs []int `json:"content_ids"`
}

type recentTailBoundaryCase struct {
	MaxTokens  int   `json:"max_tokens"`
	HeaderIDs  []int `json:"header_ids"`
	ContentIDs []int `json:"content_ids"`
	EOSTokenID int   `json:"eos_token_id"`
	Expected   []int `json:"expected_ids"`
}

func TestTokenBlockGoldenContract(t *testing.T) {
	fixture := readTokenGoldenFixture(t)
	if fixture.SchemaVersion != arcFixtureSchema {
		t.Fatalf("fixture schema = %q, want %q", fixture.SchemaVersion, arcFixtureSchema)
	}
	assertTokenGoldenProvenance(t, fixture.Provenance)
	assertTokenGoldenCases(t, fixture.Cases)
	assertRecentTailBoundaries(t, fixture.RecentTailBoundaries)
}

func assertTokenGoldenProvenance(t *testing.T, provenance tokenGoldenProvenance) {
	t.Helper()
	if provenance.SerializerSourceTag != serializerSourceTag ||
		provenance.ModelRevision != qwenModelRevision ||
		provenance.TokenizerSHA256 != qwenVocabularyDigest ||
		provenance.Serializer != "mtrouter-token-blocks-v2" ||
		provenance.Model != "Qwen/Qwen3.5-0.8B" ||
		provenance.EOSToken != "<|im_end|>" ||
		provenance.EOSTokenID != qwenEOSID ||
		provenance.AddSpecialTokens ||
		provenance.ParseSpecialTokens {
		t.Fatalf("unexpected token fixture provenance: %#v", provenance)
	}
}

func assertTokenGoldenCases(t *testing.T, cases []tokenGoldenCase) {
	t.Helper()
	if len(cases) < 5 {
		t.Fatalf("token fixture has %d cases, want at least 5", len(cases))
	}
	seen := make(map[string]bool, len(cases))
	for _, test := range cases {
		assertTokenGoldenCase(t, test, seen)
	}
	for _, required := range []string{
		"literal_special_unicode",
		"separate_header_content_and_turn_numbers",
		"no_initial_user_uses_empty_task",
		"task_prefix_truncation",
		"recent_turn_tail_truncation",
	} {
		if !seen[required] {
			t.Fatalf("token fixture is missing %q", required)
		}
	}
}

func assertTokenGoldenCase(
	t *testing.T,
	test tokenGoldenCase,
	seen map[string]bool,
) {
	t.Helper()
	if test.ID == "" || seen[test.ID] {
		t.Fatalf("fixture case ID is empty or duplicated: %q", test.ID)
	}
	seen[test.ID] = true
	if test.Limits.MaxTokens < 16 ||
		test.Limits.MinRecentTurns == 0 ||
		test.Limits.MinRecentTokens == 0 {
		t.Fatalf("%s has invalid limits: %#v", test.ID, test.Limits)
	}
	if len(test.Expected.InputIDs) > test.Limits.MaxTokens {
		t.Fatalf("%s exceeds max_tokens", test.ID)
	}
	if test.Expected.FullTokens !=
		len(test.Expected.InputIDs)+test.Expected.TruncatedTokens {
		t.Fatalf("%s has inconsistent token counts", test.ID)
	}
	if !endsWith(test.Expected.InputIDs, qwenEOSID) ||
		!endsWith(test.Expected.TaskIDs, qwenEOSID) {
		t.Fatalf("%s does not preserve EOS boundaries", test.ID)
	}
	if len(test.Expected.ContextHeaderIDs) > 0 &&
		!endsWith(test.Expected.ContextHeaderIDs, qwenEOSID) {
		t.Fatalf("%s context header does not end in EOS", test.ID)
	}
}

func assertRecentTailBoundaries(
	t *testing.T,
	boundaries []recentTailBoundaryCase,
) {
	t.Helper()
	wantBoundaryOutputs := [][]int{
		{},
		{99},
		{11, 99},
		{11, 12, 99},
		{11, 12, 23, 99},
	}
	if len(boundaries) != len(wantBoundaryOutputs) {
		t.Fatalf(
			"recent-tail fixture count = %d, want %d",
			len(boundaries),
			len(wantBoundaryOutputs),
		)
	}
	for index, test := range boundaries {
		if test.MaxTokens != index ||
			!equalInts(test.HeaderIDs, []int{11, 12}) ||
			!equalInts(test.ContentIDs, []int{21, 22, 23}) ||
			test.EOSTokenID != 99 ||
			!equalInts(test.Expected, wantBoundaryOutputs[index]) {
			t.Fatalf("unexpected recent-tail boundary %d: %#v", index, test)
		}
	}
}

func readTokenGoldenFixture(t *testing.T) tokenGoldenFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/token_block_goldens.v1.json")
	if err != nil {
		t.Fatalf("read token fixture: %v", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var fixture tokenGoldenFixture
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode token fixture: %v", err)
	}
	return fixture
}

func endsWith(values []int, value int) bool {
	return len(values) > 0 && values[len(values)-1] == value
}

func equalInts(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
