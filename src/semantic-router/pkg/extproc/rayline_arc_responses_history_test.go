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

package extproc

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A Responses request that carries previous_response_id names its earlier
// turns instead of repeating them. The retained turns reach the provider, but
// only after selection. Without them the selector sees a long session as a
// single question and routes it as one, so these goldens pin that the whole
// conversation reaches the projection in order.
//
// The fixture is separate from the turn-projection goldens on purpose: those
// hold the fork's recorded output and must not move.
type responsesHistoryFixture struct {
	SchemaVersion string                       `json:"schema_version"`
	Cases         []responsesHistoryGoldenCase `json:"cases"`
}

type responsesHistoryGoldenCase struct {
	ID      string                        `json:"id"`
	Stored  []*responseapi.StoredResponse `json:"stored"`
	Request json.RawMessage               `json:"request"`
	Turns   []raylinearc.Turn             `json:"turns"`
}

func TestRaylineARCIncludesRetainedResponsesHistory(t *testing.T) {
	data, err := os.ReadFile("testdata/responses_history_turn_goldens.v1.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture responsesHistoryFixture
	if decodeErr := json.Unmarshal(data, &fixture); decodeErr != nil {
		t.Fatalf("decode fixture: %v", decodeErr)
	}
	if fixture.SchemaVersion != "rayline.arc.responses-history-turns.v1" {
		t.Fatalf("fixture schema = %q", fixture.SchemaVersion)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("fixture carries no cases")
	}

	router := &OpenAIRouter{}
	engine, err := router.protocolEngine()
	if err != nil {
		t.Fatalf("protocolEngine() error = %v", err)
	}
	for _, test := range fixture.Cases {
		t.Run(test.ID, func(t *testing.T) {
			decoded, envelope, _, err := engine.DecodeRequestForMutation(
				llmprotocol.OpenAIResponsesV1,
				test.Request,
			)
			if err != nil {
				t.Fatalf("codec rejected the fixture request: %v", err)
			}
			reqCtx := &RequestContext{
				SourceFormat:     llmprotocol.OpenAIResponsesV1,
				SemanticRequest:  &decoded,
				ProtocolEnvelope: envelope,
				ResponseObjectState: &ResponseObjectState{
					PreviousResponseID:  decoded.PreviousResponseID,
					ConversationHistory: test.Stored,
				},
			}

			turns, err := router.projectRaylineARCTurns(
				reqCtx,
				raylinearc.TurnOptions{},
			)
			if err != nil {
				t.Fatalf("projectRaylineARCTurns() error = %v", err)
			}
			assertTurnsMatch(t, turns, test.Turns)

			// The selector reads a copy. Widening the request the dispatch
			// path owns would send the retained turns to the provider twice.
			if len(decoded.Messages) != len(reqCtx.SemanticRequest.Messages) {
				t.Fatalf("the projection widened the dispatched request")
			}
		})
	}
}

// Materialization is idempotent by ProviderContextApplied. If the dispatch
// path ever ran before selection, the retained turns would already be in the
// request and prepending them again would double the conversation.
func TestRaylineARCDoesNotDoubleAppliedResponsesHistory(t *testing.T) {
	router := &OpenAIRouter{}
	request := &llmprotocol.Request{
		Generation: 1,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "already merged"}},
		}},
	}
	reqCtx := &RequestContext{
		SourceFormat:    llmprotocol.OpenAIResponsesV1,
		SemanticRequest: request,
		ResponseObjectState: &ResponseObjectState{
			PreviousResponseID: "resp_1",
			ConversationHistory: []*responseapi.StoredResponse{{
				ID: "resp_1",
				Input: []responseapi.InputItem{{
					Type: "message", Role: "user",
					Content: json.RawMessage(`"earlier"`),
				}},
			}},
			ProviderContextApplied: true,
		},
	}

	turns, err := router.projectRaylineARCTurns(reqCtx, raylinearc.TurnOptions{})
	if err != nil {
		t.Fatalf("projectRaylineARCTurns() error = %v", err)
	}
	assertTurnsMatch(t, turns, []raylinearc.Turn{
		{Role: "user", Text: "already merged"},
	})
}

func assertTurnsMatch(t *testing.T, got []raylinearc.Turn, want []raylinearc.Turn) {
	t.Helper()
	gotJSON, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal projected turns: %v", err)
	}
	wantJSON, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal golden turns: %v", err)
	}
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("turns\n got: %s\nwant: %s", gotJSON, wantJSON)
	}
}
