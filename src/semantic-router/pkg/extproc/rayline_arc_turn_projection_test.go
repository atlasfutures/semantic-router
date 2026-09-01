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
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The selector reads the decoded request, so an ARC episode must never be
// prepared from the wire body or from SelectionContext.Query.
func TestProjectRaylineARCTurnsReadsTheDecodedRequest(t *testing.T) {
	reqCtx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		// A body that no longer matches the decoded request. The projection
		// must not look at it.
		ProtocolEnvelope: llmprotocol.Envelope{Request: []byte(`{"messages":[]}`)},
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "ask"}},
			}},
		},
	}

	turns, err := (&OpenAIRouter{}).projectRaylineARCTurns(reqCtx, raylinearc.TurnOptions{})
	if err != nil {
		t.Fatalf("projectRaylineARCTurns() error = %v", err)
	}
	if len(turns) != 1 || turns[0].Role != "user" || turns[0].Text != "ask" {
		t.Fatalf("turns = %+v, want one user turn carrying the decoded text", turns)
	}
}

// An unknown content kind must fail the episode closed and answer 503. A 503
// says this router cannot serve the request; the alternative -- stripping the
// block -- would route a trained selector on a truncated conversation.
func TestUnknownContentKindFailsClosedAndReadsAs503(t *testing.T) {
	reqCtx := &RequestContext{
		SemanticRequest: &llmprotocol.Request{
			Generation: 1,
			Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentKind("future_kind_from_a_later_release"),
				}},
			}},
		},
	}

	_, err := (&OpenAIRouter{}).projectRaylineARCTurns(reqCtx, raylinearc.TurnOptions{})
	code := raylinearc.TurnNormalizationErrorCode(err)
	if code != "unknown_item" {
		t.Fatalf("turn projection error code = %q, want %q", code, "unknown_item")
	}

	// buildRaylineARCSelectionContext reports this as a preparation failure,
	// and the decision adapter classifies preparation failures. Anything that
	// is not contention answers 503; see pkg/apiserver/route_decision.go.
	failure := prepareFailureError("turns_" + code)
	if errors.Is(failure, routerruntime.ErrRouteDecisionContended) {
		t.Fatalf("a fail-closed turn projection was classified as contention")
	}
}

// The two contention classes stay contention: they answer 429 with a
// Retry-After rather than 503. Waiting fixes them; a rejected turn is not
// fixed by waiting.
func TestEpisodeContentionStillReadsAs429(t *testing.T) {
	for _, failure := range []string{"episode_timeout", "episode_capacity"} {
		if !errors.Is(
			prepareFailureError(failure),
			routerruntime.ErrRouteDecisionContended,
		) {
			t.Fatalf("preparation failure %q lost its contention class", failure)
		}
	}
}
