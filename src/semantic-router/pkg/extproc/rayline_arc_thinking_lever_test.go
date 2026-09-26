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
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	leverWorker   = "glm@default"
	unboundWorker = "qwen@default"
	leverDown     = "Until the next steering instruction, use minimal deliberation."
)

func thinkingLeverDecision(enabled bool) *config.Decision {
	return &config.Decision{
		Name: "arc",
		Algorithm: &config.AlgorithmConfig{
			Type: config.RaylineARCAlgorithmType,
			RaylineARC: &config.RaylineARCAlgorithmConfig{ThinkingLever: &config.RaylineARCThinkingLeverConfig{
				Enabled: enabled,
				Source:  config.RaylineARCThinkingSourceRule,
				Level:   "down",
				Workers: map[string]config.RaylineARCThinkingBindingConfig{leverWorker: {
					Admission:  config.RaylineARCThinkingAdmissionCertified,
					Lever:      "prompt_steering_suffix",
					Emit:       "on_change",
					Placements: []string{"append_tail_user_text", "insert_user_after_tool_run"},
					Levels: []config.RaylineARCThinkingLevelConfig{
						{Level: "none", Rank: 0, ControlSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
						{Level: "down", Rank: -1, Suffix: leverDown, ControlSHA256: "99fca31cf4f665d12f6695cb8c46311be0cc6e4a34829347d4cc9298b24cccf6"},
					},
				}},
			}},
		},
	}
}

type leverEpisode struct {
	t        *testing.T
	store    *raylinearc.MemoryEpisodeStore
	episode  string
	decision *config.Decision
}

func newLeverEpisode(t *testing.T, enabled bool) *leverEpisode {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	return &leverEpisode{t: t, store: store, episode: raylinearc.HashEpisodeID(t.Name()), decision: thinkingLeverDecision(enabled)}
}

// turn runs one request through the lever and finalizes it, returning the
// provider-bound messages and the request context.
func (e *leverEpisode) turn(worker string, messages []llmprotocol.Message, commit bool) ([]llmprotocol.Message, *RequestContext) {
	e.t.Helper()
	return e.turnWithHeaders(worker, messages, commit, nil)
}

func leverMessages() []llmprotocol.Message {
	return []llmprotocol.Message{leverText(llmprotocol.RoleUser, "go")}
}

func (e *leverEpisode) turnWithHeaders(
	worker string,
	messages []llmprotocol.Message,
	commit bool,
	headers map[string]string,
) ([]llmprotocol.Message, *RequestContext) {
	e.t.Helper()
	lease, state, err := e.store.Prepare(context.Background(), e.episode, 2)
	if err != nil {
		e.t.Fatal(err)
	}
	transaction := newRaylineARCEpisodeTransaction(e.store, lease, state, e.episode, time.Minute, nil)
	transaction.markSelection(0, 10)
	if headers == nil {
		headers = map[string]string{}
	}
	ctx := &RequestContext{
		Headers:               headers,
		VSRSelectedDecision:   e.decision,
		RaylineARCDispatch:    &raylinearc.WorkerManifest{ID: worker},
		RaylineARCTransaction: transaction,
	}
	request := &llmprotocol.Request{Model: "m", Messages: append([]llmprotocol.Message(nil), messages...)}
	router := &OpenAIRouter{}
	if _, err := router.applyRaylineARCThinkingLever(request, ctx); err != nil {
		e.t.Fatalf("apply: %v", err)
	}
	// A second call in the same request must not write the items again.
	if changed, err := router.applyRaylineARCThinkingLever(request, ctx); err != nil || changed {
		e.t.Fatalf("second apply changed=%v err=%v", changed, err)
	}
	if commit {
		if err := transaction.commit(context.Background(), ctx); err != nil {
			e.t.Fatalf("commit: %v", err)
		}
	} else if err := transaction.abort(context.Background(), "test"); err != nil {
		e.t.Fatalf("abort: %v", err)
	}
	return request.Messages, ctx
}

func (e *leverEpisode) stored() *raylinearc.EpisodeState {
	e.t.Helper()
	lease, state, err := e.store.Prepare(context.Background(), e.episode, 2)
	if err != nil {
		e.t.Fatal(err)
	}
	_ = e.store.Abort(context.Background(), lease)
	return state
}

func leverText(role llmprotocol.Role, value string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
}

func encodedChatMessages(t *testing.T, messages []llmprotocol.Message) []json.RawMessage {
	t.Helper()
	body, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(
		llmprotocol.Request{Generation: 1, Model: "m", Messages: messages}, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Messages
}

func TestThinkingLeverReplaysAcrossCommittedTurns(t *testing.T) {
	e := newLeverEpisode(t, true)
	turn0 := []llmprotocol.Message{leverText(llmprotocol.RoleUser, "fix it")}
	provider0, ctx0 := e.turn(leverWorker, turn0, true)
	if ctx0.RaylineARCThinking == nil || !ctx0.RaylineARCThinking.Emitted ||
		provider0[0].Content[len(provider0[0].Content)-1].Text != leverDown {
		t.Fatalf("turn 0 did not steer: %+v", ctx0.RaylineARCThinking)
	}
	if e.stored().Thinking == nil {
		t.Fatal("the committed turn stored no ledger")
	}

	turn1 := append(append([]llmprotocol.Message(nil), turn0...),
		leverText(llmprotocol.RoleAssistant, "done"), leverText(llmprotocol.RoleUser, "next"))
	provider1, ctx1 := e.turn(leverWorker, turn1, true)
	if ctx1.RaylineARCThinking.Emitted || ctx1.RaylineARCThinking.Replayed != 1 {
		t.Fatalf("turn 1 trace = %+v", ctx1.RaylineARCThinking)
	}
	before, after := encodedChatMessages(t, provider0), encodedChatMessages(t, provider1)
	for index := range before {
		if !bytes.Equal(before[index], after[index]) {
			t.Fatalf("provider message %d changed between turns:\n%s\n%s", index, before[index], after[index])
		}
	}

	record := map[string]interface{}{}
	appendRaylineARCThinkingFields(record, ctx1)
	if record["thinking_level_in_force"] != "down" || record["thinking_lever"] != "prompt_steering_suffix" ||
		record["thinking_propensity"] != float64(1) || record["thinking_emitted"] != false {
		t.Fatalf("routing record fields = %v", record)
	}
}

func TestThinkingLeverLedgerCommitsOnlyWithTheTurn(t *testing.T) {
	e := newLeverEpisode(t, true)
	e.turn(leverWorker, []llmprotocol.Message{leverText(llmprotocol.RoleUser, "go")}, false)
	if state := e.stored(); state.Thinking != nil || state.TurnIndex != 0 {
		t.Fatalf("an aborted turn left state behind: %+v", state)
	}
}

func TestThinkingLeverLeavesUnboundWorkersAndTheLedgerAlone(t *testing.T) {
	e := newLeverEpisode(t, true)
	turn0 := []llmprotocol.Message{leverText(llmprotocol.RoleUser, "go")}
	e.turn(leverWorker, turn0, true)
	committed := e.stored().Thinking

	turn1 := append(append([]llmprotocol.Message(nil), turn0...),
		leverText(llmprotocol.RoleAssistant, "a"), leverText(llmprotocol.RoleUser, "b"))
	provider, ctx := e.turn(unboundWorker, turn1, true)
	if ctx.RaylineARCThinking.Skipped != thinkingSkipWorkerUnbound || len(provider[0].Content) != 1 {
		t.Fatalf("an unbound worker was steered: %+v", ctx.RaylineARCThinking)
	}
	if stored := e.stored().Thinking; stored == nil || len(stored.Entries) != len(committed.Entries) {
		t.Fatalf("an unbound turn changed the ledger: %+v", stored)
	}
}

func TestDisabledThinkingLeverChangesNothing(t *testing.T) {
	e := newLeverEpisode(t, false)
	messages := []llmprotocol.Message{leverText(llmprotocol.RoleUser, "go")}
	provider, ctx := e.turn(leverWorker, messages, true)
	if ctx.RaylineARCThinking != nil || len(provider[0].Content) != 1 {
		t.Fatalf("a disabled lever acted: %+v", ctx.RaylineARCThinking)
	}
	if e.stored().Thinking != nil {
		t.Fatal("a disabled lever stored a ledger, which would move the episode to v3")
	}
}
