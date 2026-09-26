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
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// auditTurn prepares the episode, audits one provider-bound body and
// finalizes the turn, returning the verdict.
func auditTurn(t *testing.T, e *leverEpisode, body string, commit bool) *raylineARCUpstreamAudit {
	t.Helper()
	lease, state, err := e.store.Prepare(context.Background(), e.episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	transaction := newRaylineARCEpisodeTransaction(e.store, lease, state, e.episode, time.Minute, nil)
	transaction.markSelection(0, 10)
	ctx := &RequestContext{
		Headers:               map[string]string{},
		VSRSelectedDecision:   e.decision,
		RaylineARCDispatch:    &raylinearc.WorkerManifest{ID: leverWorker},
		RaylineARCTransaction: transaction,
	}
	(&OpenAIRouter{}).auditRaylineARCUpstream([]byte(body), ctx)
	if commit {
		if err := transaction.commit(context.Background(), ctx); err != nil {
			t.Fatal(err)
		}
	} else if err := transaction.abort(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
	return ctx.RaylineARCUpstreamAudit
}

func TestUpstreamAuditJudgesEachBodyAgainstTheLastCommittedOne(t *testing.T) {
	e := newLeverEpisode(t, false)
	e.decision.Algorithm.RaylineARC.UpstreamAudit = config.RaylineARCUpstreamAuditConfig{Enabled: true}

	turn1 := `{"messages":[{"role":"system","content":"sys"},{"role":"user","content":[{"type":"text","text":"go","cache_control":{"type":"ephemeral"}}]}]}`
	// The breakpoint moved off the earlier message, which the encoder then
	// writes as a plain string: same ask, different bytes, still an extension.
	turn2 := `{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"go"},{"role":"assistant","content":"ok"},{"role":"user","content":[{"type":"text","text":"more","cache_control":{"type":"ephemeral"}}]}]}`
	rewritten := `{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"summary"},{"role":"user","content":"more"}]}`

	first := auditTurn(t, e, turn1, true)
	if first == nil || first.Extends != nil || first.Messages != 2 {
		t.Fatalf("first turn verdict = %+v", first)
	}
	second := auditTurn(t, e, turn2, true)
	if second.Extends == nil || !*second.Extends || second.PrefixMessages != 2 || second.Messages != 4 {
		t.Fatalf("an extension was not recognised: %+v", second)
	}
	retry := auditTurn(t, e, turn2, true)
	if retry.Extends == nil || !*retry.Extends || retry.Digest != second.Digest {
		t.Fatalf("a retry did not match the committed body: %+v", retry)
	}
	aborted := auditTurn(t, e, rewritten, false)
	if aborted.Extends == nil || *aborted.Extends {
		t.Fatalf("a rewritten transcript read as an extension: %+v", aborted)
	}
	// The aborted turn stored nothing, so the committed body is still the
	// reference.
	again := auditTurn(t, e, turn2, true)
	if again.Extends == nil || !*again.Extends {
		t.Fatalf("an aborted turn replaced the reference: %+v", again)
	}

	record := map[string]interface{}{}
	appendRaylineARCUpstreamFields(record, &RequestContext{RaylineARCUpstreamAudit: again})
	if record["upstream_extends_previous"] != true || record["upstream_messages"] != 4 {
		t.Fatalf("routing record = %v", record)
	}
}

func TestUpstreamAuditIsOffUnlessConfigured(t *testing.T) {
	e := newLeverEpisode(t, false)
	if audit := auditTurn(t, e, `{"messages":[{"role":"user","content":"go"}]}`, true); audit != nil {
		t.Fatalf("audit ran while disabled: %+v", audit)
	}
	if e.stored().Upstream != nil {
		t.Fatal("a disabled audit stored upstream records, which would move the episode to v3")
	}
}

func TestEligibilityHeaderLimitsSteeringToOptedInConversations(t *testing.T) {
	e := newLeverEpisode(t, true)
	e.decision.Algorithm.RaylineARC.ThinkingLever.EligibilityHeader = "x-rayline-thinking-test"
	messages := leverMessages()

	provider, ctx := e.turnWithHeaders(leverWorker, messages, true, nil)
	if ctx.RaylineARCThinking.Skipped != thinkingSkipNotEligible || len(provider[0].Content) != 1 {
		t.Fatalf("a conversation without the header was steered: %+v", ctx.RaylineARCThinking)
	}
	if e.stored().Thinking != nil {
		t.Fatal("an ineligible turn touched the ledger")
	}
	provider, ctx = e.turnWithHeaders(leverWorker, messages, true, map[string]string{"x-rayline-thinking-test": "1"})
	if !ctx.RaylineARCThinking.Emitted || len(provider[0].Content) != 2 {
		t.Fatalf("an opted-in conversation was not steered: %+v", ctx.RaylineARCThinking)
	}
}
