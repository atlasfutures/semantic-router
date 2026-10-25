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
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A delta cell (ADR 0155): turn 1 stages and, at its commit, appends its
// reply; turn 2 extends that prefix with a tool result and sends only its
// new messages, decided synchronously.

type deltaFixture struct {
	*policySelectorFixture
	store   *raylinearc.MemoryEpisodeStore
	episode string
	// decided is the action the service's decide picks.
	decided string
}

func newDeltaFixture(t *testing.T, mode string, edit func(*config.RaylineARCPolicyServiceConfig)) *deltaFixture {
	t.Helper()
	decision := policyTestDecision("")
	policy := decision.Algorithm.RaylineARC.PolicyService
	policy.DecideMode, policy.TotalTimeoutSeconds = mode, 5
	if edit != nil {
		edit(policy)
	}
	bindings := policy.Bindings
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, policyTestActionIDs(bindings))
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, decision)
	scorer.armDelta([]string{raylinearc.PolicyCapabilityStagedTurns})
	selector.arm(&raylineARCArmedComponents{
		scorer:    scorer,
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fake.URL(), TotalTimeout: 5 * time.Second}),
	})
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	fixture := &deltaFixture{
		policySelectorFixture: &policySelectorFixture{fake: fake, selector: selector, decision: decision},
		store:                 store, episode: strings.Repeat("e", 64),
		decided: bindings[0].ActionID,
	}
	fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		for _, action := range request.Selection.AvailableActionIDs {
			if action == fixture.decided {
				return action
			}
		}
		return request.Selection.AvailableActionIDs[0]
	})
	return fixture
}

func (fixture *deltaFixture) scorer() *policyServiceScorer {
	return fixture.selector.armedComponents().scorer.(*policyServiceScorer)
}

// turn prepares the strict episode, selects and stages the selection as the
// request path does.
func (fixture *deltaFixture) turn(t *testing.T, body []byte) (*selection.SelectionResult, *RequestContext) {
	t.Helper()
	lease, state, err := fixture.store.Prepare(context.Background(), fixture.episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	result, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		TraceContext:          context.Background(),
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(fixture.store, lease, state, fixture.episode, time.Minute, nil),
	}
	ctx.RaylineARCTransaction.commitOnCompletion = true
	ctx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, 10)
	stageRaylineARCPolicySelection(ctx, result.RaylineARC, "")
	// A turn a test leaves unfinished is aborted, which stops its renewal;
	// after a commit the abort is a no-op.
	t.Cleanup(func() { _ = ctx.RaylineARCTransaction.abort(context.Background(), "test_cleanup") })
	return result, ctx
}

// complete is the client receiving the whole reply: the turn commits.
func complete(t *testing.T, ctx *RequestContext, reply []byte) {
	t.Helper()
	stageRaylineARCAppendSource(ctx, reply)
	ctx.RaylineARCTransaction.stageCallUsage(responseUsageMetrics{})
	if err := ctx.RaylineARCTransaction.commit(context.Background(), ctx); err != nil {
		t.Fatal(err)
	}
}

var (
	deltaUser = map[string]any{"role": "user", "content": "list the files"}
	// deltaReplyContent is turn 1's reply as the upstream states it, and
	// deltaAssistant the same message as Claude Code echoes it in the next
	// request (testdata/thread_continuation): citations [] on text, a direct
	// caller on tool_use.
	deltaReplyContent = []any{
		map[string]any{"type": "text", "text": "Listing them."},
		map[string]any{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": map[string]any{"command": "ls"}},
	}
	deltaAssistant = map[string]any{"role": "assistant", "content": []any{
		map[string]any{"type": "text", "text": "Listing them.", "citations": []any{}},
		map[string]any{"type": "tool_use", "id": "toolu_1", "caller": map[string]any{"type": "direct"}, "name": "Bash", "input": map[string]any{"command": "ls"}},
	}}
	deltaToolResult = map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"},
	}}
)

// deltaReply is turn 1's reply as the client receives it.
func deltaReply(t *testing.T) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "think",
		"content": deltaReplyContent, "stop_reason": "tool_use", "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 10, "output_tokens": 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// stagedFirstTurn runs and commits turn 1, which stages and appends.
func (fixture *deltaFixture) stagedFirstTurn(t *testing.T) {
	t.Helper()
	first, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	if first.RaylineARC.PolicyDecisionMode != policyDecisionModeSync {
		t.Fatalf("turn 1 decision mode %q", first.RaylineARC.PolicyDecisionMode)
	}
	complete(t, ctx, deltaReply(t))
	sent := fixture.fake.received()
	if len(sent) != 1 || sent[0].Stage == nil || !*sent[0].Stage || sent[0].TurnDelta != nil {
		t.Fatalf("turn 1 decide %+v", sent)
	}
}

// The held turn goes out as a delta on the appended prefix, with the shell
// only, staged, and is served its own decide.
func TestDeltaSendsOnlyTheNewTurn(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.stagedFirstTurn(t)
	appends := fixture.fake.receivedAppends()
	if len(appends) != 1 || len(appends[0].Messages) != 1 || appends[0].ContextEpoch != "0" ||
		len(appends[0].Attribution) != 1 || appends[0].Attribution[0].Message != 1 ||
		appends[0].Attribution[0].ActionID != fixture.decided {
		t.Fatalf("turn 1 append %+v", appends)
	}
	second, _ := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	if second.RaylineARC.PolicyActionID != fixture.decided || second.RaylineARC.PolicyDecisionMode != policyDecisionModeDelta {
		t.Fatalf("turn 2 served %s (%s)", second.RaylineARC.PolicyActionID, second.RaylineARC.PolicyDecisionMode)
	}
	sent := fixture.fake.received()
	delta := sent[len(sent)-1]
	if delta.TurnDelta == nil || delta.TurnDelta.BasePrefixDigest != fakePrefixDigest(fixture.episode, 2) ||
		len(delta.TurnDelta.Messages) != 1 || delta.Request.Shell != raylinearc.PolicyShellMessages ||
		delta.Stage == nil || !*delta.Stage || len(delta.Attribution) != 1 {
		t.Fatalf("turn 2 decide %+v", delta)
	}
	if body := fixture.fake.receivedBodies()[len(sent)-1]; bytes.Contains(body, []byte(`"messages":[{"content":"list`)) {
		t.Fatalf("the delta re-sent the staged conversation: %s", body)
	}
}

// fakeCanonicalRequest is a decided request as the service's projections
// read it (canonical-conversation.v1 and transcript-projection.v9 alike):
// cache_control dropped everywhere and string content as one text block,
// JSON key order and whitespace insignificant. The two differences VSR's
// prefix digest tolerates are exactly these.
func fakeCanonicalRequest(t *testing.T, request raylinearc.PolicyClientRequest) string {
	var messages []any
	if request.Input != nil {
		for _, item := range request.Input {
			var value any
			if err := json.Unmarshal(item, &value); err != nil {
				t.Error(err)
			}
			messages = append(messages, value)
		}
	} else if err := json.Unmarshal(request.Messages, &messages); err != nil {
		t.Error(err)
	}
	var strip func(any) any
	strip = func(value any) any {
		switch typed := value.(type) {
		case map[string]any:
			delete(typed, "cache_control")
			for key, member := range typed {
				typed[key] = strip(member)
			}
			if text, ok := typed["content"].(string); ok && typed["role"] != nil {
				typed["content"] = []any{map[string]any{"type": "text", "text": text}}
			}
		case []any:
			for index, element := range typed {
				typed[index] = strip(element)
			}
		}
		return value
	}
	var system, tools any
	_ = json.Unmarshal(request.System, &system)
	_ = json.Unmarshal(request.Tools, &tools)
	encoded, err := json.Marshal(map[string]any{
		"messages": strip(messages), "system": strip(system), "tools": strip(tools), "instructions": request.Instructions,
	})
	if err != nil {
		t.Error(err)
	}
	return string(encoded)
}

// newDeltaRefusingProxy answers every delta decide as a service that
// predates turn_delta does (how: status_404, status_405, or decode for a 200
// that is no decision) and forwards every other call to target.
func newDeltaRefusingProxy(t *testing.T, target, how string) string {
	t.Helper()
	upstream, err := url.Parse(target)
	if err != nil {
		t.Fatal(err)
	}
	forward := httputil.NewSingleHostReverseProxy(upstream)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		request.Body = io.NopCloser(bytes.NewReader(body))
		if request.URL.Path == "/v1/rayline/arc/policy/decide" && bytes.Contains(body, []byte(`"turn_delta"`)) {
			switch how {
			case "status_404":
				http.NotFound(writer, request)
			case "status_405":
				writer.WriteHeader(http.StatusMethodNotAllowed)
			default:
				writer.Header().Set("Content-Type", "application/json")
				_, _ = writer.Write([]byte(`{"schema_version":"rayline.arc.policy-decision-response.v1","unexpected":true}`))
			}
			return
		}
		forward.ServeHTTP(writer, request)
	}))
	t.Cleanup(server.Close)
	return server.URL
}

// parityTurn is one request of a replayed conversation and the reply the
// client then receives.
type parityTurn struct {
	request []byte
	reply   []byte
}

// replayParity replays the same turns on a sync cell and a delta cell and
// requires every turn to be decided on the same conversation, with the same
// action, arm and scores; it returns the delta cell's decision modes.
func replayParity(t *testing.T, turns []parityTurn) []string {
	t.Helper()
	sync := newDeltaFixture(t, config.RaylineARCDecideModeSync, nil)
	delta := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	for _, fixture := range []*deltaFixture{sync, delta} {
		fixture.fake.mu.Lock()
		fixture.fake.contentScores = true
		fixture.fake.mu.Unlock()
		// The pick follows the conversation too, so a different one shows.
		fake := fixture.fake
		fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			digest := fakeContentDigest(t, request.Request)
			return request.Selection.AvailableActionIDs[int(digest[0])%len(request.Selection.AvailableActionIDs)]
		})
	}
	var modes []string
	for index, turn := range turns {
		syncResult, syncCtx := sync.turn(t, turn.request)
		deltaResult, deltaCtx := delta.turn(t, turn.request)
		syncDecided := sync.fake.resolvedDecides()
		deltaDecided := delta.fake.resolvedDecides()
		if got, want := fakeCanonicalRequest(t, deltaDecided[len(deltaDecided)-1].Request),
			fakeCanonicalRequest(t, syncDecided[len(syncDecided)-1].Request); got != want {
			t.Fatalf("turn %d: the delta cell decided on\n%s\nthe sync cell on\n%s", index+1, got, want)
		}
		s, d := syncResult.RaylineARC, deltaResult.RaylineARC
		if s.PolicyActionID != d.PolicyActionID || s.PolicyArmID != d.PolicyArmID || s.SelectedArm != d.SelectedArm ||
			syncResult.SelectedModel != deltaResult.SelectedModel || s.ThinkingLevel != d.ThinkingLevel ||
			!reflect.DeepEqual(s.RawScores, d.RawScores) || !reflect.DeepEqual(s.AdjustedScores, d.AdjustedScores) {
			t.Fatalf("turn %d: sync %s/%s %v, delta %s/%s %v", index+1,
				s.PolicyActionID, s.PolicyArmID, s.RawScores, d.PolicyActionID, d.PolicyArmID, d.RawScores)
		}
		modes = append(modes, d.PolicyDecisionMode)
		complete(t, syncCtx, turn.reply)
		complete(t, deltaCtx, turn.reply)
	}
	if len(sync.fake.receivedAppends()) != 0 {
		t.Fatal("the sync cell appended")
	}
	return modes
}

// Replay parity: a synthetic tool loop decides the same on both cells, and
// every held turn after the first goes out as a delta.
func TestDeltaReplayParityToolLoop(t *testing.T) {
	second := map[string]any{"role": "assistant", "content": []any{
		map[string]any{"type": "tool_use", "id": "toolu_2", "caller": map[string]any{"type": "direct"}, "name": "Read", "input": map[string]any{"path": "README.md"}},
	}}
	secondReply, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": []any{
		map[string]any{"type": "tool_use", "id": "toolu_2", "name": "Read", "input": map[string]any{"path": "README.md"}},
	}})
	secondResult := map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "tool_result", "tool_use_id": "toolu_2", "content": "# Readme"},
	}}
	done, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": []any{
		map[string]any{"type": "text", "text": "Done."},
	}})
	modes := replayParity(t, []parityTurn{
		{policyTestRequest(t, deltaUser), deltaReply(t)},
		{policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult), secondReply},
		{policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult, second, secondResult), done},
	})
	if !slices_Equal(modes, []string{policyDecisionModeSync, policyDecisionModeDelta, policyDecisionModeDelta}) {
		t.Fatalf("decision modes %v", modes)
	}
}

// Replay parity on Claude Code 2.1.309's captured thread
// (testdata/thread_continuation): the mid-conversation system message is a
// cache-marked text block while it is last and a plain string once it is
// not, and the tool_use reply is echoed with a direct caller. The held turn
// still goes out as a delta and decides as the full request does.
func TestDeltaReplayParityClaudeCodeThread(t *testing.T) {
	resend := readThreadFixture(t, "claude-code-2.1.309-thread-create-resend.request.json")
	var body map[string]json.RawMessage
	if err := json.Unmarshal(resend, &body); err != nil {
		t.Fatal(err)
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatal(err)
	}
	with := func(conversation []json.RawMessage) []byte {
		changed := map[string]json.RawMessage{}
		for key, value := range body {
			changed[key] = value
		}
		encoded, err := json.Marshal(conversation)
		if err != nil {
			t.Fatal(err)
		}
		changed["messages"] = encoded
		request, err := json.Marshal(changed)
		if err != nil {
			t.Fatal(err)
		}
		return request
	}
	// Turn A ends at message 7, the system message Claude Code then sends as
	// a cache-marked block; its reply is message 8 as an upstream states it.
	var lastSystem struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(messages[7], &lastSystem); err != nil || lastSystem.Content == "" {
		t.Fatalf("message 7 is not a string system message: %s", messages[7])
	}
	marked, _ := json.Marshal(map[string]any{"role": "system", "content": []any{
		map[string]any{"type": "text", "text": lastSystem.Content, "cache_control": map[string]any{"type": "ephemeral"}},
	}})
	turnA := append(append([]json.RawMessage{}, messages[:7]...), marked)
	var echoed struct {
		Content []map[string]json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(messages[8], &echoed); err != nil {
		t.Fatal(err)
	}
	for _, block := range echoed.Content {
		delete(block, "caller")
	}
	reply, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": echoed.Content})
	final, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": []any{
		map[string]any{"type": "text", "text": "Done."},
	}})
	modes := replayParity(t, []parityTurn{
		{with(turnA), reply},
		{with(messages[:11]), final},
	})
	if !slices_Equal(modes, []string{policyDecisionModeSync, policyDecisionModeDelta}) {
		t.Fatalf("decision modes %v", modes)
	}
}

func slices_Equal(a, b []string) bool { return reflect.DeepEqual(a, b) }

// An append that fails still commits the turn, and the next turn is sent in
// full.
func TestDeltaAppendFailureCommitsAndSendsTheNextTurnInFull(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.fake.appendStatus = 503
	fixture.stagedFirstTurn(t)
	if state := readLedgerTestStateNoWait(fixture.store, fixture.episode); state == nil || state.TurnIndex != 1 {
		t.Fatalf("turn 1 did not commit: %+v", state)
	}
	if _, ok := fixture.scorer().staged.get(fixture.episode); ok {
		t.Fatal("a failed append left a staged turn")
	}
	second, ctx := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	sent := fixture.fake.received()
	if second.RaylineARC.PolicyActionID != fixture.decided || sent[len(sent)-1].TurnDelta != nil ||
		second.RaylineARC.PolicyDecisionMode != policyDecisionModeSync {
		t.Fatalf("turn 2 served %s, delta %v", second.RaylineARC.PolicyActionID, sent[len(sent)-1].TurnDelta)
	}
	complete(t, ctx, deltaReply(t))
}

// A delta the service holds no staged turn for (session_mismatch), or that
// a service which predates the member answers 404, 405 or with a body that
// does not decode, is sent again in full.
func TestDeltaRefusedResendsInFull(t *testing.T) {
	for _, tc := range []struct {
		name   string
		refuse func(*fakePolicyService)
	}{
		{"session_mismatch", func(fake *fakePolicyService) { fake.refuseDelta = true }},
		{"status_404", nil},
		{"status_405", nil},
		{"decode", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
			fixture.stagedFirstTurn(t)
			if tc.refuse != nil {
				fixture.fake.mu.Lock()
				tc.refuse(fixture.fake)
				fixture.fake.mu.Unlock()
			} else {
				refuseDeltasWith(t, fixture, tc.name)
			}
			second, ctx := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
			sent := fixture.fake.received()[1:]
			if tc.refuse == nil {
				// The refusing proxy forwards only the full resend.
				if len(sent) != 1 || sent[0].TurnDelta != nil {
					t.Fatalf("turn 2 decides %+v", sent)
				}
			} else if len(sent) != 2 || sent[0].TurnDelta == nil || sent[1].TurnDelta != nil || sent[1].Request.Shell != "" {
				t.Fatalf("turn 2 decides %+v", sent)
			}
			var messages []json.RawMessage
			if err := json.Unmarshal(sent[len(sent)-1].Request.Messages, &messages); err != nil || len(messages) != 3 {
				t.Fatalf("full resend carried %d messages", len(messages))
			}
			if second.RaylineARC.PolicyActionID != fixture.decided || second.RaylineARC.PolicyDecisionMode != policyDecisionModeSync {
				t.Fatalf("turn 2 served %s (%s)", second.RaylineARC.PolicyActionID, second.RaylineARC.PolicyDecisionMode)
			}
			complete(t, ctx, deltaReply(t))
		})
	}
}

// refuseDeltasWith points the fixture's client at a proxy that answers every
// delta decide as a service that predates turn_delta would, and forwards the
// rest to the fake.
func refuseDeltasWith(t *testing.T, fixture *deltaFixture, how string) {
	t.Helper()
	proxy := newDeltaRefusingProxy(t, fixture.fake.URL(), how)
	fixture.selector.armedComponents().policy = raylinearc.NewPolicyServiceClient(
		raylinearc.PolicyServiceConfig{BaseURL: proxy, TotalTimeout: 5 * time.Second},
	)
}

// The episode lease is held until the append finishes: the next turn cannot
// start before it.
func TestDeltaLeaseHeldUntilTheAppendFinishes(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	gate := make(chan struct{})
	fixture.fake.appendGate = gate
	_, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	stageRaylineARCAppendSource(ctx, deltaReply(t))
	committed := make(chan error, 1)
	go func() { committed <- ctx.RaylineARCTransaction.commit(context.Background(), ctx) }()
	deadline := time.Now().Add(2 * time.Second)
	for len(fixture.fake.receivedAppends()) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if state := readLedgerTestStateNoWait(fixture.store, fixture.episode); state != nil {
		t.Fatal("the episode was free while its append ran")
	}
	close(gate)
	if err := <-committed; err != nil {
		t.Fatal(err)
	}
	if state := readLedgerTestStateNoWait(fixture.store, fixture.episode); state == nil || state.TurnIndex != 1 {
		t.Fatalf("after the append: %+v", state)
	}
}

// A sync cell sends neither member, posts no append and records no mode.
func TestSyncCellStagesNothing(t *testing.T) {
	fixture := newDeltaFixture(t, "", nil)
	first, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	complete(t, ctx, deltaReply(t))
	body := fixture.fake.receivedBodies()[0]
	if bytes.Contains(body, []byte(`"stage"`)) || bytes.Contains(body, []byte(`"turn_delta"`)) ||
		first.RaylineARC.PolicyDecisionMode != "" || first.RaylineARC.PolicyStaging != nil ||
		len(fixture.fake.receivedAppends()) != 0 {
		t.Fatalf("sync cell staged: %s", body)
	}
	fields := map[string]interface{}{}
	addPolicyDecisionModeFields(fields, first.RaylineARC)
	record := llmUsageRecord{}
	addUsageRecordARCAttribution(&record, &RequestContext{VSRRaylineARC: first.RaylineARC})
	if _, present := record.fields()["decision_mode"]; present || len(fields) != 0 {
		t.Fatalf("sync records carry a decision mode: %v", fields)
	}
}

// A delta cell's dispatch-time records carry the mode.
func TestDeltaDecisionModeReachesTheUsageRecord(t *testing.T) {
	for _, mode := range []string{policyDecisionModeDelta, policyDecisionModeSync} {
		trace := &selection.RaylineARCTrace{PolicyDecisionMode: mode}
		record := llmUsageRecord{}
		addUsageRecordARCAttribution(&record, &RequestContext{VSRRaylineARC: trace})
		if fields := record.fields(); fields["decision_mode"] != mode {
			t.Fatalf("usage decision_mode %v, want %s", fields["decision_mode"], mode)
		}
		fields := map[string]interface{}{}
		addPolicyDecisionModeFields(fields, trace)
		if fields["decision_mode"] != mode || len(fields) != 1 {
			t.Fatalf("dispatch fields %v", fields)
		}
	}
}

func TestPolicyAssistantOutputNormalizesResponsesItems(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","status":"completed","output":[` +
		`{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"a <b>"}]},` +
		`{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"ls","arguments":"{}"}]}`)
	items, ok := policyAssistantOutput(policyFormatResponses, body)
	if !ok || len(items) != 2 {
		t.Fatalf("items %s", items)
	}
	for _, item := range items {
		if bytes.Contains(item, []byte(`"status"`)) || bytes.Contains(item, []byte(`"id":"`)) {
			t.Fatalf("retention fields kept: %s", item)
		}
	}
	if !bytes.Contains(items[0], []byte(`"a <b>"`)) || !bytes.Contains(items[1], []byte(`"call_id":"call_1"`)) {
		t.Fatalf("items %s", items)
	}
	roles, err := policyResponsesRoles(items)
	if err != nil || roles[0] != "assistant" || roles[1] != "assistant" {
		t.Fatalf("roles %v %v", roles, err)
	}
}

// The append stays well within the next turn's acquire timeout, so it never
// makes that turn contended.
func TestPolicyAppendTimeoutStaysInsideTheAcquireTimeout(t *testing.T) {
	if got := policyAppendTimeout(30*time.Second, 8*time.Second); got != 4*time.Second {
		t.Fatalf("append timeout %v", got)
	}
	if got := policyAppendTimeout(time.Second, 8*time.Second); got != time.Second {
		t.Fatalf("append timeout %v", got)
	}
}

// An append takes its own admission slot, never one a decide needs.
func TestDeltaAppendTakesNoDecideSlot(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, func(policy *config.RaylineARCPolicyServiceConfig) { policy.MaxInflightCalls = 1 })
	fixture.selector.armedComponents().admission = raylinearc.NewAdmissionGate(1)
	gate := make(chan struct{})
	fixture.fake.appendGate = gate
	_, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	stageRaylineARCAppendSource(ctx, deltaReply(t))
	ctx.RaylineARCTransaction.stageCallUsage(responseUsageMetrics{})
	committed := make(chan error, 1)
	go func() { committed <- ctx.RaylineARCTransaction.commit(context.Background(), ctx) }()
	deadline := time.Now().Add(2 * time.Second)
	for len(fixture.fake.receivedAppends()) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	other, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("f", 64), RequestID: "req-other", State: other,
			RawRequest: policyTestRequest(t, deltaUser), RequestFormat: policyFormatAnthropic,
		},
	}); err != nil {
		t.Fatalf("another episode's decide was shed while an append ran: %v", err)
	}
	close(gate)
	if err := <-committed; err != nil {
		t.Fatal(err)
	}
}

// Claude Code's appended reply must reproduce the echo the client sends;
// one echoed otherwise is told apart as the appended reply's mismatch.
func TestDeltaClaudeCodeEchoShapes(t *testing.T) {
	resend := readThreadFixture(t, "claude-code-2.1.309-thread-create-resend.request.json")
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(resend, &body); err != nil {
		t.Fatal(err)
	}
	echoed := body.Messages[8]
	var message struct {
		Content []map[string]json.RawMessage `json:"content"`
	}
	_ = json.Unmarshal(echoed, &message)
	delete(message.Content[0], "caller")
	reply, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": message.Content})
	appended, ok := policyAssistantOutput(policyFormatAnthropic, reply)
	if !ok || raylinearc.MessagesDigest(appended, 1) != raylinearc.MessagesDigest([]json.RawMessage{echoed}, 1) {
		t.Fatalf("appended %s does not reproduce the echo %s", appended, echoed)
	}
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.stagedFirstTurn(t)
	bare := map[string]any{"role": "assistant", "content": deltaReplyContent}
	var messages []json.RawMessage
	_ = json.Unmarshal(requestMessages(t, policyTestRequest(t, deltaUser, bare, deltaToolResult)), &messages)
	_, reason := policyDeltaFor(fixture.scorer(), fixture.episode, &raylinearc.EpisodeState{TurnIndex: 1},
		&raylinearc.PolicyEpisodeState{}, raylinearc.PolicyTransitionExtends, policyFormatAnthropic, messages)
	if reason != "appended_mismatch" {
		t.Fatalf("delta reason %q, want appended_mismatch", reason)
	}
}

func requestMessages(t *testing.T, body []byte) []byte {
	t.Helper()
	var request struct {
		Messages json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	return request.Messages
}

// A reply with thinking is not appended: no capture shows its echo.
func TestDeltaThinkingReplyIsNotAppended(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	_, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	reply, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "content": []any{
		map[string]any{"type": "thinking", "thinking": "", "signature": "sig"},
		map[string]any{"type": "text", "text": "done"},
	}})
	complete(t, ctx, reply)
	if len(fixture.fake.receivedAppends()) != 0 {
		t.Fatal("a thinking reply was appended")
	}
	if _, ok := fixture.scorer().staged.get(fixture.episode); ok {
		t.Fatal("a thinking reply left a staged turn")
	}
}

func TestPolicyReplyThinksInEveryFormat(t *testing.T) {
	for _, tc := range []struct {
		format string
		body   string
		thinks bool
	}{
		{policyFormatAnthropic, `{"role":"assistant","content":[{"type":"redacted_thinking","data":"x"}]}`, true},
		{policyFormatAnthropic, `{"role":"assistant","content":[{"type":"text","text":"a"}]}`, false},
		{policyFormatOpenAI, `{"choices":[{"message":{"role":"assistant","content":"a","reasoning":"why"}}]}`, true},
		{policyFormatOpenAI, `{"choices":[{"message":{"role":"assistant","content":"a","reasoning_details":[{"type":"reasoning.text"}]}}]}`, true},
		{policyFormatOpenAI, `{"choices":[{"message":{"role":"assistant","content":"a","reasoning":null}}]}`, false},
		{policyFormatResponses, `{"output":[{"type":"reasoning","summary":[]},{"type":"message","role":"assistant","content":[]}]}`, true},
		{policyFormatResponses, `{"output":[{"type":"message","role":"assistant","content":[]}]}`, false},
	} {
		if got := policyReplyThinks(tc.format, []byte(tc.body)); got != tc.thinks {
			t.Errorf("%s %s: thinks %v", tc.format, tc.body, got)
		}
	}
}

// A delta cell stages only on a service that advertises staged turns, read
// at readiness; without the capability it sends every turn in full.
func TestDeltaNeedsTheStagedTurnsCapability(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	scorer := fixture.scorer()
	ready := func() {
		t.Helper()
		client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second})
		if err := requireRaylineARCPolicyServing(context.Background(), client,
			fixture.decision.Algorithm.RaylineARC.PolicyService, nil, scorer.armDelta); err != nil {
			t.Fatal(err)
		}
	}
	ready()
	if !scorer.stagedTurnsServed.Load() {
		t.Fatal("an advertised staged_turns did not arm the delta")
	}
	fixture.fake.mu.Lock()
	fixture.fake.advertised = []string{raylinearc.PolicyCapabilityDecisionsV2}
	fixture.fake.mu.Unlock()
	ready()
	if scorer.stagedTurnsServed.Load() {
		t.Fatal("a service without staged_turns armed the delta")
	}
	first, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	complete(t, ctx, deltaReply(t))
	body := fixture.fake.receivedBodies()[0]
	if bytes.Contains(body, []byte(`"stage"`)) || first.RaylineARC.PolicyDecisionMode != "" || len(fixture.fake.receivedAppends()) != 0 {
		t.Fatalf("an unarmed delta cell staged: %s", body)
	}
}

// A relaxed cell and a replica-pool cell send every turn in full whatever
// decide_mode says.
func TestDeltaOnlyOnStrictSingleServiceCells(t *testing.T) {
	decision := policyTestDecision("")
	decision.Algorithm.RaylineARC.PolicyService.DecideMode = config.RaylineARCDecideModeDelta
	if scorer := newPolicyServiceScorer(&config.RouterConfig{}, decision); !scorer.delta {
		t.Fatal("a strict delta cell does not stage")
	}
	decision.Algorithm.RaylineARC.PolicyService.Replicas = []config.RaylineARCEncoderReplicaConfig{{}}
	if scorer := newPolicyServiceScorer(&config.RouterConfig{}, decision); scorer.delta {
		t.Fatal("a replica-pool cell stages")
	}
}

// A cache_control key nested inside a tool's input is the tool's data, not a
// cache marker: a history that changed it goes in full.
func TestDeltaDigestKeepsNestedCacheControl(t *testing.T) {
	marked := json.RawMessage(`{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}],"cache_control":{"type":"ephemeral"}}`)
	bare := json.RawMessage(`{"content":[{"text":"hi","type":"text"}],"role":"user"}`)
	if policyDeltaDigest([]json.RawMessage{marked}, 1) != policyDeltaDigest([]json.RawMessage{bare}, 1) {
		t.Fatal("a moved cache marker changed the digest")
	}
	block := json.RawMessage(`{"role":"system","content":[{"type":"text","text":"note","cache_control":{"type":"ephemeral"}}]}`)
	demoted := json.RawMessage(`{"role":"system","content":"note"}`)
	if policyDeltaDigest([]json.RawMessage{block}, 1) != policyDeltaDigest([]json.RawMessage{demoted}, 1) {
		t.Fatal("a demoted system message changed the digest")
	}
	old := json.RawMessage(`{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Edit","input":{"cache_control":"old"}}]}`)
	changed := json.RawMessage(`{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Edit","input":{"cache_control":"new"}}]}`)
	if policyDeltaDigest([]json.RawMessage{old}, 1) == policyDeltaDigest([]json.RawMessage{changed}, 1) {
		t.Fatal("a changed tool input matched")
	}
	userBlock := json.RawMessage(`{"role":"user","content":[{"type":"text","text":"hi"}]}`)
	userString := json.RawMessage(`{"role":"user","content":"hi"}`)
	if policyDeltaDigest([]json.RawMessage{userBlock}, 1) == policyDeltaDigest([]json.RawMessage{userString}, 1) {
		t.Fatal("only a system message's single text block is its text")
	}
}

// An append VSR stopped waiting for may still hold the service's episode
// lock: the next turn goes in full and retries session_busy one acquire
// timeout longer.
func TestDeltaUnsettledAppendExtendsTheNextBusyWait(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	scorer := fixture.scorer()
	gate := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-gate:
		default:
			close(gate)
		}
	})
	fixture.fake.appendGate = gate
	_, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	ctx.RaylineARCTransaction.policyStaging.appendTimeout = 50 * time.Millisecond
	complete(t, ctx, deltaReply(t))
	staged, ok := scorer.staged.get(fixture.episode)
	if !ok || !staged.unsettled {
		t.Fatalf("a timed-out append left %+v (%v)", staged, ok)
	}
	close(gate)
	if got := policyFirstBusyWait(time.Second, "append_unsettled"); got != 2*time.Second {
		t.Fatalf("unsettled busy wait %v", got)
	}
	if got := policyFirstBusyWait(time.Second, "delta"); got != time.Second {
		t.Fatalf("busy wait %v", got)
	}
	second, ctx := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	sent := fixture.fake.received()
	if second.RaylineARC.PolicyDecisionMode != policyDecisionModeSync || sent[len(sent)-1].TurnDelta != nil {
		t.Fatalf("turn 2 went %s", second.RaylineARC.PolicyDecisionMode)
	}
	complete(t, ctx, deltaReply(t))
}

