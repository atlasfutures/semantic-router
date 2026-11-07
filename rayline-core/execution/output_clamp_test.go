package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// messagesUpstream records the max_tokens each dispatched Messages body
// carries, so a clamp is read off the wire rather than inferred.
func messagesUpstream(t *testing.T, seen *[]int64, mu *sync.Mutex) *httptest.Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			MaxTokens int64 `json:"max_tokens"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		mu.Lock()
		*seen = append(*seen, body.MaxTokens)
		mu.Unlock()
		writeJSON(w, 200, map[string]any{
			"id": "msg_clamp", "type": "message", "role": "assistant", "model": "qwen-fixture",
			"content":     []map[string]any{{"type": "text", "text": "ok"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 5, "output_tokens": 2},
		})
	}))
	t.Cleanup(upstream.Close)
	return upstream
}

// A harness that states one session-wide output allowance, Hermes' fixed
// 128,000 on the Messages wire among them, is refused by a deployment whose
// limit is smaller: it has no setting to lower what it sends. With the
// deployment opted in, the allowance is lowered to the limit and the turn is
// dispatched, carrying the limit upstream rather than the caller's number.
func TestOutputAllowanceOverTheDeploymentLimitIsClampedWhenOptedIn(t *testing.T) {
	for _, clamp := range []bool{false, true} {
		t.Run(fmt.Sprintf("clamp=%v", clamp), func(t *testing.T) {
			var (
				mu   sync.Mutex
				seen []int64
			)
			upstream := messagesUpstream(t, &seen, &mu)
			config := testConfig(t, upstream.URL, true)
			config.Deployments[0].MaxOutputTokens = 24576
			config.Deployments[0].ContextWindow = 98304
			config.Deployments[0].ClampOutputAllowance = clamp
			_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})

			body := `{"model":"rayline-auto","max_tokens":128000,"messages":[{"role":"user","content":"hi"}]}`
			status, raw := executeHTTP(t, server.URL, "/v1/messages", body, "clamp-session")

			if !clamp {
				if status != http.StatusBadRequest {
					t.Fatalf("status = %d, want 400 without the opt-in; body %s", status, raw)
				}
				if !strings.Contains(string(raw), "output allowance exceeds the admitted deployment limit") {
					t.Fatalf("body = %s, want the deployment-limit refusal", raw)
				}
				mu.Lock()
				defer mu.Unlock()
				if len(seen) != 0 {
					t.Fatalf("upstream saw %v, want no dispatch on a refused turn", seen)
				}
				return
			}
			if status != http.StatusOK {
				t.Fatalf("status = %d, want 200 with the opt-in; body %s", status, raw)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(seen) != 1 {
				t.Fatalf("upstream saw %d requests, want exactly 1", len(seen))
			}
			if seen[0] != 24576 {
				t.Fatalf("upstream max_tokens = %d, want the deployment limit 24576", seen[0])
			}
		})
	}
}

// prepare runs on the capacity and prefix paths as well as on dispatch, so a
// clamped turn reports the reduction once, from the dispatching caller, while
// every preparation still carries the clamped allowance: capacity admission
// sizes prompt + outputLimit, and the caller's unclamped figure would judge
// the deployment unable to hold a request it can.
func TestAClampedTurnReportsItOnceAndSizesCapacityWithTheClampedAllowance(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []int64
	)
	upstream := messagesUpstream(t, &seen, &mu)
	config := testConfig(t, upstream.URL, true)
	config.Deployments[0].MaxOutputTokens = 24576
	config.Deployments[0].ContextWindow = 98304
	config.Deployments[0].ClampOutputAllowance = true
	server, httpServer := executorWithHTTPDecisions(t, config, &decisionCapture{})

	request, envelope, _, err := server.engine.DecodeRequestForMutation(
		llmprotocol.AnthropicMessagesV1,
		[]byte(`{"model":"rayline-auto","max_tokens":128000,"messages":[{"role":"user","content":"hi"}]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	selected := server.actions[config.Actions[0].ID]
	prepared, err := server.prepare(request, envelope, selected, nil, newBranchState())
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if prepared.clamped == nil {
		t.Fatalf("clamped = nil, want the reduction reported to the caller")
	}
	if prepared.clamped.From != 128000 || prepared.clamped.To != 24576 {
		t.Fatalf("clamped = %d -> %d, want 128000 -> 24576", prepared.clamped.From, prepared.clamped.To)
	}
	if prepared.outputLimit != 24576 {
		t.Fatalf("outputLimit = %d, want the clamped 24576 so capacity reserves it", prepared.outputLimit)
	}
	// Why that matters: capacity admission reserves prompt + outputLimit, so
	// the caller's 128,000 against a 98,304 window would veto this deployment
	// for a prompt of any size.
	const shortPrompt = int64(100)
	if fact := sizedCapacityFact(selected.deployment, shortPrompt, capacityProvenanceMeasured, "t", prepared.outputLimit); !fact.Fits {
		t.Fatalf("required = %d of %d, want a short prompt to fit at the clamped allowance", *fact.Required, *fact.Available)
	}
	if fact := sizedCapacityFact(selected.deployment, shortPrompt, capacityProvenanceMeasured, "t", prepared.clamped.From); fact.Fits {
		t.Fatalf("required = %d of %d, want the caller's unclamped allowance not to fit", *fact.Required, *fact.Available)
	}

	status, raw := executeHTTP(t, httpServer.URL, "/v1/messages", `{"model":"rayline-auto","max_tokens":128000,"messages":[{"role":"user","content":"hi"}]}`, "clamp-once")
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", status, raw)
	}
}

// The tightest valid clamp: an admitted thinking budget one token below the
// deployment limit. Anthropic requires only that the budget be strictly below
// max_tokens, and RaiseMessagesAllowance leaves the clamped allowance alone
// because it then exceeds the budget, so the turn dispatches at the limit
// rather than meeting the refusal.
func TestAThinkingBudgetJustBelowTheLimitStillClamps(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []int64
	)
	upstream := messagesUpstream(t, &seen, &mu)
	config := testConfig(t, upstream.URL, true)
	config.Deployments[0].MaxOutputTokens = 24576
	config.Deployments[0].ContextWindow = 98304
	config.Deployments[0].ClampOutputAllowance = true
	server, httpServer := executorWithHTTPDecisions(t, config, &decisionCapture{})

	// Admit the action's control with a budget one token below the limit.
	for id, action := range server.actions {
		budget := int64(24575)
		action.control.BudgetTokens = &budget
		server.actions[id] = action
	}

	status, raw := executeHTTP(t, httpServer.URL, "/v1/messages",
		`{"model":"rayline-auto","max_tokens":128000,"messages":[{"role":"user","content":"hi"}]}`, "clamp-tight")
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200 at the tightest valid budget; body %s", status, raw)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 || seen[0] != 24576 {
		t.Fatalf("upstream max_tokens = %v, want the limit 24576", seen)
	}
}

// A dispatched clamped turn records the reduction exactly once, however many
// times prepare ran for it: this catches a report that moves back inside
// prepare.
func TestAClampedTurnIsReportedExactlyOnce(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	defer slog.SetDefault(previous)

	var (
		mu   sync.Mutex
		seen []int64
	)
	upstream := messagesUpstream(t, &seen, &mu)
	config := testConfig(t, upstream.URL, true)
	config.Deployments[0].MaxOutputTokens = 24576
	config.Deployments[0].ContextWindow = 98304
	config.Deployments[0].ClampOutputAllowance = true
	_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})

	executeHTTP(t, server.URL, "/v1/messages",
		`{"model":"rayline-auto","max_tokens":128000,"messages":[{"role":"user","content":"hi"}]}`, "clamp-once-log")

	records := clampRecords(t, &logs)
	if len(records) != 1 {
		t.Fatalf("clamp records = %d, want exactly 1 for one dispatched turn", len(records))
	}
	if got := records[0]; got.Stated != 128000 || got.Sent != 24576 {
		t.Fatalf("clamp record = %+v, want 128000 lowered to 24576", got)
	}
}

// The report sits after the pre-dispatch capacity gates, so a selection those
// gates veto records no clamp: the allowance it would have carried never
// reached a provider. The vetoed arm is the clamping one here, and the arm the
// service reselects has a limit the caller's allowance fits, so exactly one
// record is kept for the two selections of a single turn.
func TestAVetoedClampingSelectionRecordsNoClamp(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	defer slog.SetDefault(previous)

	// The first dispatch fails before output; the identical retry keeps the
	// decision but re-measures a prompt that no longer fits the local arm, so
	// the computed capacity gate vetoes it and the service picks cloud.
	local, cloud := &upstreamFixture{fail: 1}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	// Only the local arm clamps, and only it has a limit under the allowance
	// the caller states below.
	config.Deployments[0].ClampOutputAllowance = true
	fixture := &v2Decisions{choose: func(sent contextV2) string {
		if capacity := sent.Targets[0].Capacity; capacity != nil && !capacity.Fits {
			return cloudActionID
		}
		return localActionID
	}}
	_, httpServer := v2Server(t, config, fixture, 300, 990)

	// 1000 is over the local arm's 200 limit and under the cloud arm's 4096.
	// Otherwise chatTurn's own shape, whose tools and identity the v2 envelope
	// requires.
	turn := `{"model":"arc-1.0","max_tokens":1000,` + v2Tools + `,"messages":[` + userTask + `]}`
	if status, _ := executeHTTP(t, httpServer.URL, "/v1/chat/completions", turn, "clamp-veto"); status != 503 {
		t.Fatalf("first attempt status %d, want the upstream failure", status)
	}
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", turn, "clamp-veto"); status != 200 {
		t.Fatalf("retry status %d: %s", status, response)
	}

	// The computed gate is what refused the retry: the recorded fact still
	// fits (300 + 200), only the regrown prompt (990 + 200 over the 1000
	// window) does not. Asserting it keeps the test honest if retry ever
	// recomputes capacity before selection and reaches these statuses another
	// way.
	lookups := fixture.snapshot()
	if len(lookups) != 2 {
		t.Fatalf("got %d decision lookups, want 2", len(lookups))
	}
	retry := lookups[1].Execution
	if retry.Turn.Reconsult == nil || *retry.Turn.Reconsult != "veto" {
		t.Fatalf("retry reconsult = %v, want the pre-dispatch veto", retry.Turn.Reconsult)
	}
	if capacity := retry.Targets[0].Capacity; capacity == nil || capacity.Fits || *capacity.Required != 990+200 {
		t.Fatalf("vetoed capacity = %+v, want 1190 not fitting", capacity)
	}

	// The first attempt clamped and dispatched to local; the retry's local
	// selection was vetoed before dispatch and must leave no record, and the
	// cloud arm it reselected does not clamp at all.
	records := clampRecords(t, &logs)
	if len(records) != 1 {
		t.Fatalf("clamp records = %d, want 1: one clamped dispatch, nothing for the vetoed selection", len(records))
	}
	if got := records[0]; got.Deployment != config.Deployments[0].ID || got.Stated != 1000 || got.Sent != 200 {
		t.Fatalf("clamp record = %+v, want the local deployment lowering 1000 to 200", got)
	}
	if local.count() != 1 || cloud.count() != 1 {
		t.Fatalf("dispatches local=%d cloud=%d, want one each", local.count(), cloud.count())
	}
	if got := chatAllowance(t, local, 0); got != 200 {
		t.Fatalf("local max_completion_tokens = %d, want the clamped 200", got)
	}
	if got := chatAllowance(t, cloud, 0); got != 1000 {
		t.Fatalf("cloud max_completion_tokens = %d, want the caller's 1000 unclamped", got)
	}
}

// clampRecord is one decoded clamp report.
type clampRecord struct {
	Message    string `json:"msg"`
	Deployment string `json:"deployment"`
	Stated     int64  `json:"stated"`
	Sent       int64  `json:"sent"`
}

// clampRecords decodes the clamp reports out of captured JSON log lines, so a
// record is attributed to its deployment and figures rather than matched as a
// substring of the whole buffer.
func clampRecords(t *testing.T, logs *bytes.Buffer) []clampRecord {
	t.Helper()
	var records []clampRecord
	for _, line := range strings.Split(strings.TrimSpace(logs.String()), "\n") {
		if line == "" {
			continue
		}
		var record clampRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("log line %q: %v", line, err)
		}
		if record.Message == "output allowance clamped to the deployment limit" {
			records = append(records, record)
		}
	}
	return records
}

// chatAllowance is the output allowance the fixture's nth recorded Chat body
// carried.
func chatAllowance(t *testing.T, fixture *upstreamFixture, index int) int64 {
	t.Helper()
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	if index >= len(fixture.bodies) {
		t.Fatalf("fixture recorded %d bodies, wanted index %d", len(fixture.bodies), index)
	}
	for _, field := range []string{"max_completion_tokens", "max_tokens"} {
		if raw, ok := fixture.bodies[index][field]; ok {
			var value int64
			if err := json.Unmarshal(raw, &value); err != nil {
				t.Fatal(err)
			}
			return value
		}
	}
	t.Fatalf("body %d carried no output allowance: %v", index, fixture.bodies[index])
	return 0
}

// An allowance the deployment limit already holds is the caller's and reaches
// upstream unchanged, so opting a deployment in does not rewrite the turns
// that never needed it.
func TestAnAllowanceWithinTheLimitIsUnchangedUnderTheClamp(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []int64
	)
	upstream := messagesUpstream(t, &seen, &mu)
	config := testConfig(t, upstream.URL, true)
	config.Deployments[0].MaxOutputTokens = 24576
	config.Deployments[0].ContextWindow = 98304
	config.Deployments[0].ClampOutputAllowance = true
	_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})

	body := `{"model":"rayline-auto","max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`
	status, raw := executeHTTP(t, server.URL, "/v1/messages", body, "clamp-within")
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", status, raw)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 || seen[0] != 8192 {
		t.Fatalf("upstream max_tokens = %v, want the caller's 8192 untouched", seen)
	}
}
