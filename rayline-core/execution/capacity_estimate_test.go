package execution

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/tokencount"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

// The Rayline Local default: a custom local endpoint with no prompt counter
// (98,304-token window) and two cloud deployments with 1,000,000-token windows.
const (
	estimateLocalWindow = 98304
	estimateCloudWindow = 1_000_000
)

var secondCloudActionID = strings.Repeat("e", 64)

func capacityEstimateConfig(t *testing.T, localURL, cloudURL string) Config {
	t.Helper()
	config := hybridV2Config(t, localURL, cloudURL)
	config.Deployments[0].ContextWindow, config.Deployments[0].MaxOutputTokens = estimateLocalWindow, 8192
	config.Deployments[1].ContextWindow, config.Deployments[1].MaxOutputTokens = estimateCloudWindow, 32000
	second := config.Deployments[1]
	second.ID = "cloud-second"
	action := config.Actions[1]
	action.ID, action.Worker, action.DeploymentID = secondCloudActionID, "kimi-k3-second", "cloud-second"
	config.Deployments = append(config.Deployments, second)
	config.Actions = append(config.Actions, action)
	return config
}

// fitPolicy stands in for the service: it never offers a target the core
// reports fits:false, prefers local, and otherwise takes the last target left.
func fitPolicy(sent contextV2) string {
	chosen := ""
	for _, target := range sent.Targets {
		if target.Capacity != nil && !target.Capacity.Fits {
			continue
		}
		if target.Placement == "local" {
			return target.ActionID
		}
		chosen = target.ActionID
	}
	return chosen
}

// capacityFixture is real or seeded content with its real Qwen3.8-27B token
// count, from testdata/capacity/make_fixtures.py (counts.json).
type capacityFixture struct {
	text   string
	tokens int64
}

func loadCapacityFixtures(t *testing.T) map[string]capacityFixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/capacity/counts.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Fixtures map[string]struct {
			Tokens int64 `json:"qwen_tokens"`
		} `json:"fixtures"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	fixtures := map[string]capacityFixture{}
	for name, entry := range manifest.Fixtures {
		file, err := os.Open(filepath.Join("testdata/capacity", name+".txt.gz"))
		if err != nil {
			t.Fatal(err)
		}
		reader, err := gzip.NewReader(file)
		if err != nil {
			t.Fatal(err)
		}
		text, err := io.ReadAll(reader)
		file.Close()
		if err != nil {
			t.Fatal(err)
		}
		fixtures[name] = capacityFixture{text: string(text), tokens: entry.Tokens}
	}
	return fixtures
}

func jsonMessage(t *testing.T, role, content string) string {
	t.Helper()
	raw, err := json.Marshal(map[string]string{"role": role, "content": content})
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func toolResultMessage(t *testing.T, content string) string {
	t.Helper()
	raw, err := json.Marshal(map[string]string{"role": "tool", "tool_call_id": "call_1", "content": content})
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func bigTurn(messages ...string) string {
	return `{"model":"arc-1.0","max_tokens":4096,` + v2Tools + `,"messages":[` + strings.Join(messages, ",") + `]}`
}

type qwenConversation struct {
	Body   json.RawMessage `json:"body"`
	Tokens int64           `json:"qwen_prompt_tokens"`
}

// loadQwenConversations reads Chat bodies with the real Qwen3.8-27B template
// token counts (testdata/capacity/make_conversations.py).
func loadQwenConversations(t *testing.T) map[string]qwenConversation {
	t.Helper()
	file, err := os.Open("testdata/capacity/conversations.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	reader, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	conversations := map[string]qwenConversation{}
	if err := json.NewDecoder(reader).Decode(&conversations); err != nil {
		t.Fatal(err)
	}
	return conversations
}

// TestQwenTemplateCountMatchesTheReference renders each conversation as the
// Qwen3.8 chat template does and counts it with the embedded tokenizer: the
// count before margins equals transformers' apply_chat_template count, and
// the capacity count (with margins) is never below it.
func TestQwenTemplateCountMatchesTheReference(t *testing.T) {
	conversations := loadQwenConversations(t)
	if len(conversations) < 6 {
		t.Fatalf("conversations %d", len(conversations))
	}
	counter, err := tokencount.Qwen38()
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(conversations))
	for name := range conversations {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		conversation := conversations[name]
		rendered, _, _, _, err := qwenRender(conversation.Body)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		exact := int64(counter.Count(rendered))
		counted, images, err := qwenPromptTokens(conversation.Body)
		if err != nil || images != 0 {
			t.Fatalf("%s: %v images=%d", name, err, images)
		}
		t.Logf("%-32s reference=%6d rendered=%6d capacity=%6d", name, conversation.Tokens, exact, counted)
		if exact != conversation.Tokens || counted < conversation.Tokens {
			t.Errorf("%s: rendered %d, capacity %d, reference %d", name, exact, counted, conversation.Tokens)
		}
	}
}

// TestCapacityFactsAtTheWindowBoundary checks both local paths with a window
// one token short of the real prompt plus output: the Qwen3.8 count on every
// conversation, and the byte bound of another model on every text fixture.
func TestCapacityFactsAtTheWindowBoundary(t *testing.T) {
	const output = 4096
	qwen := Deployment{Kind: "local", RegistryModel: qwen38RegistryModel, Protocol: llmprotocol.OpenAIChatV1}
	for name, conversation := range loadQwenConversations(t) {
		qwen.ContextWindow = conversation.Tokens + output - 1
		fact := estimatedCapacityFact(qwen, conversation.Body, output)
		if fact.Fits || fact.Provenance != capacityProvenanceMeasured || *fact.Units != qwenTokenUnits {
			t.Errorf("qwen %s: %s", name, mustJSON(fact))
		}
	}
	other := Deployment{Kind: "local", RegistryModel: "mistralai/devstral-small", Protocol: llmprotocol.OpenAIChatV1}
	fixtures := loadCapacityFixtures(t)
	names := make([]string, 0, len(fixtures))
	for name := range fixtures {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fixture := fixtures[name]
		body := []byte(bigTurn(jsonMessage(t, "user", fixture.text)))
		other.ContextWindow = fixture.tokens + output - 1
		fact := estimatedCapacityFact(other, body, output)
		t.Logf("%-34s qwen tokens=%6d byte bound=%7d (%.2fx)", name, fixture.tokens, fact.prompt, float64(fact.prompt)/float64(fixture.tokens))
		if fact.Fits || fact.Provenance != capacityProvenanceEstimated || fact.prompt < int64(len(fixture.text)) {
			t.Errorf("byte bound %s: %s", name, mustJSON(fact))
		}
	}
}

func TestCapacityEstimateRoutesRealContentByLocalWindow(t *testing.T) {
	fixtures := loadCapacityFixtures(t)
	code, lock, prose := fixtures["code_30k"], fixtures["uvlock_120k"], fixtures["prose"]
	if code.tokens+lock.tokens <= estimateLocalWindow {
		t.Fatalf("the uv.lock case must exceed the local window: %d", code.tokens+lock.tokens)
	}
	// About 626k real tokens: far over the local window, well inside the cloud's.
	large := strings.Repeat(prose.text, 20)
	for _, tc := range []struct {
		name     string
		messages []string
		local    bool
	}{
		{"30k real tokens of code stays local", []string{jsonMessage(t, "user", code.text)}, true},
		// The review's case: bytes/3 * 1.15 put this at ~90k and sent it to
		// Ollama, which would have truncated it.
		{"code plus a uv.lock (110k real) hands off before local sees it", []string{jsonMessage(t, "user", code.text), jsonMessage(t, "user", lock.text)}, false},
		{"626k real tokens goes to cloud", []string{jsonMessage(t, "user", large)}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			local, cloud := &upstreamFixture{}, &upstreamFixture{}
			config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
			fixture := &v2Decisions{choose: fitPolicy}
			_, httpServer := v2Server(t, config, fixture)
			status, header, response := executeWithHeaders(t, httpServer.URL, bigTurn(tc.messages...), "estimate")
			if status != 200 {
				t.Fatalf("status %d: %.300s", status, response)
			}
			lookups := fixture.snapshot()
			if len(lookups) != 1 {
				t.Fatalf("got %d lookups", len(lookups))
			}
			for _, target := range lookups[0].Execution.Targets {
				fact := target.Capacity
				if target.Placement == "cloud" {
					// No counter and no estimate: the provider refuses an
					// oversized prompt itself.
					if fact != nil {
						t.Fatalf("cloud target carries an estimated fact %s", mustJSON(fact))
					}
					continue
				}
				if fact == nil || fact.Provenance != "local_measured" || fact.Fits != tc.local || *fact.Available != estimateLocalWindow || *fact.AvailableSource != "configured" {
					t.Fatalf("local fact %s", mustJSON(fact))
				}
				if !fact.Fits && (fact.Reason == nil || *fact.Reason != CapacityContext) {
					t.Fatalf("an infeasible target needs its reason: %s", mustJSON(fact))
				}
			}
			executed := header.Get("X-Rayline-Executed-Action-Id")
			if tc.local && (executed != localActionID || local.count() != 1 || cloud.count() != 0) {
				t.Fatalf("executed %q local=%d cloud=%d", executed, local.count(), cloud.count())
			}
			if !tc.local && (executed == localActionID || local.count() != 0 || cloud.count() != 1) {
				t.Fatalf("executed %q local=%d cloud=%d", executed, local.count(), cloud.count())
			}
		})
	}
}

func TestCapacityEstimateNeverRefusesACloudConversation(t *testing.T) {
	// Over 1M estimated tokens: local is excluded, cloud stays unknown and is
	// selected. The provider's own OpenAI-style refusal reaches the caller; on
	// the Messages wire it starts with "prompt is too long:" so Claude Code
	// compacts, followed by the provider's text.
	local := &upstreamFixture{}
	cloudCalls := 0
	cloud := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cloudCalls++
		_, _ = io.Copy(io.Discard, r.Body)
		modelPinHeaders(w.Header())
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"This endpoint's maximum context length is 1000000 tokens. However, you requested about 1180000 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}`)
	}))
	defer cloud.Close()
	config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, cloud.URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: fitPolicy}
	_, httpServer := v2Server(t, config, fixture)
	huge := bigTurn(jsonMessage(t, "user", strings.Repeat(loadCapacityFixtures(t)["prose"].text, 36)))
	for _, tc := range []struct{ path, want string }{
		{"/v1/chat/completions", `"code":"context_length_exceeded","message":"This endpoint's maximum context length is 1000000 tokens.`},
		{"/v1/messages", `"message":"prompt is too long: This endpoint's maximum context length is 1000000 tokens.`},
	} {
		status, response := executeHTTP(t, httpServer.URL, tc.path, huge, "cloud-refuses"+tc.path)
		if status != http.StatusBadRequest || !strings.Contains(string(response), tc.want) || strings.Contains(string(response), CodeContextExceedsAllDeployments) {
			t.Fatalf("%s status %d: %.400s", tc.path, status, response)
		}
	}
	if len(fixture.snapshot()) != 2 || local.count() != 0 || cloudCalls != 2 {
		t.Fatalf("lookups=%d local=%d cloud=%d", len(fixture.snapshot()), local.count(), cloudCalls)
	}
	// A shortfall on local alone never maps a service refusal to the 400
	// while a cloud target is unknown.
	fixture.refuse = func(contextV2) string { return "execution_no_feasible_target" }
	status, _, response := executeWithHeaders(t, httpServer.URL, bigTurn(jsonMessage(t, "user", strings.Repeat(loadCapacityFixtures(t)["prose"].text, 4))), "service-refuses")
	if status != http.StatusBadGateway || !strings.Contains(string(response), "decision_failed") {
		t.Fatalf("status %d: %.300s", status, response)
	}
}

func TestProviderContextRefusalOnlyRewritesRecognisedRefusals(t *testing.T) {
	for _, tc := range []struct {
		status int
		body   string
		want   bool
	}{
		{400, `{"error":{"code":"context_length_exceeded","message":"too big"}}`, true},
		{400, `{"error":{"message":"This model's maximum context length is 200000 tokens."}}`, true},
		{413, `{"error":{"message":"Input is too long for requested model."}}`, true},
		{400, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`, true},
		{400, `{"error":{"message":"invalid tool schema"}}`, false},
		{500, `{"error":{"message":"maximum context length exceeded upstream"}}`, false},
		{400, `{"object":"error","message":"This model's maximum context length is 98304 tokens. However, you requested 120000 tokens.","type":"BadRequestError","code":400}`, true},
		{400, `{"error":"prompt is too long"}`, true},
		// A marker outside the error's code, type or message is not a refusal.
		{400, `{"error":{"message":"invalid tool schema","param":"maximum context length"}}`, false},
		{400, `{"detail":"prompt is too long"}`, false},
	} {
		if got := isProviderContextRefusal(tc.status, []byte(tc.body)); got != tc.want {
			t.Errorf("%d %s: %v", tc.status, tc.body, got)
		}
	}
	already := []byte(`{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`)
	if got := withPromptTooLong(already); string(got) != string(already) {
		t.Fatalf("an Anthropic refusal was rewritten: %s", got)
	}
}

func TestCapacityShortfallOnEveryTargetRefusesWhateverItsProvenance(t *testing.T) {
	fixtures := loadCapacityFixtures(t)
	over := bigTurn(jsonMessage(t, "user", fixtures["code_30k"].text), jsonMessage(t, "user", fixtures["uvlock_120k"].text))
	t.Run("local only, no counter", func(t *testing.T) {
		local := &upstreamFixture{}
		config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, "http://127.0.0.1:9/v1/chat/completions")
		config.Deployments, config.Actions = config.Deployments[:1], config.Actions[:1]
		fixture := &v2Decisions{choose: fitPolicy}
		_, httpServer := v2Server(t, config, fixture)
		status, response := executeHTTP(t, httpServer.URL, "/v1/messages", over, "local-only")
		if status != http.StatusBadRequest || !strings.Contains(string(response), `"message":"prompt is too long: about `) || !strings.Contains(string(response), "prompt tokens (measured)") || !strings.Contains(string(response), "the largest maximum context length is 98304 tokens (qwen3.8-27b)") {
			t.Fatalf("status %d: %s", status, response)
		}
		if len(fixture.snapshot()) != 0 || local.count() != 0 {
			t.Fatalf("lookups=%d local=%d", len(fixture.snapshot()), local.count())
		}
	})
	t.Run("counted cloud and estimated local", func(t *testing.T) {
		local, cloud := &upstreamFixture{}, &upstreamFixture{}
		config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
		fixture := &v2Decisions{choose: fitPolicy}
		server, httpServer := v2Server(t, config, fixture)
		for _, deployment := range config.Deployments[1:] {
			if err := server.RegisterPromptCounter(deployment.ID, &fixedCounter{identity: promptIdentity(deployment), counts: []int{1_100_000}}); err != nil {
				t.Fatal(err)
			}
		}
		status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", over, "mixed")
		if status != http.StatusBadRequest || !strings.Contains(string(response), `"code":"context_exceeds_all_deployments"`) || !strings.Contains(string(response), "about 1100000 prompt tokens (measured)") || !strings.Contains(string(response), "1000000 tokens (kimi-k3)") {
			t.Fatalf("status %d: %s", status, response)
		}
		if len(fixture.snapshot()) != 0 || local.count() != 0 || cloud.count() != 0 {
			t.Fatalf("lookups=%d local=%d cloud=%d", len(fixture.snapshot()), local.count(), cloud.count())
		}
	})
	t.Run("largest window comes from config", func(t *testing.T) {
		config := capacityEstimateConfig(t, "http://127.0.0.1:8/v1/chat/completions", "http://127.0.0.1:9/v1/chat/completions")
		server, _ := v2Server(t, config, &v2Decisions{choose: fitPolicy})
		reason, source := CapacityContext, "configured"
		required, available := int64(114_000), int64(estimateLocalWindow)
		facts := map[string]*CapacityFact{
			localActionID:       {Reason: &reason, Required: &required, Available: &available, AvailableSource: &source, Provenance: capacityProvenanceEstimated, prompt: 109_904},
			cloudActionID:       runtimeContextRejection(config.Deployments[1], nil),
			secondCloudActionID: runtimeContextRejection(config.Deployments[2], nil),
		}
		if !server.exceedsAllDeployments(facts) {
			t.Fatal("every target reported a shortfall")
		}
		message := server.contextExceededMessage(facts)
		if !strings.Contains(message, "the largest maximum context length is 1000000 tokens (kimi-k3)") {
			t.Fatalf("message %q", message)
		}
		delete(facts, secondCloudActionID)
		if server.exceedsAllDeployments(facts) {
			t.Fatal("an unknown target must block the refusal")
		}
	})
}

func TestCapacityMeasuredShortfallEverywhereRefusesInEachWireFormat(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: fitPolicy}
	server, httpServer := v2Server(t, config, fixture, 1_100_000)
	for _, deployment := range config.Deployments[1:] {
		if err := server.RegisterPromptCounter(deployment.ID, &fixedCounter{identity: promptIdentity(deployment), counts: []int{1_100_000}}); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct{ path, body, code string }{
		{"/v1/chat/completions", chatTurn(userTask), `"code":"context_exceeds_all_deployments"`},
		{"/v1/responses", `{"model":"arc-1.0","max_output_tokens":64,"input":[{"role":"user","content":"Read the file."}]}`, `"code":"context_exceeds_all_deployments"`},
		{"/v1/messages", `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"Read the file."}]}`, `"type":"invalid_request_error"`},
	} {
		status, response := executeHTTP(t, httpServer.URL, tc.path, tc.body, "measured"+tc.path)
		if status != http.StatusBadRequest || !strings.Contains(string(response), tc.code) {
			t.Fatalf("%s status %d: %s", tc.path, status, response)
		}
		for _, part := range []string{`"message":"prompt is too long: about 1100000 prompt tokens (measured)`, "the largest maximum context length is 1000000 tokens (kimi-k3)", "Compact or shorten the conversation"} {
			if !strings.Contains(string(response), part) {
				t.Fatalf("%s: %s lacks %q", tc.path, response, part)
			}
		}
	}
	if len(fixture.snapshot()) != 0 || local.count() != 0 || cloud.count() != 0 {
		t.Fatalf("refusal contacted Decisions=%d local=%d cloud=%d", len(fixture.snapshot()), local.count(), cloud.count())
	}
}

func TestCapacityEstimateHandoffOutsideCacheBoundariesNeverReturnsLocal(t *testing.T) {
	fixtures := loadCapacityFixtures(t)
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := capacityEstimateConfig(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	// The service's model schedule is cache_boundaries: a tool-result turn is
	// not a model boundary, but a capacity handoff is still forced.
	fixture := &v2Decisions{choose: fitPolicy, mutate: func(response map[string]any) {
		response["receipt"].(map[string]any)["eligible"] = map[string]any{"model": "cache_boundaries", "thinking": "all_turns"}
	}}
	_, httpServer := v2Server(t, config, fixture)
	task := jsonMessage(t, "user", fixtures["code_30k"].text)
	result := toolResultMessage(t, fixtures["uvlock_120k"].text)
	turns := []struct {
		body, executed, decidedBy, event string
	}{
		{bigTurn(task), localActionID, "schedule/schedule", EventTaskStart},
		{bigTurn(task, toolCall, result), secondCloudActionID, "capacity_handoff/schedule", EventToolResult},
		{bigTurn(task, toolCall, result, answer, followUp), secondCloudActionID, "schedule/schedule", EventUserMessage},
	}
	for index, turn := range turns {
		status, header, response := executeWithHeaders(t, httpServer.URL, turn.body, "spill-estimate")
		if status != 200 || header.Get("X-Rayline-Executed-Action-Id") != turn.executed || header.Get("X-Rayline-Decided-By") != turn.decidedBy || !strings.HasPrefix(header.Get("X-Rayline-Event"), turn.event+"/") {
			t.Fatalf("turn %d status %d executed %q decided %q event %q: %.300s", index, status, header.Get("X-Rayline-Executed-Action-Id"), header.Get("X-Rayline-Decided-By"), header.Get("X-Rayline-Event"), response)
		}
	}
	lookups := fixture.snapshot()
	if len(lookups) != 3 {
		t.Fatalf("a pre-decision fact needs no re-consult; got %d lookups", len(lookups))
	}
	for index, lookup := range lookups[1:] {
		fact := lookup.Execution.Targets[0].Capacity
		if fact == nil || fact.Fits || *fact.Reason != CapacityContext || fact.Provenance != "local_measured" {
			t.Fatalf("turn %d local fact %s", index+1, mustJSON(fact))
		}
	}
	if served := lookups[2].Execution.LastServed; served == nil || served.ActionID != secondCloudActionID {
		t.Fatalf("last_served %+v", served)
	}
	if local.count() != 1 || cloud.count() != 2 {
		t.Fatalf("dispatch local=%d cloud=%d", local.count(), cloud.count())
	}
}

func TestByteBoundChargesInlineImagesNotTheirBytes(t *testing.T) {
	blob := strings.Repeat("iVBORw0KGgo", 30_000) // 330 kB of base64
	zero := 0.0
	bare := &CapacityEstimate{HeadroomRatio: &zero}
	messages := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + blob + `"}}`
	chat := `{"type":"image_url","image_url":{"url":"data:image/png;base64,` + blob + `"}}`
	for name, body := range map[string]string{"messages": messages, "chat": chat} {
		got := byteBoundTokens([]byte(body), bare) - estimateTemplateMarginTokens
		if got < 8192 || got > 8192+128 {
			t.Fatalf("%s: estimate %d, want the 8192-token image charge plus structure", name, got)
		}
	}
	// A PDF and the same payloads quoted inside a string are text, charged by
	// the byte.
	pdf := `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + blob + `"}}`
	escaped := `{"content":"{\"data\":\"` + blob + `\"} and \"data:image/png;base64,` + blob + `\""}`
	for name, body := range map[string]string{"pdf": pdf, "escaped": escaped} {
		if got := byteBoundTokens([]byte(body), bare); got < int64(len(blob)) {
			t.Fatalf("%s was discounted: %d", name, got)
		}
	}
	media := int64(100) // structure of the Messages image block is about 80 bytes
	if got := byteBoundTokens([]byte(messages), &CapacityEstimate{HeadroomRatio: &zero, MediaTokens: &media}) - estimateTemplateMarginTokens; got > 228 {
		t.Fatalf("media_tokens override ignored: %d", got)
	}
}

func TestCapacityEstimateConfigIsPerDeploymentAndValidated(t *testing.T) {
	var deployment Deployment
	if err := json.Unmarshal([]byte(`{"id":"local","capacity_estimate":{"headroom_ratio":0.3,"media_tokens":4096}}`), &deployment); err != nil {
		t.Fatal(err)
	}
	if estimate := deployment.CapacityEstimate; estimate == nil || *estimate.HeadroomRatio != 0.3 || *estimate.MediaTokens != 4096 {
		t.Fatalf("capacity_estimate %+v", deployment.CapacityEstimate)
	}
	t.Setenv("TEST_DECISIONS_KEY", "fixture-hosted-key-never-for-local")
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	bad := -0.5
	config := capacityEstimateConfig(t, "http://127.0.0.1:1/v1/chat/completions", "http://127.0.0.1:2/v1/chat/completions")
	config.Deployments[0].CapacityEstimate = &CapacityEstimate{HeadroomRatio: &bad}
	config.Decisions.URL = "http://127.0.0.1:3/v1/routes"
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewServer(config, registry, provider, testCoreToken); err == nil || !strings.Contains(err.Error(), "capacity_estimate") {
		t.Fatalf("invalid estimate admitted: %v", err)
	}
}

func TestLocalOpenAIStyleContextRefusalReconsultsToCloud(t *testing.T) {
	// A vLLM/MLX-style refusal from the local runtime (the count said it fits,
	// the runtime's real allocation is smaller) re-consults before any output,
	// as llama.cpp's refusal does.
	localCalls := 0
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		localCalls++
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"object":"error","message":"This model's maximum context length is 32768 tokens. However, you requested 40000 tokens in the messages.","type":"BadRequestError","param":null,"code":400}`)
	}))
	defer local.Close()
	cloud := &upstreamFixture{}
	config := capacityEstimateConfig(t, local.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: fitPolicy}
	_, httpServer := v2Server(t, config, fixture)
	status, header, response := executeWithHeaders(t, httpServer.URL, bigTurn(jsonMessage(t, "user", loadCapacityFixtures(t)["code_30k"].text)), "local-refuses")
	if status != 200 || header.Get("X-Rayline-Executed-Action-Id") == localActionID {
		t.Fatalf("status %d executed %q: %.300s", status, header.Get("X-Rayline-Executed-Action-Id"), response)
	}
	lookups := fixture.snapshot()
	if len(lookups) != 2 || lookups[1].Execution.Turn.Reconsult == nil || *lookups[1].Execution.Turn.Reconsult != "capacity" {
		t.Fatalf("lookups %d", len(lookups))
	}
	fact := lookups[1].Execution.Targets[0].Capacity
	if fact == nil || fact.Fits || fact.Provenance != "provider_reported" || fact.Required == nil {
		t.Fatalf("refusal fact %s", mustJSON(fact))
	}
	if localCalls != 1 || cloud.count() != 1 {
		t.Fatalf("local=%d cloud=%d", localCalls, cloud.count())
	}
}

func TestContextExceededMessageNeverUnderstatesTheRequest(t *testing.T) {
	config := capacityEstimateConfig(t, "http://127.0.0.1:8/v1/chat/completions", "http://127.0.0.1:9/v1/chat/completions")
	server, _ := v2Server(t, config, &v2Decisions{choose: fitPolicy})
	reason := CapacityContext
	required := int64(114_000)
	facts := map[string]*CapacityFact{
		localActionID:       {Reason: &reason, Required: &required, Provenance: capacityProvenanceMeasured, prompt: 109_904},
		cloudActionID:       runtimeContextRejection(config.Deployments[1], nil),
		secondCloudActionID: runtimeContextRejection(config.Deployments[2], nil),
	}
	message := server.contextExceededMessage(facts)
	if strings.Contains(message, "114000") || !strings.HasPrefix(message, "prompt is too long: ") || !strings.Contains(message, "the largest maximum context length is 1000000 tokens (kimi-k3)") {
		t.Fatalf("message %q", message)
	}
}
