package execution

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

func TestClaudeTrailingSystemUsesQualifiedSharedFold(t *testing.T) {
	var observed [][]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		observed = append(observed, body.Messages)
		_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`)
	}))
	defer upstream.Close()
	config := testConfig(t, upstream.URL, false)
	config.Deployments[0].InbandSystem = thinkingcontrol.InbandSystemFold
	config.Deployments[0].RuntimeIdentity += ":" + thinkingcontrol.InbandSystemFold
	core, server := executorWithHTTPDecisions(t, config, &decisionCapture{})
	first := `{"role":"user","content":"Read the requested file."},{"role":"system","content":[{"type":"text","text":"Environment at this turn.","cache_control":{"type":"ephemeral"}}]}`
	body := `{"model":"arc-1.0","max_tokens":64,"system":[{"type":"text","text":"Global instructions."},{"type":"text","text":"More instructions."}],"messages":[` + first + `]}`
	status, result := executeHTTP(t, server.URL, "/v1/messages", body, "fold")
	if status != 200 {
		t.Fatalf("first execution: %d %s", status, result)
	}
	next := strings.TrimSuffix(body, "]}") + `,{"role":"assistant","content":"ready"},{"role":"user","content":"Continue."},{"role":"system","content":"New turn environment."}]}`
	status, result = executeHTTP(t, server.URL, "/v1/messages", next, "fold")
	if status != 200 {
		t.Fatalf("second execution: %d %s", status, result)
	}
	if len(observed) != 2 || len(observed[0]) != 2 || len(observed[1]) != 4 {
		t.Fatalf("unexpected message counts: %v", observed)
	}
	if !reflect.DeepEqual(observed[0], observed[1][:2]) {
		t.Fatal("extending the transcript rewrote the prior folded prefix")
	}
	var user struct {
		Role    string `json:"role"`
		Content []struct {
			Text  string          `json:"text"`
			Cache json.RawMessage `json:"cache_control"`
		} `json:"content"`
	}
	if err := json.Unmarshal(observed[0][1], &user); err != nil {
		t.Fatal(err)
	}
	if user.Role != "user" || len(user.Content) != 2 || user.Content[0].Text != "Read the requested file." || user.Content[1].Text != "Environment at this turn." || string(user.Content[1].Cache) != `{"type":"ephemeral"}` {
		t.Fatalf("text/order/cache units changed: %s", observed[0][1])
	}
	selected := core.actions[config.Actions[0].ID]
	if selected.control.ID() != config.Actions[0].ControlID || selected.control.Instruction != nil {
		t.Fatal("compatibility rule changed the admitted control")
	}
}

func TestInbandSystemRequiresExplicitLocalDefaultProfile(t *testing.T) {
	valid := Deployment{Kind: "local", Protocol: llmprotocol.OpenAIChatV1, InbandSystem: thinkingcontrol.InbandSystemFold, RuntimeIdentity: "qualified:" + thinkingcontrol.InbandSystemFold}
	control := thinkingcontrol.Control{Native: "default"}
	if err := validateInbandSystem(valid, control); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*Deployment, *thinkingcontrol.Control){
		"cloud":         func(d *Deployment, c *thinkingcontrol.Control) { d.Kind = "cloud" },
		"messages":      func(d *Deployment, c *thinkingcontrol.Control) { d.Protocol = llmprotocol.AnthropicMessagesV1 },
		"unqualified":   func(d *Deployment, c *thinkingcontrol.Control) { d.RuntimeIdentity = "old" },
		"unknown rule":  func(d *Deployment, c *thinkingcontrol.Control) { d.InbandSystem = "other" },
		"native effort": func(d *Deployment, c *thinkingcontrol.Control) { c.Native = "high" },
		"private append": func(d *Deployment, c *thinkingcontrol.Control) {
			c.Instruction = &thinkingcontrol.Instruction{Level: "up"}
		},
	} {
		t.Run(name, func(t *testing.T) {
			d, c := valid, control
			change(&d, &c)
			if validateInbandSystem(d, c) == nil {
				t.Fatal("unqualified composition accepted")
			}
		})
	}
}

// Claude Code's shape on a model it does not know: the system prompt, the
// user's turn, then a system message after it (and after each tool result on
// later turns). A local Chat deployment receives exactly one leading system
// message without any inband_system configuration, and each turn extends the
// previous folded prefix unchanged.
func TestLocalChatFoldsClaudeCodeTrailingSystemWithoutConfiguration(t *testing.T) {
	var observed [][]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		observed = append(observed, body.Messages)
		_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`)
	}))
	defer upstream.Close()
	config := testConfig(t, upstream.URL, false)
	_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})
	tools := `"tools":[{"name":"read","description":"Read a file.","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]`
	turn := `{"role":"user","content":[{"type":"text","text":"<system-reminder>Context.</system-reminder>"},{"type":"text","text":"Fix parser.py."}]},{"role":"system","content":[{"type":"text","text":"# Environment\nWorking directory: /work"}]}`
	first := `{"model":"rayline-auto","max_tokens":64,"system":[{"type":"text","text":"You are a coding agent."}],` + tools + `,"messages":[` + turn + `]}`
	second := strings.TrimSuffix(first, "]}") + `,{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read","input":{"path":"parser.py"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"def parse(): pass"}]},{"role":"system","content":[{"type":"text","text":"# Environment\nWorking directory: /work"}]}]}`
	for _, body := range []string{first, second} {
		if status, result := executeHTTP(t, server.URL, "/v1/messages", body, "claude-shape"); status != 200 {
			t.Fatalf("execution: %d %s", status, result)
		}
	}
	if len(observed) != 2 {
		t.Fatalf("upstream saw %d requests", len(observed))
	}
	for call, messages := range observed {
		for position, raw := range messages {
			var message struct {
				Role string `json:"role"`
			}
			if err := json.Unmarshal(raw, &message); err != nil {
				t.Fatal(err)
			}
			if (message.Role == "system") != (position == 0) {
				t.Fatalf("call %d: %s at %d; want one leading system message: %s", call, message.Role, position, messages)
			}
		}
	}
	if len(observed[1]) < len(observed[0]) || !reflect.DeepEqual(observed[0], observed[1][:len(observed[0])]) {
		t.Fatalf("the next turn rewrote the folded prefix:\n%s\n%s", observed[0], observed[1])
	}
	var user struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(observed[0][1], &user); err != nil {
		t.Fatal(err)
	}
	if len(user.Content) != 3 || user.Content[1].Text != "Fix parser.py." || !strings.HasPrefix(user.Content[2].Text, "# Environment") {
		t.Fatalf("the trailing system text did not fold after the turn's own content: %s", observed[0][1])
	}
}

func TestOnlyLocalChatDefaultControlFoldsWithoutConfiguration(t *testing.T) {
	local := Deployment{Kind: "local", Protocol: llmprotocol.OpenAIChatV1}
	if !foldsInbandSystem(local, thinkingcontrol.Control{Native: "default"}) {
		t.Fatal("local Chat default control is not folded")
	}
	cloud, messages := local, local
	cloud.Kind = "cloud"
	messages.Protocol = llmprotocol.AnthropicMessagesV1
	for name, deployment := range map[string]Deployment{"cloud": cloud, "messages": messages} {
		if foldsInbandSystem(deployment, thinkingcontrol.Control{Native: "default"}) {
			t.Fatalf("%s deployment folded", name)
		}
	}
	if foldsInbandSystem(local, thinkingcontrol.Control{Native: "default", Instruction: &thinkingcontrol.Instruction{Level: "up"}}) {
		t.Fatal("a steering control folded")
	}
	// A body without an in-band system message is never rewritten.
	body := []byte(`{"messages":[{"role":"system","content":"s"},{"role":"user","content":"<b>"}]}`)
	selected := admittedAction{deployment: local, control: thinkingcontrol.Control{Native: "default"}, format: thinkingcontrol.FormatChat}
	if got, err := foldInbandSystem(body, local, selected); err != nil || string(got) != string(body) {
		t.Fatalf("unchanged body rewritten: %s %v", got, err)
	}
}
