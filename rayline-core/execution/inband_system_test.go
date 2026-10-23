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
