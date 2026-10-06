//go:build !windows && cgo

package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// responsesToolsClientTools is a Responses tools array as a client might
// send it: pretty-printed, with characters json.Marshal would escape, so any
// re-encoding on the way to the service shows as a byte difference.
const responsesToolsClientTools = "[\n  { \"type\" : \"function\", \"name\" : \"shell\",\n" +
	"    \"description\" : \"Run <cmd> & read its output\",\n" +
	"    \"parameters\" : {\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}}} },\n" +
	"  {\"type\":\"web_search\"}\n]"

func newResponsesToolsRouter(t *testing.T) (*OpenAIRouter, *fakePolicyService) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"off":        policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off": policyAction("claude-off", "none", "claude-opus-5", policyTestEffort("none"), nil, ""),
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage,
		[]string{actions["think"].ActionID, actions["off"].ActionID, actions["claude"].ActionID, actions["claude-off"].ActionID})
	router, err := NewOpenAIRouter(writePolicyDispatchConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["think"].ActionID })
	return router, fake
}

// decideRequestMember is one member of the decide body's request object, as
// the service received it, and whether it was present at all.
func decideRequestMember(t *testing.T, body []byte, name string) (json.RawMessage, bool) {
	t.Helper()
	var decide struct {
		Request map[string]json.RawMessage `json:"request"`
	}
	if err := json.Unmarshal(body, &decide); err != nil {
		t.Fatalf("decide body is not JSON: %v", err)
	}
	member, present := decide.Request[name]
	return member, present
}

// arc-0.3 encodes the harness shell, so a Responses decide must carry the
// client's tools: without them the service cannot tell the tool-definition
// coverage and refuses the turn (runtime_refused). The tools reach the
// service byte for byte as the client sent them.
func TestPolicyResponsesDecideForwardsTheClientsTools(t *testing.T) {
	router, fake := newResponsesToolsRouter(t)
	client := "{\"model\":\"auto\",\"instructions\":\"You are Codex.\",\"tools\" : " + responsesToolsClientTools +
		",\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":\"List the files.\"}]}"
	dispatchPolicyClientRequest(t, router, "responses-tools", "/v1/responses", client)

	bodies := fake.receivedBodies()
	if len(bodies) != 1 {
		t.Fatalf("%d decide calls, want 1", len(bodies))
	}
	tools, present := decideRequestMember(t, bodies[0], "tools")
	if !present || !bytes.Equal(tools, []byte(responsesToolsClientTools)) {
		t.Fatalf("decide request.tools = %q (present %v), want the client's bytes %q", tools, present, responsesToolsClientTools)
	}
	if sent := fake.received()[0]; !bytes.Equal(sent.Request.Tools, []byte(responsesToolsClientTools)) {
		t.Fatalf("decoded request.tools = %s", sent.Request.Tools)
	}
}

// A Responses request without tools is decided exactly as before: the
// optional member is omitted, not sent as null or [].
func TestPolicyResponsesDecideOmitsAbsentTools(t *testing.T) {
	router, fake := newResponsesToolsRouter(t)
	for index, client := range []string{
		`{"model":"auto","instructions":"You are Codex.","input":[{"type":"message","role":"user","content":"hi"}]}`,
		`{"model":"auto","tools":null,"input":[{"type":"message","role":"user","content":"hi"}]}`,
	} {
		dispatchPolicyClientRequest(t, router, "responses-no-tools-"+string(rune('a'+index)), "/v1/responses", client)
		bodies := fake.receivedBodies()
		body := bodies[len(bodies)-1]
		if tools, present := decideRequestMember(t, body, "tools"); present {
			t.Fatalf("client %d: decide request carries tools %s: %s", index, tools, body)
		}
		if !bytes.Contains(body, []byte(`"request":{"input":[`)) {
			t.Fatalf("client %d: decide body is not the Responses shape: %s", index, body)
		}
	}
}

// A previous_response_id turn resolves its earlier items from the store, but
// the tools it forwards are the ones this request carries.
func TestPolicyResponsesDecideForwardsThisTurnsToolsOnAStoredHistory(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	current := `[{"type":"function","name":"apply_patch","parameters":{"type":"object"}}]`
	body := []byte(`{"model":"auto","previous_response_id":"resp_1","tools":` + current + `,"input":"and then?"}`)
	items := []json.RawMessage{
		json.RawMessage(`{"type":"message","role":"user","content":"List the files."}`),
		json.RawMessage(`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"README.md"}]}`),
		json.RawMessage(`{"type":"message","role":"user","content":"and then?"}`),
	}
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), State: state, RawRequest: body,
			RequestFormat: policyFormatResponses, PolicyInput: items,
		},
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.receivedBodies()[0]
	tools, present := decideRequestMember(t, sent, "tools")
	if !present || string(tools) != current {
		t.Fatalf("decide request.tools = %s (present %v), want this turn's %s", tools, present, current)
	}
	if input, _ := decideRequestMember(t, sent, "input"); !strings.Contains(string(input), "List the files.") {
		t.Fatalf("decide request.input lost the stored history: %s", input)
	}
}

// Messages and Chat decides keep their shape: system, tools and messages,
// each the client's bytes, in that order and nothing else.
func TestPolicyMessagesAndChatDecidesAreUnchanged(t *testing.T) {
	tools := `[{"name":"bash","description":"run <cmd> & see","input_schema":{"type":"object"}}]`
	messages := `[{"role":"user","content":"fix <this> & that"}]`
	for _, test := range []struct {
		format, body, want string
	}{
		{
			format: policyFormatAnthropic,
			body:   `{"model":"auto","max_tokens":1024,"system":"be brief","tools":` + tools + `,"messages":` + messages + `}`,
			want:   `{"system":"be brief","tools":` + tools + `,"messages":` + messages + `}`,
		},
		{
			format: policyFormatOpenAI,
			body:   `{"model":"auto","tools":` + tools + `,"messages":` + messages + `}`,
			want:   `{"system":null,"tools":` + tools + `,"messages":` + messages + `}`,
		},
	} {
		t.Run(test.format, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, "")
			bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
			fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
			state, _ := raylinearc.NewEpisodeState(2)
			_, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
				DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs,
				RaylineARC: &selection.RaylineARCSelectionContext{
					EpisodeIDHash: strings.Repeat("e", 64), State: state,
					RawRequest: []byte(test.body), RequestFormat: test.format,
				},
			})
			if err != nil {
				t.Fatalf("select: %v", err)
			}
			body := fixture.fake.receivedBodies()[0]
			// request is the envelope's last member.
			if !bytes.HasSuffix(body, []byte(`,"request":`+test.want+`}`)) {
				t.Fatalf("decide request differs:\n got %s\nwant %s", body, test.want)
			}
		})
	}
}
