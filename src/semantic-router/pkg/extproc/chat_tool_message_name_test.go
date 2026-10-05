package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A Responses history whose call came from a Claude worker, as it reads after
// the policy switched workers mid-episode.
const crossModelToolHistoryResponses = `{
	"model":"m",
	"input":[
		{"type":"message","role":"user","content":"list the files"},
		{"type":"function_call","call_id":"toolu_01AbCdEf","name":"shell","arguments":"{\"cmd\":\"ls\"}"},
		{"type":"function_call_output","call_id":"toolu_01AbCdEf","output":"a.txt"}
	],
	"tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]
}`

// The same history from a Chat client, whose own bytes carry no name.
const crossModelToolHistoryChat = `{
	"model":"m",
	"messages":[
		{"role":"user","content":"list the files"},
		{"role":"assistant","tool_calls":[{"id":"toolu_01AbCdEf","type":"function","function":{"name":"shell","arguments":"{\"cmd\":\"ls\"}"}}]},
		{"role":"tool","tool_call_id":"toolu_01AbCdEf","content":"a.txt"}
	],
	"tools":[{"type":"function","function":{"name":"shell","parameters":{"type":"object"}}}]
}`

// dispatchToChatWorker sends body through ingress, provider dispatch and
// encode to a Chat worker whose provider profile has the given type, and
// returns the upstream body.
func dispatchToChatWorker(t *testing.T, source llmprotocol.WireFormat, providerType, body string) []byte {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	profile := router.Config.ProviderProfiles["provider"]
	profile.Type = providerType
	router.Config.ProviderProfiles["provider"] = profile
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: source,
		RequestID: "chat-tool-message-name", TraceContext: context.Background(),
	}
	request, immediate := router.prepareProtocolRequest([]byte(body), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("ingress refused %s: %+v", body, ctx.ImmediateProtocolError)
	}
	dispatch, err := router.prepareProviderDispatch(request, logicalModel, "", false, ctx)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	response, err := router.finalizeProviderDispatchResponse(dispatch, router.buildProviderDispatchResponse(dispatch, ctx), ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	return response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
}

func dispatchedToolMessage(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var wire struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("upstream body: %v: %s", err, body)
	}
	for _, message := range wire.Messages {
		if string(message["role"]) == `"tool"` {
			return message
		}
	}
	t.Fatalf("no tool message in %s", body)
	return nil
}

func TestChatToolMessageIsNamedOnlyForAnOpenRouterBackend(t *testing.T) {
	for name, tc := range map[string]struct {
		source   llmprotocol.WireFormat
		body     string
		provider string
		want     string
	}{
		"Responses client, OpenRouter": {llmprotocol.OpenAIResponsesV1, crossModelToolHistoryResponses, "openrouter", `"shell"`},
		"Responses client, OpenAI":     {llmprotocol.OpenAIResponsesV1, crossModelToolHistoryResponses, "openai", ""},
		"Chat client, OpenRouter":      {llmprotocol.OpenAIChatV1, crossModelToolHistoryChat, "openrouter", `"shell"`},
		"Chat client, OpenAI":          {llmprotocol.OpenAIChatV1, crossModelToolHistoryChat, "openai", ""},
	} {
		t.Run(name, func(t *testing.T) {
			tool := dispatchedToolMessage(t, dispatchToChatWorker(t, tc.source, tc.provider, tc.body))
			if string(tool["tool_call_id"]) != `"toolu_01AbCdEf"` || string(tool["name"]) != tc.want {
				t.Fatalf("tool message = %s %s, want name %q", tool["tool_call_id"], tool["name"], tc.want)
			}
		})
	}
}

// A direct OpenAI Chat dispatch from a Chat client is the client's own bytes,
// unchanged: naming is never a reason to re-encode one.
func TestChatToolMessageNamingLeavesADirectOpenAIDispatchUnchanged(t *testing.T) {
	body := dispatchToChatWorker(t, llmprotocol.OpenAIChatV1, "openai", crossModelToolHistoryChat)
	var sent, client map[string]json.RawMessage
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(crossModelToolHistoryChat), &client); err != nil {
		t.Fatal(err)
	}
	var sentMessages, clientMessages any
	_ = json.Unmarshal(sent["messages"], &sentMessages)
	_ = json.Unmarshal(client["messages"], &clientMessages)
	sentJSON, _ := json.Marshal(sentMessages)
	clientJSON, _ := json.Marshal(clientMessages)
	if string(sentJSON) != string(clientJSON) {
		t.Fatalf("messages changed:\n got %s\nwant %s", sentJSON, clientJSON)
	}
}

// A Chat client's request that nothing else changed would be replayed as its
// own bytes, which name no tool; on an OpenRouter dispatch it is encoded
// instead, so the tool message carries the name.
func TestChatToolMessageNamingRetiresReplayOfTheClientBytes(t *testing.T) {
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(
		llmprotocol.OpenAIChatV1, []byte(crossModelToolHistoryChat))
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &request, ProtocolEnvelope: envelope, DispatchNamesToolResults: true,
	}
	body, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if tool := dispatchedToolMessage(t, body); string(tool["name"]) != `"shell"` {
		t.Fatalf("tool message name = %s, want \"shell\": %s", tool["name"], body)
	}
	if request.NamesToolResults || request.Generation != envelope.Generation {
		t.Fatal("the dispatch flag leaked into the request the client's response is rendered from")
	}
}
