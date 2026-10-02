package protocolcodec

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The provider charge is accounting evidence the Router reads, never part of
// the completion. A charge member of a shape the Router does not expect must
// leave that member unknown and the completion intact: before it was decoded,
// the member was pruned, and a paid, successful turn must not become a
// failure because the charge became typed.

// costProbe is one usage-cost fragment and what each member should decode to.
type costProbe struct {
	name     string
	fragment string
	charged  *float64
	upstream *float64
	byok     *bool
}

func costProbes() []costProbe {
	charged, upstream, byok := 0.0012, 0.001, true
	return []costProbe{
		{
			name: "well formed", fragment: `"cost":0.0012,"is_byok":true,"cost_details":{"upstream_inference_cost":0.001}`,
			charged: &charged, upstream: &upstream, byok: &byok,
		},
		{name: "cost as string", fragment: `"cost":"0.0012"`},
		{name: "cost as object", fragment: `"cost":{"amount":0.0012}`},
		{name: "is_byok as string", fragment: `"is_byok":"false"`},
		{name: "cost_details as array", fragment: `"cost_details":[0.001]`},
		{name: "upstream_inference_cost as string", fragment: `"cost_details":{"upstream_inference_cost":"0.001"}`},
	}
}

// costProbePaths decodes one fragment through every path that reads a usage
// object off a provider, and returns the usage each produced.
func costProbePaths(t *testing.T, fragment string) map[string]llmprotocol.Usage {
	t.Helper()
	engine := NewBuiltinEngine()
	streamPolicy := engine.providerStreamPolicy()
	results := map[string]llmprotocol.Usage{}

	translate := func(path string, source, target llmprotocol.WireFormat, body string) {
		result, err := engine.TranslateResponse(source, target, []byte(body), nil)
		if err != nil {
			t.Errorf("%s: a completion with an odd charge failed: %v", path, err)
			return
		}
		results[path] = result.Response.Usage
	}
	stream := func(path string, decoder llmprotocol.StreamDecoder, payload string) {
		events, _, err := decoder.Push([]byte(payload))
		if err != nil {
			t.Errorf("%s: a stream with an odd charge failed: %v", path, err)
			return
		}
		terminal := events[len(events)-1]
		if terminal.Type != llmprotocol.EventResponseCompleted || terminal.Usage == nil {
			t.Errorf("%s: terminal event = %+v", path, terminal)
			return
		}
		results[path] = *terminal.Usage
	}
	streamContext := llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"}

	translate("chat response", llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		`{"id":"c1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4,`+fragment+`}}`)
	stream("chat stream", OpenAIChatCodec{}.NewDecoder(streamContext, streamPolicy),
		"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"+
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4,"+fragment+"}}\n\n"+
			"data: [DONE]\n\n")
	translate("messages response", llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],`+
			`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1,`+fragment+`}}`)
	messagesStream := func(startUsage, deltaUsage string) string {
		return "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0" + startUsage + "}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1" + deltaUsage + "}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	}
	stream("messages stream (message_delta)", AnthropicMessagesCodec{}.NewDecoder(streamContext, streamPolicy),
		messagesStream("", ","+fragment))
	stream("messages stream (message_start)", AnthropicMessagesCodec{}.NewDecoder(streamContext, streamPolicy),
		messagesStream(","+fragment, ""))
	responsesBody := `{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"completed",` +
		`"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],` +
		`"usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4,` + fragment + `}}`
	translate("responses response", llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, responsesBody)
	frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	stream("responses stream (response.completed)", OpenAIResponsesCodec{}.NewDecoder(streamContext, streamPolicy),
		frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"in_progress","output":[]}}`)+
			frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}`)+
			frame("response.content_part.added", `{"type":"response.content_part.added","sequence_number":2,"output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":"","annotations":[]}}`)+
			frame("response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"ok"}`)+
			frame("response.output_text.done", `{"type":"response.output_text.done","sequence_number":4,"output_index":0,"content_index":0,"item_id":"msg_1","text":"ok"}`)+
			frame("response.content_part.done", `{"type":"response.content_part.done","sequence_number":5,"output_index":0,"content_index":0,"item_id":"msg_1","part":{"type":"output_text","text":"ok","annotations":[]}}`)+
			frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":6,"output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}}`)+
			frame("response.completed", `{"type":"response.completed","sequence_number":7,"response":`+responsesBody+`}`))
	return results
}

func TestProviderCostOfAnyShapeNeverFailsACompletion(t *testing.T) {
	for _, probe := range costProbes() {
		t.Run(strings.ReplaceAll(probe.name, " ", "_"), func(t *testing.T) {
			results := costProbePaths(t, probe.fragment)
			if len(results) != 7 {
				t.Fatalf("decoded %d of 7 paths", len(results))
			}
			for path, usage := range results {
				if usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != 1 {
					t.Errorf("%s: the counts beside the charge were lost: %+v", path, usage)
				}
				cost := usage.ProviderCost
				if !sameFloat(cost.Charged, probe.charged) || !sameFloat(cost.UpstreamInference, probe.upstream) ||
					!sameBool(cost.BYOK, probe.byok) {
					t.Errorf("%s: charge = %s, want charged %s upstream %s byok %s", path,
						describeCost(cost), describeFloat(probe.charged), describeFloat(probe.upstream), describeBool(probe.byok))
				}
			}
		})
	}
}

func sameFloat(got, want *float64) bool {
	return (got == nil) == (want == nil) && (got == nil || *got == *want)
}

func sameBool(got, want *bool) bool {
	return (got == nil) == (want == nil) && (got == nil || *got == *want)
}

func describeFloat(value *float64) string {
	if value == nil {
		return "nil"
	}
	return strconvFloat(*value)
}

func describeBool(value *bool) string {
	if value == nil {
		return "nil"
	}
	if *value {
		return "true"
	}
	return "false"
}

func describeCost(cost llmprotocol.ProviderCost) string {
	return "charged " + describeFloat(cost.Charged) + " upstream " + describeFloat(cost.UpstreamInference) + " byok " + describeBool(cost.BYOK)
}

func strconvFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
