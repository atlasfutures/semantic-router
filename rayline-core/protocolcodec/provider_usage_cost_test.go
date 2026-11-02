package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// OpenRouter states what it charged on the usage object. The Router keeps
// that number as the provider's own, beside the counts, on every protocol
// OpenRouter serves.
func TestOpenRouterChatResponseCarriesProviderCost(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		loadProviderFixture(t, openRouterResponseReasoning), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertProviderCost(t, result.Response.Usage.ProviderCost, 5.824944e-05, 5.824944e-05, false)
}

// The charge arrives on the final usage frame; a later totals-only frame that
// does not restate it must not erase it.
func TestOpenRouterChatStreamKeepsProviderCostAcrossUsageFrames(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"r1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"model\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":68,\"total_tokens\":79," +
			"\"cost\":0.000242618,\"is_byok\":false,\"cost_details\":{\"upstream_inference_cost\":0.000242618}}}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"model\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":68,\"total_tokens\":79}}\n\n" +
			"data: [DONE]\n\n",
	)
	events, _, err := decoder.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	terminal := events[len(events)-1]
	if terminal.Type != llmprotocol.EventResponseCompleted || terminal.Usage == nil {
		t.Fatalf("terminal event = %#v", terminal)
	}
	assertProviderCost(t, terminal.Usage.ProviderCost, 0.000242618, 0.000242618, false)
}

func TestOpenRouterMessagesResponseCarriesProviderCost(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"anthropic/claude-opus-5",` +
		`"content":[{"type":"text","text":"Fixed."}],"stop_reason":"end_turn","stop_sequence":null,` +
		`"usage":{"input_tokens":12,"output_tokens":3,"cost":0.0004,"is_byok":true,` +
		`"cost_details":{"upstream_inference_cost":0.0039}}}`)
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertProviderCost(t, result.Response.Usage.ProviderCost, 0.0004, 0.0039, true)
}

// A provider that states no charge leaves every member unknown. An absent
// charge is not a free call.
func TestUsageWithoutProviderCostLeavesItUnknown(t *testing.T) {
	body := []byte(`{"id":"c1","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`)
	response, _, _, err := OpenAIChatCodec{}.DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if cost := response.Usage.ProviderCost; cost.Charged != nil || cost.UpstreamInference != nil || cost.BYOK != nil {
		t.Fatalf("provider cost = %+v, want every member unknown", cost)
	}
}

// The charge is accounting evidence for the Router. Re-encoding for a client
// never publishes it.
func TestProviderCostIsNeverPublishedToAClient(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		loadProviderFixture(t, openRouterResponseReasoning), renameModel("public-model"),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, member := range [][]byte{[]byte(`"cost"`), []byte(`"is_byok"`), []byte(`"cost_details"`)} {
		if bytes.Contains(result.Body, member) {
			t.Fatalf("client body publishes %s: %s", member, result.Body)
		}
	}
}

func assertProviderCost(t *testing.T, cost llmprotocol.ProviderCost, charged, upstream float64, byok bool) {
	t.Helper()
	if cost.Charged == nil || *cost.Charged != charged {
		t.Fatalf("charged = %v, want %v", cost.Charged, charged)
	}
	if cost.UpstreamInference == nil || *cost.UpstreamInference != upstream {
		t.Fatalf("upstream inference = %v, want %v", cost.UpstreamInference, upstream)
	}
	if cost.BYOK == nil || *cost.BYOK != byok {
		t.Fatalf("byok = %v, want %v", cost.BYOK, byok)
	}
}

// openRouterMessagesCostStream is an OpenRouter Messages API stream whose
// final usage states the full charge.
const openRouterMessagesCostStream = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"openai/gpt-6-luna","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Fixed."}}` + "\n\n" +
	"event: content_block_stop\n" +
	`data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":38,"output_tokens":79,` +
	`"cost":0.0000433,"is_byok":false,"cost_details":{"upstream_inference_cost":0.0000433,"upstream_inference_prompt_cost":0.0000038,"upstream_inference_completions_cost":0.0000395}}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

func providerCostEngine(t *testing.T) *Engine {
	t.Helper()
	policy := llmprotocol.DefaultPolicy()
	policy.PublishProviderCost = true
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	return engine
}

func translateCostStream(t *testing.T, engine *Engine, target llmprotocol.WireFormat, stream string) []byte {
	t.Helper()
	includeUsage := true
	translator, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
		Options: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := translator.Push([]byte(stream))
	if err != nil {
		t.Fatal(err)
	}
	final, _, _, err := translator.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Join(append(frames, final...), nil)
}

const restatedProviderCost = `"cost":0.0000433,"is_byok":false,"cost_details":{"upstream_inference_completions_cost":0.0000395,` +
	`"upstream_inference_cost":0.0000433,"upstream_inference_prompt_cost":0.0000038}`

// A Router that publishes the charge to its own accounting front restates it,
// as OpenRouter spells it, on a stream it re-encodes from another protocol.
func TestPublishedProviderCostIsRestatedOnAnEncodedStream(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		output := translateCostStream(t, providerCostEngine(t), target, openRouterMessagesCostStream)
		if !bytes.Contains(output, []byte(restatedProviderCost)) {
			t.Fatalf("%s stream does not restate the charge: %s", target, output)
		}
		if bytes.Contains(translateCostStream(t, NewBuiltinEngine(), target, openRouterMessagesCostStream), []byte(`"cost"`)) {
			t.Fatalf("%s stream publishes the charge by default", target)
		}
	}
}

// A charge the upstream did not state is not restated, not even as 0.
func TestPublishedProviderCostLeavesAnUnstatedChargeOut(t *testing.T) {
	stream := strings.Replace(openRouterMessagesCostStream,
		`,"cost":0.0000433,"is_byok":false,"cost_details":{"upstream_inference_cost":0.0000433,"upstream_inference_prompt_cost":0.0000038,"upstream_inference_completions_cost":0.0000395}`, "", 1)
	output := translateCostStream(t, providerCostEngine(t), llmprotocol.OpenAIChatV1, stream)
	if !bytes.Contains(output, []byte(`"usage"`)) {
		t.Fatalf("no usage chunk: %s", output)
	}
	for _, member := range []string{`"cost"`, `"is_byok"`, `"cost_details"`} {
		if bytes.Contains(output, []byte(member)) {
			t.Fatalf("unstated %s restated: %s", member, output)
		}
	}
}

// A buffered response re-encoded for the Router's front restates the charge
// on Chat and Responses alike.
func TestPublishedProviderCostIsRestatedOnAnEncodedResponse(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"openai/gpt-6-luna",` +
		`"content":[{"type":"text","text":"Fixed."}],"stop_reason":"end_turn","stop_sequence":null,` +
		`"usage":{"input_tokens":38,"output_tokens":79,"cost":0.0000433,"is_byok":false,` +
		`"cost_details":{"upstream_inference_cost":0.0000433,"upstream_inference_prompt_cost":0.0000038,"upstream_inference_completions_cost":0.0000395}}}`)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		result, err := providerCostEngine(t).TranslateResponse(llmprotocol.AnthropicMessagesV1, target, body, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(result.Body, []byte(restatedProviderCost)) {
			t.Fatalf("%s body does not restate the charge: %s", target, result.Body)
		}
	}
}
