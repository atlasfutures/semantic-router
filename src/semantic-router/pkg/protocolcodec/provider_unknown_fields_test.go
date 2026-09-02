package protocolcodec

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The bodies under testdata/provider/openrouter were captured from OpenRouter
// on 2026-09-02 for the two arc arms the dev cell routes to. They carry seven
// members the Chat wire contract does not name, at four nesting levels. Before
// this fix every one of them failed the whole response.
const (
	openRouterResponse          = "chat-response.json"
	openRouterResponseReasoning = "chat-response-reasoning.json"
	openRouterStream            = "chat-stream.sse"
	openRouterStreamReasoning   = "chat-stream-reasoning.sse"
)

func loadProviderFixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "provider", "openrouter", name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func renameModel(name string) ResponseMutation {
	return func(response *llmprotocol.Response) error {
		response.Model = name
		return nil
	}
}

// A buffered OpenRouter response must survive translation to the Anthropic
// client format. This is the path an Anthropic-speaking client takes.
func TestOpenRouterResponseTranslatesToAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, fixture := range []string{openRouterResponse, openRouterResponseReasoning} {
		t.Run(fixture, func(t *testing.T) {
			result, err := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
				loadProviderFixture(t, fixture), renameModel("public-model"),
			)
			if err != nil {
				t.Fatalf("translate to Anthropic: %v", err)
			}
			if len(result.Body) == 0 {
				t.Fatal("translated Anthropic body is empty")
			}
		})
	}
}

// The same body must also survive the same-format Chat path. Model renaming
// mutates the response, so the byte-identical replay path does not apply and
// the body is decoded strictly.
func TestOpenRouterResponsePassesThroughChat(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, fixture := range []string{openRouterResponse, openRouterResponseReasoning} {
		t.Run(fixture, func(t *testing.T) {
			result, err := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
				loadProviderFixture(t, fixture), renameModel("public-model"),
			)
			if err != nil {
				t.Fatalf("translate to Chat: %v", err)
			}
			if result.Response.Model != "public-model" {
				t.Fatalf("model is %q, want the routed public model", result.Response.Model)
			}
		})
	}
}

// The two captured streams are the reasoning arms of the arc cell. Both end
// the same way: a chunk with finish_reason "stop" and an empty delta, then a
// SECOND chunk repeating that finish_reason with another empty delta and the
// usage object, then [DONE].
var providerStreams = map[string]struct {
	fixture                        string
	promptTokens, completionTokens int64
	totalTokens                    int64
}{
	openRouterStream:          {openRouterStream, 11, 68, 79},
	openRouterStreamReasoning: {openRouterStreamReasoning, 258, 22, 280},
}

// runProviderStream drives the whole pipeline over one captured stream: every
// frame including the trailing usage chunk and [DONE], nothing withheld.
func runProviderStream(t *testing.T, fixture string, target llmprotocol.WireFormat) ([]byte, []llmprotocol.Event) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, pushErr := stream.Push(loadProviderFixture(t, fixture))
	if pushErr != nil {
		t.Fatalf("push %s: %v", fixture, pushErr)
	}
	finalFrames, finalEvents, _, finalErr := stream.Finalize(nil)
	if finalErr != nil {
		t.Fatalf("finalize %s: %v", fixture, finalErr)
	}
	return append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...), append(events, finalEvents...)
}

// terminalUsage returns the usage the stream settled on, from its completion
// event. The trailing chunk is the only frame that carries token counts, so an
// absent or zero usage here means the stream died before ingesting it.
func terminalUsage(t *testing.T, events []llmprotocol.Event) llmprotocol.Usage {
	t.Helper()
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseCompleted && event.Usage != nil {
			return *event.Usage
		}
	}
	t.Fatal("stream emitted no completion event carrying usage")
	return llmprotocol.Usage{}
}

// An Anthropic client must see the whole terminal sequence: the text block
// closes, then message_delta carries the stop reason and usage, then
// message_stop. Before this fix the stream died after content_block_stop.
func TestOpenRouterStreamCompletesForAnthropicClients(t *testing.T) {
	for name, want := range providerStreams {
		t.Run(name, func(t *testing.T) {
			body, events := runProviderStream(t, want.fixture, llmprotocol.AnthropicMessagesV1)
			for _, marker := range []string{"event: content_block_stop", "event: message_delta", "event: message_stop"} {
				if !bytes.Contains(body, []byte(marker)) {
					t.Fatalf("Anthropic stream is missing %q:\n%s", marker, body)
				}
			}
			if bytes.Index(body, []byte("event: message_delta")) > bytes.Index(body, []byte("event: message_stop")) {
				t.Fatal("Anthropic stream emitted message_stop before message_delta")
			}
			assertStreamUsage(t, terminalUsage(t, events), want.promptTokens, want.completionTokens, want.totalTokens)
		})
	}
}

// A Chat client must see its own terminal chunk and the [DONE] sentinel.
func TestOpenRouterStreamCompletesForChatClients(t *testing.T) {
	for name, want := range providerStreams {
		t.Run(name, func(t *testing.T) {
			body, events := runProviderStream(t, want.fixture, llmprotocol.OpenAIChatV1)
			if !bytes.Contains(body, []byte(`"finish_reason":"stop"`)) {
				t.Fatalf("Chat stream never emitted a terminal chunk:\n%s", body)
			}
			if !bytes.Contains(body, []byte("data: [DONE]")) {
				t.Fatalf("Chat stream never emitted [DONE]:\n%s", body)
			}
			assertStreamUsage(t, terminalUsage(t, events), want.promptTokens, want.completionTokens, want.totalTokens)
		})
	}
}

func assertStreamUsage(t *testing.T, usage llmprotocol.Usage, prompt, completion, total int64) {
	t.Helper()
	if usage.State != llmprotocol.UsageAvailable {
		t.Fatalf("usage state = %q, want the trailing chunk's counts to be available", usage.State)
	}
	for _, count := range []struct {
		name string
		got  llmprotocol.TokenCount
		want int64
	}{
		{"input total", usage.InputTotal, prompt},
		{"output total", usage.OutputTotal, completion},
		{"total", usage.Total, total},
	} {
		if count.got.Value == nil {
			t.Fatalf("%s is absent, want %d from the trailing chunk", count.name, count.want)
		}
		if *count.got.Value != count.want {
			t.Fatalf("%s = %d, want %d from the trailing chunk", count.name, *count.got.Value, count.want)
		}
	}
}

// A delta that carries anything after its choice has finished is still a
// lifecycle error. Only an empty one is a no-op.
func TestNonEmptyDeltaAfterFinishStaysAnError(t *testing.T) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	head := `{"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model","choices":[{"index":0,`
	if _, _, _, pushErr := stream.Push([]byte(
		"data: " + head + `"delta":{"content":"hi","role":"assistant"},"finish_reason":null}]}` + "\n\n" +
			"data: " + head + `"delta":{"content":"","role":"assistant"},"finish_reason":"stop"}]}` + "\n\n",
	)); pushErr != nil {
		t.Fatalf("push: %v", pushErr)
	}
	_, _, _, pushErr := stream.Push([]byte(
		"data: " + head + `"delta":{"content":"more","role":"assistant"},"finish_reason":"stop"}]}` + "\n\n"))
	if pushErr == nil {
		t.Fatal("a non-empty delta after finish was accepted")
	}
	if !strings.Contains(pushErr.Error(), "invalid_item_lifecycle") {
		t.Fatalf("error = %v, want invalid_item_lifecycle", pushErr)
	}
}

// The dropped members are named, so an operator can see which provider field
// the Router stopped carrying without reading any response content.
func TestOpenRouterUnknownFieldsAreNamed(t *testing.T) {
	_, dropped := pruneUnknownProviderFields(
		loadProviderFixture(t, openRouterResponseReasoning), reflect.TypeOf(&chatResponseWire{}),
	)
	want := []string{
		"choices[].message.reasoning_details",
		"choices[].native_finish_reason",
		"provider",
		"usage.completion_tokens_details.image_tokens",
		"usage.cost",
		"usage.cost_details",
		"usage.is_byok",
		"usage.prompt_tokens_details.video_tokens",
	}
	if !reflect.DeepEqual(dropped, want) {
		t.Fatalf("dropped fields are %v, want %v", dropped, want)
	}
}

func TestOpenRouterStreamUnknownFieldsAreNamed(t *testing.T) {
	seen := map[string]bool{}
	for _, line := range strings.Split(string(loadProviderFixture(t, openRouterStream)), "\n") {
		payload, found := strings.CutPrefix(strings.TrimSpace(line), "data: ")
		if !found || payload == "[DONE]" {
			continue
		}
		_, dropped := pruneUnknownProviderFields([]byte(payload), reflect.TypeOf(&chatChunkWire{}))
		for _, field := range dropped {
			seen[field] = true
		}
	}
	for _, field := range []string{
		"choices[].delta.reasoning_details", "choices[].native_finish_reason", "provider",
		"usage.completion_tokens_details.image_tokens", "usage.cost", "usage.cost_details",
		"usage.is_byok", "usage.prompt_tokens_details.video_tokens",
	} {
		if !seen[field] {
			t.Fatalf("stream did not report %q as dropped; reported %v", field, seen)
		}
	}
}

// Tolerance belongs to the upstream leg alone. A client request that names a
// member the contract does not is still refused.
func TestClientRequestStillRejectsUnknownFields(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"client-model","messages":[{"role":"user","content":"hello"}],"future_field":true}`)
	_, err := engine.TranslateRequest(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body,
		func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		},
	)
	if err == nil {
		t.Fatal("an unknown client request field was accepted")
	}
}

// Tolerance covers members the contract does not name. A member it does name
// still has to hold the shape the contract declares, so a provider cannot
// change the meaning of accounting or content and have it pass.
func TestNamedProviderFieldsKeepTheirShape(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":"eleven","completion_tokens":8,"total_tokens":19}}`)
	_, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, renameModel("public-model"),
	)
	if err == nil {
		t.Fatal("a named provider field with the wrong type was accepted")
	}
}

// A refusal that still stands -- the request leg, the transport-error leg --
// must say which member caused it. Without the cause the log reads only as
// "some field was non-canonical", which does not locate the field.
func TestRefusalNamesTheOffendingField(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"client-model","messages":[{"role":"user","content":"hello"}],"future_field":true}`)
	_, err := engine.TranslateRequest(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body,
		func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			return nil
		},
	)
	if err == nil {
		t.Fatal("an unknown client request field was accepted")
	}
	if !strings.Contains(err.Error(), `"future_field"`) {
		t.Fatalf("error %q does not name the offending field", err)
	}
}
