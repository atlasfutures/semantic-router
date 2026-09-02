package protocolcodec

import (
	"context"
	"errors"
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

// providerStreamFrames splits the captured SSE body into wire frames. The
// terminal frame is returned separately: OpenRouter repeats the stop choice on
// the chunk that carries usage, and a repeated stop is a second defect that
// this change does not address.
func providerStreamFrames(t *testing.T) (frames []string, terminal string) {
	t.Helper()
	for _, frame := range strings.SplitAfter(string(loadProviderFixture(t, openRouterStream)), "\n\n") {
		if strings.TrimSpace(frame) == "" {
			continue
		}
		frames = append(frames, frame)
	}
	if len(frames) < 2 {
		t.Fatalf("stream fixture has %d frames, want the content frames and a terminal usage frame", len(frames))
	}
	return frames[:len(frames)-2], frames[len(frames)-2]
}

func newProviderStream(t *testing.T, target llmprotocol.WireFormat) *StreamEngine {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "deepseek/deepseek-v4-pro",
	})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

// Every content frame of a real OpenRouter stream must decode. Each one names
// provider, native_finish_reason and, while the model is thinking,
// reasoning_details.
func pushProviderStreamContent(t *testing.T, target llmprotocol.WireFormat) {
	t.Helper()
	stream := newProviderStream(t, target)
	frames, _ := providerStreamFrames(t)
	for _, frame := range frames {
		if _, _, _, err := stream.Push([]byte(frame)); err != nil {
			t.Fatalf("push frame %.160q: %v", frame, err)
		}
	}
}

func TestOpenRouterStreamTranslatesToAnthropic(t *testing.T) {
	pushProviderStreamContent(t, llmprotocol.AnthropicMessagesV1)
}

func TestOpenRouterStreamPassesThroughChat(t *testing.T) {
	pushProviderStreamContent(t, llmprotocol.OpenAIChatV1)
}

// The terminal usage frame carries the four accounting members OpenRouter adds
// -- cost, is_byok, cost_details and the two token-detail additions -- and it
// must no longer fail on any of them. It still fails, on the repeated stop
// choice that precedes the usage in the same chunk. That is the next defect on
// this path and it is deliberately not fixed here; this test pins which of the
// two failures the frame produces, so the fix for the other one flips it.
func TestOpenRouterTerminalUsageFrameNoLongerFailsOnFieldNames(t *testing.T) {
	stream := newProviderStream(t, llmprotocol.OpenAIChatV1)
	frames, terminal := providerStreamFrames(t)
	for _, frame := range frames {
		if _, _, _, err := stream.Push([]byte(frame)); err != nil {
			t.Fatalf("push frame %.160q: %v", frame, err)
		}
	}
	_, _, _, err := stream.Push([]byte(terminal))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("terminal frame error is %v, want a protocol error", err)
	}
	if protocolError.Code == "invalid_upstream_json" {
		t.Fatal("the terminal usage frame still fails on a provider field name")
	}
	if protocolError.Code != "invalid_item_lifecycle" {
		t.Fatalf("terminal frame code is %q, want the repeated-stop lifecycle failure", protocolError.Code)
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
