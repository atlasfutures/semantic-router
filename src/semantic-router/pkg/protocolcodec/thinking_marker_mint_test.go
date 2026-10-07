package protocolcodec

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Phase 2 of #191: thinking no provider signed reaches a Messages client
// signed with a Router marker, on every path, and the marker is the one the
// request pre-pass recognises.

func TestThinkingMarkerIsDeterministicAndRecognised(t *testing.T) {
	for model, want := range map[string]string{
		"moonshotai/kimi-k3":               "moonshotai",
		"z-ai/glm-5.3-flash":               "z-ai",
		"kimi-k3":                          "kimi-k3",
		"xiaomi/mimo-v2.5-pro@thinking-on": "xiaomi",
		"GLM_5.3":                          "glm-5-3",
		"":                                 "unknown",
		strings.Repeat("a", 80) + "/x":     strings.Repeat("a", 64),
	} {
		if got := ThinkingMarkerFamily(model); got != want {
			t.Errorf("ThinkingMarkerFamily(%q) = %q, want %q", model, got, want)
		}
	}
	marker := ThinkingMarker("moonshotai", "kimi reasoning")
	if again := ThinkingMarker("moonshotai", "kimi reasoning"); again != marker {
		t.Fatalf("the marker is not deterministic: %q then %q", marker, again)
	}
	if other := ThinkingMarker("moonshotai", "other reasoning"); other == marker {
		t.Fatal("two texts share a marker")
	}
	if family, reserved := routerSignature(marker); !reserved || family != "moonshotai" {
		t.Fatalf("routerSignature(%q) = %q, %v; the request pre-pass would not recognise it", marker, family, reserved)
	}
}

func mintedThinkingSignature(t *testing.T, body []byte) (string, bool) {
	t.Helper()
	var wire struct {
		Content []struct {
			Type      string  `json:"type"`
			Signature *string `json:"signature"`
		} `json:"content"`
	}
	if err := json.Unmarshal(body, &wire); err != nil || len(wire.Content) == 0 || wire.Content[0].Type != "thinking" {
		t.Fatalf("body = %s", body)
	}
	if wire.Content[0].Signature == nil {
		return "", false
	}
	return *wire.Content[0].Signature, true
}

// Buffered: a non-Anthropic worker's thinking, translated for a Messages
// client, is signed with the marker when the Router asks and left unsigned
// when it does not. Claude's carried signature always wins.
func TestBufferedMessagesOutputMintsAMarkerForUnsignedThinking(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	translate := func(t *testing.T, details, family string) []byte {
		t.Helper()
		result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
			[]byte(chatResponseWithReasoning(`"Check the tool."`, details)),
			func(response *llmprotocol.Response) error {
				response.ThinkingMarkerFamily = family
				return nil
			})
		if err != nil {
			t.Fatal(err)
		}
		return result.Body
	}
	unsigned := `[{"type":"reasoning.text","text":"Check the tool.","format":"unknown","index":0}]`

	got, signed := mintedThinkingSignature(t, translate(t, unsigned, "moonshotai"))
	if !signed || got != ThinkingMarker("moonshotai", "Check the tool.") {
		t.Fatalf("signature = %q (present %v), want the marker", got, signed)
	}
	if got, _ := mintedThinkingSignature(t, translate(t, unsigned, "")); got != "" {
		t.Fatalf("a marker was minted with minting off: %q", got)
	}
	if got, _ := mintedThinkingSignature(t, translate(t, "["+claudeDetail+"]", "moonshotai")); got != "sig-claude" {
		t.Fatalf("Claude's carried signature was replaced: %q", got)
	}
}

func streamToMessages(t *testing.T, family string, chunks []string) string {
	t.Helper()
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public", ProviderModel: "m", ThinkingMarkerFamily: family,
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire strings.Builder
	for _, chunk := range chunks {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		for _, frame := range frames {
			wire.Write(frame)
		}
	}
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, frame := range frames {
		wire.Write(frame)
	}
	return wire.String()
}

// Streamed: the marker travels as a signature_delta after the thinking and
// before its block stops, and it is the marker the buffered path mints for
// the same text, so a replay of either is byte-identical.
func TestStreamedMessagesOutputMintsTheSameMarker(t *testing.T) {
	chunk := func(delta string) string {
		return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}` + "\n\n"
	}
	done := `data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"
	chunks := []string{
		chunk(`{"role":"assistant","content":"","reasoning":"Check "}`),
		chunk(`{"content":"","reasoning":"the tool."}`),
		chunk(`{"content":"Done."}`),
		done,
	}
	out := streamToMessages(t, "moonshotai", chunks)
	marker := ThinkingMarker("moonshotai", "Check the tool.")
	signedAt := strings.Index(out, `"signature":"`+marker+`"`)
	stopAt := strings.Index(out, `"content_block_stop"`)
	if signedAt < 0 || stopAt < signedAt || strings.Index(out, `"thinking":"the tool."`) > signedAt {
		t.Fatalf("no marker signature_delta between the thinking and its block stop: %s", out)
	}
	if strings.Count(out, `"signature_delta"`) != 1 {
		t.Fatalf("the block was signed more than once: %s", out)
	}
	if off := streamToMessages(t, "", chunks); strings.Contains(off, `"signature_delta"`) {
		t.Fatalf("a marker was minted with minting off: %s", off)
	}
}

const unsignedThinkingMessagesStream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Check \"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"the tool.\"}}\n\n"

const messagesStreamTail = "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Done.\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// Same-format Messages: the stream still travels byte for byte, with one
// signature_delta added before the stop of a thinking block the upstream left
// unsigned. A block the upstream signed gets nothing.
func TestMessagesPassthroughMintsAMarkerOnlyForUnsignedThinking(t *testing.T) {
	upstream := unsignedThinkingMessagesStream + messagesStreamTail
	out := string(pushInSmallChunks(t, NewAnthropicPublicStreamFilter(1<<20, "moonshotai"), []byte(upstream)))
	signature := "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"" +
		ThinkingMarker("moonshotai", "Check the tool.") + "\"}}\n\n"
	if want := unsignedThinkingMessagesStream + signature + messagesStreamTail; out != want {
		t.Fatalf("passthrough =\n%s\nwant\n%s", out, want)
	}

	signed := unsignedThinkingMessagesStream +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"provider-sig\"}}\n\n" +
		messagesStreamTail
	if out := string(pushInSmallChunks(t, NewAnthropicPublicStreamFilter(1<<20, "moonshotai"), []byte(signed))); out != signed {
		t.Fatalf("a signed block was changed:\n%s", out)
	}
	if out := string(pushInSmallChunks(t, NewAnthropicPublicStreamFilter(1<<20, ""), []byte(upstream))); out != upstream {
		t.Fatalf("a marker was minted with minting off:\n%s", out)
	}
}

// The round trip: a minted marker a Messages client resends is stripped by
// the request pre-pass, so the block is unsigned reasoning again and never
// reaches a provider signed with the marker.
func TestAMintedMarkerIsStrippedWhenTheClientResendsIt(t *testing.T) {
	marker := ThinkingMarker("moonshotai", "kimi reasoning")
	body := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":[{"type":"thinking","thinking":"kimi reasoning","signature":"` + marker + `"},{"type":"text","text":"ok"}]},` +
		`{"role":"user","content":"go on"}]}`
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, []byte(body), nil)
		if err != nil {
			t.Fatalf("%s: %v", target, err)
		}
		if strings.Contains(string(result.Body), "vsr.") {
			t.Fatalf("%s: a minted marker reached the provider body: %s", target, result.Body)
		}
	}
}

// A same-format Messages body is passed through as the upstream wrote it
// unless something must change. Unsigned thinking a marker will sign is such
// a change; Claude's signed thinking and minting off are not.
func TestResponseMintsThinkingMarkersDecidesTheReencode(t *testing.T) {
	thinking := func(signature string) llmprotocol.Response {
		return llmprotocol.Response{Output: []llmprotocol.OutputItem{{Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentReasoning, Text: "kimi reasoning", Signature: signature},
			{Kind: llmprotocol.ContentText, Text: "ok"},
		}}}}
	}
	unsigned := thinking("")
	unsigned.ThinkingMarkerFamily = "moonshotai"
	if !ResponseMintsThinkingMarkers(unsigned) {
		t.Fatal("unsigned thinking with minting on is not re-encoded")
	}
	signed := thinking("provider-sig")
	signed.ThinkingMarkerFamily = "moonshotai"
	if ResponseMintsThinkingMarkers(signed) {
		t.Fatal("a signed block forces a re-encode")
	}
	if ResponseMintsThinkingMarkers(thinking("")) {
		t.Fatal("minting off forces a re-encode")
	}
}
