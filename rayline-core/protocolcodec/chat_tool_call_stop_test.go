package protocolcodec

import (
	"context"
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// Some OpenAI-compatible servers end a Chat reply that holds a whole tool call
// with finish_reason "stop" (or, buffered, with none). An agent client reads
// the stop reason to decide whether to run the call, so a plain end of turn
// beside one stalls its loop. These tests pin that every client is told the
// turn ended on a tool call, and that nothing else changes.

// chatToolCallReply is a buffered Chat reply holding one whole tool call,
// ended by finish (no finish_reason member when finish is empty).
func chatToolCallReply(finish string) []byte {
	finishMember := ""
	if finish != "" {
		finishMember = `,"finish_reason":` + strconv.Quote(finish)
	}
	return []byte(`{"id":"gen-fixture","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
		`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_fixture","type":"function",` +
		`"function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}` + finishMember + `}],` +
		`"usage":{"prompt_tokens":92,"completion_tokens":12,"total_tokens":104}}`)
}

const chatTextReplyUnderStop = `{"id":"gen-fixture","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
	`"message":{"role":"assistant","content":"Done."},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":92,"completion_tokens":2,"total_tokens":94}}`

func TestABufferedChatToolCallUnderAStopFinishIsAToolCallStop(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, run := range map[string]struct {
		body      []byte
		stop      llmprotocol.StopReason
		source    string
		misstated bool
		chat      string
		messages  string
	}{
		"tool call, finish_reason stop":       {chatToolCallReply("stop"), llmprotocol.StopToolCall, "stop", true, `"finish_reason":"tool_calls"`, `"stop_reason":"tool_use"`},
		"tool call, no finish_reason":         {chatToolCallReply(""), llmprotocol.StopToolCall, "", false, `"tool_calls":[`, `"stop_reason":"tool_use"`},
		"tool call, finish_reason tool_calls": {chatToolCallReply("tool_calls"), llmprotocol.StopToolCall, "tool_calls", false, `"finish_reason":"tool_calls"`, `"stop_reason":"tool_use"`},
		"cut tool call, native length": {
			chatBufferedToolCall(chatBufferedCutArguments, "tool_calls", "length"),
			llmprotocol.StopMaxTokens, "tool_calls", true, `"finish_reason":"length"`, `"stop_reason":"max_tokens"`,
		},
		"text, finish_reason stop": {[]byte(chatTextReplyUnderStop), llmprotocol.StopEndTurn, "stop", false, `"finish_reason":"stop"`, `"stop_reason":"end_turn"`},
	} {
		t.Run(name, func(t *testing.T) {
			response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, run.body)
			if err != nil {
				t.Fatal(err)
			}
			if response.StopReason != run.stop || response.SourceStopReason != run.source {
				t.Fatalf("stop = %q, source stop = %q; want %q, %q", response.StopReason, response.SourceStopReason, run.stop, run.source)
			}
			// A same-format client is handed the upstream body unless it
			// misstates the stop; then it gets the re-encoded one.
			if misstated := UpstreamBodyMisstatesStop(llmprotocol.OpenAIChatV1, response); misstated != run.misstated {
				t.Fatalf("UpstreamBodyMisstatesStop = %v, want %v", misstated, run.misstated)
			}
			for target, want := range map[llmprotocol.WireFormat]string{
				llmprotocol.OpenAIChatV1:        run.chat,
				llmprotocol.AnthropicMessagesV1: run.messages,
			} {
				translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, run.body, nil)
				if err != nil {
					t.Fatal(err)
				}
				if !strings.Contains(string(translated.Body), want) {
					t.Fatalf("%s body = %s, want %s", target, translated.Body, want)
				}
			}
		})
	}
}

// chatToolCallStreamUnder is a Chat stream holding one whole tool call, ended
// by finish and, when not empty, OpenRouter's native reason.
func chatToolCallStreamUnder(finish, native string) string {
	nativeMember := ""
	if native != "" {
		nativeMember = `,"native_finish_reason":` + strconv.Quote(native)
	}
	return `data: {"id":"gen-fixture","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
		`"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_fixture","type":"function",` +
		`"function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"gen-fixture","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
		`"delta":{},"finish_reason":` + strconv.Quote(finish) + nativeMember + `}],` +
		`"usage":{"prompt_tokens":92,"completion_tokens":12,"total_tokens":104}}` + "\n\n" +
		"data: [DONE]\n\n"
}

// chatTextStreamUnderStop is a Chat stream holding one text turn ended by
// "stop".
const chatTextStreamUnderStop = `data: {"id":"gen-fixture","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
	`"delta":{"role":"assistant","content":"Done."},"finish_reason":null}]}` + "\n\n" +
	`data: {"id":"gen-fixture","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
	`"delta":{},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":92,"completion_tokens":2,"total_tokens":94}}` + "\n\n" +
	"data: [DONE]\n\n"

func TestAChatStreamToolCallUnderAStopFinishIsAToolCallStop(t *testing.T) {
	cut := strings.Join(loadMaxTokensToolCutChunks(t, "stream/041-chat-max-tokens-mid-tool-in.json"), "")
	for name, run := range map[string]struct {
		stream   string
		stop     llmprotocol.StopReason
		chat     string
		messages string
	}{
		"tool call, finish_reason stop":           {chatToolCallStreamUnder("stop", ""), llmprotocol.StopToolCall, `"finish_reason":"tool_calls"`, `"stop_reason":"tool_use"`},
		"tool call, finish_reason stop, native":   {chatToolCallStreamUnder("stop", "stop"), llmprotocol.StopToolCall, `"finish_reason":"tool_calls"`, `"stop_reason":"tool_use"`},
		"tool call, finish_reason tool_calls":     {chatToolCallStreamUnder("tool_calls", ""), llmprotocol.StopToolCall, `"finish_reason":"tool_calls"`, `"stop_reason":"tool_use"`},
		"cut tool call, native max_output_tokens": {cut, llmprotocol.StopMaxTokens, `"finish_reason":"length"`, `"stop_reason":"max_tokens"`},
		"text, finish_reason stop":                {chatTextStreamUnderStop, llmprotocol.StopEndTurn, `"finish_reason":"stop"`, `"stop_reason":"end_turn"`},
	} {
		t.Run(name, func(t *testing.T) {
			streamContext := llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"}
			response, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.OpenAIChatV1, []byte(run.stream), streamContext)
			if err != nil {
				t.Fatal(err)
			}
			if response.StopReason != run.stop {
				t.Fatalf("stop = %q, want %q", response.StopReason, run.stop)
			}
			for target, want := range map[llmprotocol.WireFormat]string{
				llmprotocol.OpenAIChatV1:        run.chat,
				llmprotocol.AnthropicMessagesV1: run.messages,
			} {
				stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target, streamContext)
				if err != nil {
					t.Fatal(err)
				}
				frames, _, _, err := stream.Push([]byte(run.stream))
				if err != nil {
					t.Fatal(err)
				}
				final, _, _, err := stream.Finalize(nil)
				if err != nil {
					t.Fatal(err)
				}
				var body strings.Builder
				for _, frame := range append(frames, final...) {
					body.Write(frame)
				}
				if !strings.Contains(body.String(), want) {
					t.Fatalf("%s stream = %s, want %s", target, body.String(), want)
				}
			}
		})
	}
}

// A same-format Chat stream reaches its client through the public filter, as
// the upstream wrote it: the filter restates "stop" as "tool_calls" once the
// stream has carried a tool call, whatever the read boundaries, and leaves
// every other stream byte for byte.
func TestChatStreamFiltersRestateAStopFinishAfterAToolCall(t *testing.T) {
	cut := strings.Join(loadMaxTokensToolCutChunks(t, "stream/041-chat-max-tokens-mid-tool-in.json"), "")
	for name, run := range map[string]struct {
		stream string
		want   string
	}{
		"tool call, finish_reason stop":         {chatToolCallStreamUnder("stop", ""), strings.Replace(chatToolCallStreamUnder("stop", ""), `"finish_reason":"stop"`, `"finish_reason":"tool_calls"`, 1)},
		"tool call, finish_reason stop, native": {chatToolCallStreamUnder("stop", "stop"), strings.Replace(chatToolCallStreamUnder("stop", "stop"), `"finish_reason":"stop"`, `"finish_reason":"tool_calls"`, 1)},
		"tool call, finish_reason tool_calls":   {chatToolCallStreamUnder("tool_calls", ""), chatToolCallStreamUnder("tool_calls", "")},
		"text, finish_reason stop":              {chatTextStreamUnderStop, chatTextStreamUnderStop},
	} {
		for _, readSize := range []int{len(run.stream), 7, 1} {
			filter := NewChatPassthroughStreamFilter(1 << 20)
			var output []byte
			for start := 0; start < len(run.stream); start += readSize {
				pushed, err := filter.Push([]byte(run.stream[start:min(start+readSize, len(run.stream))]))
				if err != nil {
					t.Fatalf("%s, reads of %d: %v", name, readSize, err)
				}
				output = append(output, pushed...)
			}
			final, err := filter.Finalize()
			if err != nil {
				t.Fatalf("%s, reads of %d: %v", name, readSize, err)
			}
			output = append(output, final...)
			if got, want := chatStreamJSON(t, string(output)), chatStreamJSON(t, run.want); !reflect.DeepEqual(got, want) {
				t.Fatalf("%s, reads of %d:\n%s\nwant\n%s", name, readSize, output, run.want)
			}
		}
	}
	// The usage filter drops usage the client did not ask for but restates
	// the stop the same way.
	filter := NewChatUsageStreamFilter(1 << 20)
	output, err := filter.Push([]byte(chatToolCallStreamUnder("stop", "")))
	if err != nil {
		t.Fatal(err)
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if body := string(append(output, final...)); !strings.Contains(body, `"finish_reason":"tool_calls"`) || strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("usage filter stream = %s, want finish_reason tool_calls", body)
	}
	// The recorded cut keeps its length restatement.
	filter = NewChatPassthroughStreamFilter(1 << 20)
	output, err = filter.Push([]byte(cut))
	if err != nil {
		t.Fatal(err)
	}
	if body := string(output); !strings.Contains(body, `"finish_reason":"length"`) || strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Fatalf("cut stream = %s, want finish_reason length only", body)
	}
}

// chatStreamJSON is a Chat stream's data frames, each decoded, so a frame the
// filter re-encoded compares by meaning rather than member order.
func chatStreamJSON(t *testing.T, stream string) []any {
	t.Helper()
	var frames []any
	for _, frame := range strings.Split(stream, "\n\n") {
		data, found := strings.CutPrefix(frame, "data: ")
		if !found {
			continue
		}
		if data == "[DONE]" {
			frames = append(frames, data)
			continue
		}
		var decoded any
		if err := json.Unmarshal([]byte(data), &decoded); err != nil {
			t.Fatalf("frame %q: %v", frame, err)
		}
		frames = append(frames, decoded)
	}
	return frames
}
