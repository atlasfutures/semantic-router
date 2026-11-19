package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// responsesCallStream is a Responses provider stream of one get_time call
// whose added item holds added as its arguments, followed by the given
// argument deltas and a done event carrying done.
func responsesCallStream(added string, deltas []string, done string) string {
	quote := func(text string) string {
		encoded, _ := json.Marshal(text)
		return string(encoded)
	}
	frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	frames := []string{
		frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":100,"model":"local-model","status":"in_progress","output":[]}}`),
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":`+quote(added)+`,"call_id":"call_1","name":"get_time"}}`),
	}
	sequence := 2
	for _, delta := range deltas {
		frames = append(frames, frame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","sequence_number":`+strconv.Itoa(sequence)+`,"item_id":"fc_1","output_index":0,"delta":`+quote(delta)+`}`))
		sequence++
	}
	item := `{"id":"fc_1","type":"function_call","status":"completed","arguments":` + quote(done) + `,"call_id":"call_1","name":"get_time"}`
	frames = append(frames,
		frame("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","sequence_number":`+strconv.Itoa(sequence)+`,"item_id":"fc_1","output_index":0,"arguments":`+quote(done)+`}`),
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":`+strconv.Itoa(sequence+1)+`,"output_index":0,"item":`+item+`}`),
		frame("response.completed", `{"type":"response.completed","sequence_number":`+strconv.Itoa(sequence+2)+`,"response":{"id":"resp_1","object":"response","created_at":100,"model":"local-model","status":"completed","output":[`+item+`],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`),
	)
	return strings.Join(frames, "")
}

// assembledToolArguments reads a client stream the way a client that
// accumulates a call while it streams does, and returns each call's
// arguments by the order the calls started. Chat: the arguments of every
// tool_calls entry of an index, concatenated. Responses (the OpenAI SDK's
// stream helper): the added item's arguments plus every argument delta.
// Messages (the Anthropic SDK): the input_json_delta text joined, or the
// tool_use block's start input when no delta came.
func assembledToolArguments(t *testing.T, client llmprotocol.WireFormat, written []byte) []string {
	t.Helper()
	var calls []string
	chatCalls := map[int]int{}
	messagesDeltas := map[int]bool{}
	messagesCalls := map[int]int{}
	for _, line := range bytes.Split(written, []byte("\n")) {
		data, ok := bytes.CutPrefix(line, []byte("data: "))
		if !ok || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var frame struct {
			Type  string          `json:"type"`
			Index int             `json:"index"`
			Delta json.RawMessage `json:"delta"`
			Item  struct {
				Type      string `json:"type"`
				Arguments string `json:"arguments"`
			} `json:"item"`
			ContentBlock struct {
				Type  string          `json:"type"`
				Input json.RawMessage `json:"input"`
			} `json:"content_block"`
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int `json:"index"`
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(data, &frame); err != nil {
			t.Fatalf("frame is not JSON: %v\n%s", err, data)
		}
		switch client {
		case llmprotocol.OpenAIChatV1:
			for _, choice := range frame.Choices {
				for _, call := range choice.Delta.ToolCalls {
					position, seen := chatCalls[call.Index]
					if !seen {
						position = len(calls)
						chatCalls[call.Index] = position
						calls = append(calls, "")
					}
					calls[position] += call.Function.Arguments
				}
			}
		case llmprotocol.OpenAIResponsesV1:
			switch frame.Type {
			case "response.output_item.added":
				if frame.Item.Type == "function_call" {
					calls = append(calls, frame.Item.Arguments)
				}
			case "response.function_call_arguments.delta":
				var delta string
				if err := json.Unmarshal(frame.Delta, &delta); err != nil {
					t.Fatal(err)
				}
				calls[len(calls)-1] += delta
			}
		case llmprotocol.AnthropicMessagesV1:
			switch frame.Type {
			case "content_block_start":
				if frame.ContentBlock.Type == "tool_use" {
					messagesCalls[frame.Index] = len(calls)
					calls = append(calls, string(frame.ContentBlock.Input))
				}
			case "content_block_delta":
				var delta struct {
					Type        string `json:"type"`
					PartialJSON string `json:"partial_json"`
				}
				if err := json.Unmarshal(frame.Delta, &delta); err != nil {
					t.Fatal(err)
				}
				if delta.Type != "input_json_delta" {
					continue
				}
				position := messagesCalls[frame.Index]
				if !messagesDeltas[frame.Index] {
					messagesDeltas[frame.Index] = true
					calls[position] = ""
				}
				calls[position] += delta.PartialJSON
			}
		}
	}
	return calls
}

func requireSameJSONObject(t *testing.T, got, want string) {
	t.Helper()
	var gotValue, wantValue map[string]any
	if err := json.Unmarshal([]byte(got), &gotValue); err != nil {
		t.Fatalf("assembled arguments %q are not a JSON object: %v", got, err)
	}
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("assembled arguments = %s, want %s", got, want)
	}
}

// A call's added item may hold the "{}" placeholder, or nothing, and its
// arguments may come as deltas or only in the done event. Every client
// that accumulates the call as it streams ends with exactly one call whose
// arguments are the done event's, never the placeholder twice or the
// placeholder followed by the arguments.
func TestToolCallArgumentsAssembleOnceForEveryAddedShape(t *testing.T) {
	const cityArguments = `{"city":"Paris"}`
	cases := map[string]struct {
		added  string
		deltas []string
		done   string
	}{
		"placeholder, zero arguments in done only":    {added: "{}", done: "{}"},
		"placeholder, arguments in done only":         {added: "{}", done: cityArguments},
		"empty, zero arguments in done only":          {added: "", done: "{}"},
		"empty, arguments in done only":               {added: "", done: cityArguments},
		"placeholder, zero arguments as a delta":      {added: "{}", deltas: []string{"{}"}, done: "{}"},
		"placeholder, arguments as deltas":            {added: "{}", deltas: []string{`{"city":`, `"Paris"}`}, done: cityArguments},
		"empty, arguments as deltas":                  {added: "", deltas: []string{`{"city":`, `"Paris"}`}, done: cityArguments},
		"arguments held by the added item, no deltas": {added: cityArguments, done: cityArguments},
	}
	clients := []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1}
	for name, test := range cases {
		for _, client := range clients {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				written, response, err := streamResponsesTo(t, client, responsesCallStream(test.added, test.deltas, test.done))
				if err != nil {
					t.Fatalf("stream refused: %v", err)
				}
				if client == llmprotocol.OpenAIChatV1 && test.added == cityArguments {
					// A Chat client is sent calls from argument deltas only, and
					// arguments held by the added item are not resent as one; no
					// server is known to stream a call this way.
					return
				}
				calls := assembledToolArguments(t, client, written)
				if len(calls) != 1 {
					t.Fatalf("client assembled %d calls %q, want one\n%s", len(calls), calls, written)
				}
				requireSameJSONObject(t, calls[0], test.done)
				if len(response.Output) != 1 || len(response.Output[0].Content) != 1 || response.Output[0].Content[0].ToolCall == nil {
					t.Fatalf("accumulated output = %+v, want one call", response.Output)
				}
				requireSameJSONObject(t, response.Output[0].Content[0].ToolCall.Arguments, test.done)
			})
		}
	}
}

// Arguments that genuinely conflict are still refused: streamed deltas, or
// real arguments in the added item, that the done event contradicts.
func TestConflictingToolArgumentsAreStillRefused(t *testing.T) {
	cases := map[string]string{
		"deltas then other done arguments":     responsesCallStream("", []string{`{"city":"Paris"}`}, `{"city":"Rome"}`),
		"added arguments then other done ones": responsesCallStream(`{"city":"Paris"}`, nil, `{"city":"Rome"}`),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, err := streamResponsesTo(t, llmprotocol.OpenAIChatV1, body)
			if err == nil || !strings.Contains(err.Error(), "stream_tool_arguments_mismatch") {
				t.Fatalf("err = %v, want stream_tool_arguments_mismatch", err)
			}
		})
	}
}

// A zero-argument Anthropic tool_use, as Claude streams it (a tool_use
// block whose input is {} and one input_json_delta of "{}"), translated to
// a Responses client and then from that Responses stream to a Chat client,
// reaches the Chat client as one call with arguments {}.
func TestAnthropicZeroArgumentCallRoundTripsThroughResponsesToChat(t *testing.T) {
	frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	anthropic := strings.Join([]string{
		frame("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`),
		frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_time","input":{}}}`),
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`),
		frame("content_block_stop", `{"type":"content_block_stop","index":0}`),
		frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`),
		frame("message_stop", `{"type":"message_stop"}`),
	}, "")
	toResponses, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "claude-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, pushErr := toResponses.Push([]byte(anthropic))
	final, _, _, finalErr := toResponses.Finalize(pushErr)
	if pushErr != nil || finalErr != nil {
		t.Fatalf("Anthropic to Responses refused: %v, %v", pushErr, finalErr)
	}
	responses := bytes.Join(append(frames, final...), nil)
	written, _, err := streamResponsesTo(t, llmprotocol.OpenAIChatV1, string(responses))
	if err != nil {
		t.Fatalf("Responses to Chat refused: %v\n%s", err, responses)
	}
	calls := assembledToolArguments(t, llmprotocol.OpenAIChatV1, written)
	if len(calls) != 1 || calls[0] != "{}" {
		t.Fatalf("Chat client assembled %q, want one call with {}\n%s", calls, written)
	}
}
