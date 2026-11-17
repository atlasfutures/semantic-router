package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// OpenRouter streams Claude's thinking over Responses as a reasoning item
// whose reasoning_text part is announced with content_part.added and closed
// with content_part.done, the same events a message uses for output_text:
//
//	output_item.added    {item: {type: reasoning}}
//	content_part.added   {part: {type: reasoning_text, text: ""}}
//	reasoning_text.delta / reasoning_text.done
//	content_part.done    {part: {type: reasoning_text, text}}
//	output_item.done     {item: {type: reasoning, content: [reasoning_text], signature, format}}
//
// The decoder took every content_part event for a message part and aborted
// the stream with stream_item_kind_mismatch, so every Claude turn that
// thought before calling a tool failed. The fixture is a captured stream of
// that shape with its ids, text, signature and tools replaced.
const openRouterClaudeReasoningStream = "responses-stream-claude-reasoning.sse"

const openRouterClaudeReasoningText = "The user wants 6*7 evaluated with the js tool, so I will call it."

func pushResponsesStream(body []byte, client llmprotocol.WireFormat) ([]byte, error) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, client, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "anthropic/claude-sonnet-5.5",
		ProviderModel: "anthropic/claude-sonnet-5.5", WorkerFamily: "anthropic",
	})
	if err != nil {
		return nil, err
	}
	frames, _, _, err := stream.Push(body)
	if err != nil {
		return nil, err
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		return nil, err
	}
	return bytes.Join(append(frames, final...), nil), nil
}

func TestOpenRouterClaudeReasoningStreamReachesEveryClient(t *testing.T) {
	body := loadProviderFixture(t, openRouterClaudeReasoningStream)
	for _, client := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(client), func(t *testing.T) {
			routed, err := pushResponsesStream(body, client)
			if err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			for _, want := range []string{"6*7 evaluated ", `\"code\":`} {
				if !bytes.Contains(routed, []byte(want)) {
					t.Fatalf("routed stream lost %q:\n%s", want, routed)
				}
			}
		})
	}
}

var responsesSequenceNumber = regexp.MustCompile(`"sequence_number":(\d+)`)

// dropResponsesStreamEvent removes the event with sequence number `sequence`
// from an SSE body and renumbers the events after it, since the decoder
// requires contiguous sequence numbers.
func dropResponsesStreamEvent(t *testing.T, body string, sequence int) string {
	t.Helper()
	events := strings.SplitAfter(body, "\n\n")
	kept := events[:0]
	for _, event := range events {
		match := responsesSequenceNumber.FindStringSubmatch(event)
		if match != nil && match[1] == strconv.Itoa(sequence) && strings.HasPrefix(event, "data: ") {
			continue
		}
		kept = append(kept, event)
	}
	if len(kept) != len(events)-1 {
		t.Fatalf("the fixture has no single event with sequence number %d", sequence)
	}
	return responsesSequenceNumber.ReplaceAllStringFunc(strings.Join(kept, ""), func(field string) string {
		number, _ := strconv.Atoi(responsesSequenceNumber.FindStringSubmatch(field)[1])
		if number > sequence {
			number--
		}
		return `"sequence_number":` + strconv.Itoa(number)
	})
}

// A reasoning_text part may open with text of its own, which is the first
// reasoning delta and reaches every client like the deltas after it.
func TestOpenRouterClaudeReasoningInitialPartTextReachesEveryClient(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	if !strings.Contains(body, `"delta":"The user wants ","sequence_number":4}`) {
		t.Fatal("the fixture's first reasoning delta is no longer event 4")
	}
	mutated := strings.Replace(body,
		`"part":{"type":"reasoning_text","text":""}`, `"part":{"type":"reasoning_text","text":"The user wants "}`, 1)
	if mutated == body {
		t.Fatal("the fixture no longer opens an empty reasoning_text part")
	}
	mutated = dropResponsesStreamEvent(t, mutated, 4)
	for _, client := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(client), func(t *testing.T) {
			routed, err := pushResponsesStream([]byte(mutated), client)
			if err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			if !bytes.Contains(routed, []byte(`"The user wants "`)) {
				t.Fatalf("routed stream lost the part's initial text:\n%s", routed)
			}
		})
	}
}

// A reasoning_text part that content_part.added opened must be closed by its
// content_part.done, and a content_part.done must close a part that
// content_part.added opened, not reasoning text that started with a delta.
func TestResponsesReasoningTextPartMustCloseAsItOpened(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	for _, check := range []struct {
		name     string
		sequence int
		event    string
		code     string
	}{
		{"part opened and never closed", 9, "response.content_part.done", "stream_item_content_incomplete"},
		{"part closed and never opened", 3, "response.content_part.added", "stream_content_part_start_missing"},
	} {
		t.Run(check.name, func(t *testing.T) {
			event := regexp.MustCompile(`(?m)^data: \{"type":"` + regexp.QuoteMeta(check.event) +
				`",.*"sequence_number":` + strconv.Itoa(check.sequence) + `\}$`)
			if !event.MatchString(body) {
				t.Fatalf("the fixture's event %d is no longer %s", check.sequence, check.event)
			}
			_, err := pushResponsesStream([]byte(dropResponsesStreamEvent(t, body, check.sequence)), llmprotocol.OpenAIResponsesV1)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != check.code {
				t.Fatalf("got %v, want %s", err, check.code)
			}
		})
	}
}

// A reasoning_text part is reasoning content: it belongs to a reasoning item
// and to no other, and a message part still belongs only to a message.
func TestResponsesContentPartMustMatchItsItem(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	for name, mutated := range map[string]string{
		"reasoning_text part on a message": strings.Replace(body,
			`"item":{"id":"rs_fixture_1","type":"reasoning","status":"in_progress","summary":[]}`,
			`"item":{"id":"rs_fixture_1","type":"message","role":"assistant","status":"in_progress","content":[]}`, 1),
		"output_text part on a reasoning item": strings.Replace(body,
			`"part":{"type":"reasoning_text","text":""}`,
			`"part":{"type":"output_text","text":"","annotations":[]}`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if mutated == body {
				t.Fatal("the fixture no longer has the event this case rewrites")
			}
			_, err := pushResponsesStream([]byte(mutated), llmprotocol.OpenAIResponsesV1)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != "stream_item_kind_mismatch" {
				t.Fatalf("got %v, want stream_item_kind_mismatch", err)
			}
		})
	}
}
