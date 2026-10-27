package protocolcodec

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"

	logging "github.com/atlasfutures/semantic-router/rayline-core/diagnostics"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// The count a cut turn carries, per target format.
//
// A turn the Router ends never reaches its usage frame, so the count it hands
// the codec is the only record of generation the upstream already billed us
// for. It used to reach the client on the Messages lane alone: the capability
// was optional, the type assertion that tested for it returned silently, and
// every Chat and Responses client received an error frame with no numbers for
// two releases without one log line saying so (memex-desktop#7214).
//
// So each lane's terminal is asserted here as literal wire bytes, and the
// drop that cannot be helped is asserted to be loud.

func truncatedStreamTerminal(
	t *testing.T,
	target llmprotocol.WireFormat,
	usage *llmprotocol.Usage,
	includeUsage bool,
) string {
	t.Helper()
	engine := NewBuiltinEngine()
	options := llmprotocol.StreamOptions{}
	if includeUsage {
		requested := true
		options.IncludeUsage = &requested
	}
	stream, err := engine.NewStream(target, target, llmprotocol.StreamContext{
		Context: t.Context(), Source: target, Target: target, Options: options,
	})
	requireNoError(t, err)

	frames, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(target)))
	requireNoError(t, err)
	if usage != nil {
		stream.SetTruncationUsage(usage, llmprotocol.UsageSourceStreamEstimate)
		// The host that cut the turn says why; here it is its deadline.
		stream.SetTruncationCause(llmprotocol.TruncationDeadline)
	}
	cut := llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated",
		"the router ended this stream at its own deadline, below the platform's", nil,
	)
	terminal, _, _, finalizeErr := stream.Finalize(cut)
	// The cut is the reason the stream is being finalized, so it is not an
	// error the engine reports back. Anything else is: swallowing it here is
	// what let a terminal that failed on its own usage look like a pass.
	if finalizeErr != nil {
		requireErrorIs(t, finalizeErr, cut,
			"finalizing a cut stream failed for a reason other than the cut")
	}
	return string(joinFrames(append(frames, terminal...)))
}

func joinFrames(frames [][]byte) []byte {
	var body []byte
	for _, frame := range frames {
		body = append(body, frame...)
	}
	return body
}

// truncatedUpstreamChunk is one delta in the target's own shape, so the stream
// has forwarded real output before it is cut.
func truncatedUpstreamChunk(target llmprotocol.WireFormat) string {
	switch target {
	case llmprotocol.AnthropicMessagesV1:
		return "event: message_start\n" +
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",` +
			`"model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}` +
			"\n\nevent: content_block_start\n" +
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` +
			"\n\nevent: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ninety-one is"}}` +
			"\n\n"
	case llmprotocol.OpenAIResponsesV1:
		return "event: response.created\n" +
			`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response",` +
			`"created_at":0,"model":"m","status":"in_progress","output":[]}}` + "\n\n" +
			"event: response.output_item.added\n" +
			`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,` +
			`"item":{"type":"message","id":"item_1","role":"assistant","status":"in_progress","content":[]}}` + "\n\n" +
			"event: response.content_part.added\n" +
			`data: {"type":"response.content_part.added","sequence_number":2,"item_id":"item_1",` +
			`"output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}` + "\n\n" +
			"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","sequence_number":3,"item_id":"item_1",` +
			`"output_index":0,"content_index":0,"delta":"ninety-one is"}` + "\n\n"
	default:
		return `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m",` +
			`"choices":[{"index":0,"delta":{"role":"assistant","content":"ninety-one is"},"finish_reason":null}]}` +
			"\n\n"
	}
}

func estimatedTruncationUsage() *llmprotocol.Usage {
	count := func(value int64) llmprotocol.TokenCount {
		return llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageEstimated}
	}
	return &llmprotocol.Usage{
		State:       llmprotocol.UsageAvailable,
		InputTotal:  count(55),
		OutputTotal: count(268),
		Total:       count(323),
	}
}

// Chat ends a cut turn as the format ends a turn stopped at a limit: the chunk
// that names the stop, the usage-only chunk, the sentinel. Those are the three
// frames the format defines and the only three it gets.
func TestChatTruncationWritesTheFormatsTerminal(t *testing.T) {
	body := truncatedStreamTerminal(t, llmprotocol.OpenAIChatV1, estimatedTruncationUsage(), true)

	wantContains(t, body, `"finish_reason":"length"`)
	wantContains(t, body, `"choices":[],"usage":{"prompt_tokens":55,"completion_tokens":268,"total_tokens":323}`)
	wantContains(t, body, "data: [DONE]")
	// The cause rides the chunk that states the stop it qualifies, because
	// "length" alone would say the model exhausted its budget.
	wantContains(t, body, `"finish_reason":"length"}],"rayline":{"truncated":"deadline"}`)
	// The error frame the lane used to end on is gone, and nothing outside the
	// one namespaced member was added.
	wantNotContains(t, body, `"error"`)
	wantNotContains(t, body, "stream_truncated")
	wantNotContains(t, body, "usage_source")

	// The sentinel is last: nothing follows the turn the client has been told
	// is over.
	wantTrue(t, strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]"),
		"the sentinel was not the last thing the client received:\n%s", body)
}

// Responses ends a cut turn as response.incomplete, which states usage and
// takes a reason from a closed set. A value outside that set is one no SDK's
// type contains and this repo's own provider validation rejects.
func TestResponsesTruncationWritesIncomplete(t *testing.T) {
	body := truncatedStreamTerminal(t, llmprotocol.OpenAIResponsesV1, estimatedTruncationUsage(), true)

	wantContains(t, body, `"type":"response.incomplete"`)
	wantContains(t, body, `"status":"incomplete"`)
	wantContains(t, body, `"incomplete_details":{"reason":"max_output_tokens"}`)
	wantContains(t, body, `"input_tokens":55`)
	wantContains(t, body, `"output_tokens":268`)
	wantNotContains(t, body, `"type":"error"`)
	wantNotContains(t, body, "response.failed")

	// The item the cut landed in is closed as incomplete -- the turn ended in
	// the middle of it -- so the terminal states a resource whose lifecycle
	// is whole, says so, and keeps the text the client was streamed.
	wantContains(t, body, `"role":"assistant","status":"incomplete","type":"message"`,
		"the item the cut landed in claimed to be complete")
	wantNotContains(t, body, `"status":"completed","type":"message"`,
		"partial text was presented as a finished item")
	// This terminal follows the completion lifecycle, where an open item is a
	// lifecycle error -- which this codec's own decoder refuses, so a stream it
	// emits and cannot read back is the clearest statement that it is invalid.
	wantContains(t, body, "response.output_item.done",
		"the item the cut landed in was left open")
	wantContains(t, body, "ninety-one is",
		"the terminal resource dropped the text the client was streamed")

	engine := NewBuiltinEngine()
	readback, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)
	_, _, _, err = readback.Push([]byte(body))
	requireNoError(t, err, "this codec emitted a Responses stream it cannot itself read")
}

// Messages keeps the frame it has always written. It is the published
// cross-repo contract the gateway reads (decision 31), and this branch is not
// the place to renegotiate it.
func TestMessagesTruncationKeepsItsPublishedFrame(t *testing.T) {
	body := truncatedStreamTerminal(t, llmprotocol.AnthropicMessagesV1, estimatedTruncationUsage(), false)

	wantContains(t, body, `"type":"error"`)
	wantContains(t, body, `"usage_source":"stream_estimate"`)
	wantContains(t, body, `"output_tokens":268`)
	wantContains(t, body, "content_block_stop")
	// A turn that did not finish still publishes no success terminal here.
	wantNotContains(t, body, "message_stop")
}

// A cut turn with nothing counted cannot close as a length stop: a usage of
// zeros would bill a turn that produced no output. It ends as the failure it
// is, and the client is still given a terminal.
func TestTruncationWithoutACountEndsAsAFailure(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		body := truncatedStreamTerminal(t, target, nil, true)

		wantContains(t, body, "stream_truncated", "%s: the cut did not name itself", target)
		wantNotContains(t, body, `"finish_reason":"length"`, "%s: claimed a length stop", target)
		wantNotContains(t, body, "response.incomplete", "%s: published an incomplete resource", target)
		wantNotContains(t, body, `"usage"`, "%s: billed a turn that generated nothing", target)
	}
}

// A count the stream already stated authoritatively is kept, and the turn
// still ends in the format's own terminal.
//
// These are two separate claims and both were wrong once. Replacing the whole
// usage with the estimate made the stream contradict itself -- restating an
// authoritative count as an estimate is the quality decrease mergeTokenCount
// refuses -- and the turn then ended on a usage-accounting error instead of the
// deadline the client should hear about. Declining to re-scope at all kept the
// provider's count but left the old error-only terminal, so the count never
// reached a Chat client: this format buffers usage until the terminal and the
// error terminal has nowhere to put it. The turn needs both.
func TestTruncationKeepsAStatedCountAndStillTerminates(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
	requireNoError(t, err)
	// The upstream's own usage frame arrives, and the cut follows it.
	counted, _, _, err := stream.Push([]byte(
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m","choices":[],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n"))
	requireNoError(t, err)
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
	requireNoError(t, err, "the cut was reported as a usage-accounting failure")

	body := string(joinFrames(append(append(served, counted...), terminal...)))
	wantContains(t, body, `"finish_reason":"length"`, "the turn did not close for the client")
	wantContains(t, body, `"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`,
		"the provider's own count was replaced by the Router's estimate")
	wantNotContains(t, body, "completion_tokens\":268", "the estimate overrode a stated count")
	wantContains(t, body, "data: [DONE]")
}

// Messages opens with the provider's authoritative input tokens and states no
// output count at all. The estimate fills the output leg and leaves the input
// leg alone, and the overall total is restated so the usage still adds up --
// without which the turn ended on usage_evidence_decreased or
// usage_total_mismatch and the client was told that instead of the deadline.
func TestTruncationFillsOnlyTheCountsTheStreamNeverStated(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.AnthropicMessagesV1)))
	requireNoError(t, err)
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated",
		"the router ended this stream at its own deadline, below the platform's", nil))
	requireNoError(t, err, "a usage the Router filled in made the stream contradict itself")

	body := string(joinFrames(append(served, terminal...)))
	// The client hears about the deadline, not about usage bookkeeping.
	wantContains(t, body, `"type":"timeout_error"`)
	wantContains(t, body, "at its own deadline")
	wantNotContains(t, body, "usage_evidence_decreased")
	wantNotContains(t, body, "usage_total_mismatch")
	wantContains(t, body, `"usage_source":"stream_estimate"`)
	wantContains(t, body, `"output_tokens":268`, "the output leg was not filled from the estimate")
}

// A turn the upstream finished, cut only in its transport, ends on the stop
// the upstream stated -- not on a length stop, and not on an error.
//
// Which stop travels is the whole basis for this change: a deadline stopped the
// model where it stood, so "stopped at a limit" is true of it. An upstream that
// already sent its finish_reason and its usage did not run out of room; it had
// said its piece, and the cut was the connection closing behind it. So the
// terminal states that upstream's own stop.
//
// It is still the format's own terminal rather than an error frame, and that is
// deliberate: a Chat stream holds its usage back until the terminal, so a turn
// that ends on the error frame bills as nothing however much the upstream
// stated mid-stream. Ending on the error here would have kept exactly the hole
// this change exists to close.
func TestACutAfterTheUpstreamFinishedStatesItsOwnStop(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
	requireNoError(t, err)
	// The upstream states its stop and its usage: the answer is finished.
	finished, _, _, err := stream.Push([]byte(
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m",` +
			`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n"))
	requireNoError(t, err)
	// Only then is the body cut, with no final chunk.
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, _ := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))

	body := string(joinFrames(append(append(served, finished...), terminal...)))
	wantContains(t, body, `"finish_reason":"stop"`,
		"the upstream's own stop did not reach the client")
	wantNotContains(t, body, `"finish_reason":"length"`,
		"a finished answer was restated as one stopped at a limit")
	// The count the upstream stated is the count that bills, and it only
	// reaches the client on a terminal that has somewhere to put it.
	wantContains(t, body, `"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`,
		"the turn billed as nothing")
	wantNotContains(t, body, "completion_tokens\":268", "the estimate overrode a stated count")
}

// A deadline does not fall on an SSE frame boundary.
//
// A transport read can end mid-frame, so a cut routinely leaves half a frame
// behind. That fragment is a consequence of the cut, not a provider sending
// something malformed, and it used to divert the whole turn down the
// decode-error path -- where the truncation terminal was never built, so the
// count was lost and the client got the bare error frame again. Whether the cut
// landed on a boundary cannot decide whether the turn bills.
func TestTruncationAtAPartialFrameStillTerminatesWithItsCount(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		engine := NewBuiltinEngine()
		requested := true
		stream, err := engine.NewStream(target, target, llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
		requireNoError(t, err)

		served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(target)))
		requireNoError(t, err)
		// Half of the next frame lands, then the deadline falls.
		partial, _, _, err := stream.Push([]byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chu`))
		requireNoError(t, err)
		stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
		terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
			llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
		requireNoError(t, err, "%s: a cut mid-frame was reported as malformed upstream JSON", target)

		body := string(joinFrames(append(append(served, partial...), terminal...)))
		wantNotContains(t, body, "invalid_upstream_json", "%s: the fragment became the client's answer", target)
		// The fragment itself never reaches the client.
		wantNotContains(t, body, "chat.completion.chu\"", "%s: a partial frame was published", target)
		if target == llmprotocol.OpenAIChatV1 {
			wantContains(t, body, `"completion_tokens":268`, "%s: the count was lost to the fragment", target)
			wantContains(t, body, `"finish_reason":"length"`)
			wantContains(t, body, "data: [DONE]")
		} else {
			wantContains(t, body, `"output_tokens":268`, "%s: the count was lost to the fragment", target)
			wantContains(t, body, `"type":"response.incomplete"`)
		}
	}
}

// A stream that had already broken the protocol ends on that, not on the
// deadline that arrived afterwards.
//
// A clean length stop here would present an upstream's violation to the client
// as an ordinary truncation, and reaching a deadline later does not make a
// malformed stream well-formed. The count has no terminal to ride and that is
// the right outcome, so it is reported rather than dropped quietly.
func TestACutAfterAProtocolViolationEndsOnTheViolation(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	_, _, _, err = stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
	requireNoError(t, err)
	// A complete frame that breaks the format's own envelope contract.
	_, _, _, pushErr := stream.Push([]byte(`data: {"object":"not.a.chunk","choices":[]}` + "\n\n"))
	requireError(t, pushErr, "a malformed envelope was accepted")

	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, _ := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))

	body := string(joinFrames(terminal))
	wantContains(t, body, "invalid_chat_stream_object",
		"the upstream's violation was hidden from the client")
	wantNotContains(t, body, `"finish_reason":"length"`,
		"a malformed stream was presented as an ordinary truncation")
	wantNotContains(t, body, "data: [DONE]")
}

// A turn the Router cut mid tool-call still terminates, and the call it closes
// says it is incomplete.
//
// Closing the open item was itself a regression risk: the completion event the
// engine synthesizes carries no tool call, and the Responses tool encoder
// refuses an item without one -- which left this turn with no terminal at all,
// strictly worse than the open item. The call is reconstructed from what it had
// streamed and marked incomplete, because a caller reading cut arguments as
// whole would parse a fragment as an argument object.
func TestTruncationMidToolCallClosesTheCallAsIncomplete(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)

	var served [][]byte
	for _, chunk := range []string{
		"event: response.created\n" + `data: {"type":"response.created","sequence_number":0,` +
			`"response":{"id":"resp_1","object":"response","created_at":0,"model":"m",` +
			`"status":"in_progress","output":[]}}` + "\n\n",
		"event: response.output_item.added\n" + `data: {"type":"response.output_item.added",` +
			`"sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc_1",` +
			`"call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}` + "\n\n",
		"event: response.function_call_arguments.delta\n" +
			`data: {"type":"response.function_call_arguments.delta","sequence_number":2,` +
			`"item_id":"fc_1","output_index":0,"delta":"{\"q\":\"par"}` + "\n\n",
	} {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		requireNoError(t, pushErr)
		served = append(served, frames...)
	}

	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
	requireNoError(t, err, "a cut mid tool-call produced no terminal at all")

	body := string(joinFrames(append(served, terminal...)))
	wantContains(t, body, `"type":"response.incomplete"`, "the turn never terminated")
	wantContains(t, body, `"status":"incomplete"`, "a cut call claimed to be whole")
	wantContains(t, body, `"arguments":"{\"q\":\"par"`,
		"the arguments the client was streamed were dropped")
	wantContains(t, body, `"output_tokens":268`, "the turn billed as nothing")

	readback, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)
	_, _, _, err = readback.Push([]byte(body))
	requireNoError(t, err, "this codec emitted a tool stream it cannot itself read")
}

// A completion held back pending a clean end of stream still establishes what
// the turn cost and how the model stopped.
//
// Such a terminal is dropped, because a turn that never reached a clean end of
// stream must not publish a success terminal. But it had already stated the
// authoritative count and the real stop, and a successful terminal never
// passes through the engine's event recorder -- so dropping the event without
// keeping that evidence replaced a provider's own numbers with the Router's
// estimate and a stop the model never reached.
func TestAHeldCompletionsCountAndStopSurviveACut(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
	requireNoError(t, err)
	finished, _, _, err := stream.Push([]byte(
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m",` +
			`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n"))
	requireNoError(t, err)
	// The upstream's sentinel arrives, so its completion is held pending a
	// clean end of stream that never comes.
	sentinel, _, _, err := stream.Push([]byte("data: [DONE]\n\n"))
	requireNoError(t, err)

	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(append(append(served, finished...), sentinel...), terminal...)))
	wantContains(t, body, `"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}`,
		"the provider's own count was replaced by the Router's estimate")
	wantNotContains(t, body, "completion_tokens\":268", "the estimate overrode a stated count")
	wantContains(t, body, `"finish_reason":"stop"`, "the stop the model reached was lost")
	wantNotContains(t, body, `"finish_reason":"length"`,
		"a finished answer was restated as one stopped at a limit")
}

// The cause the host states travels beside the stop, whichever it is.
//
// The stop has to be "length" -- Chat's vocabulary is closed and holds nothing
// for a turn infrastructure ended -- and read literally that says the model
// exhausted its budget. For a deadline that is near enough: the turn did run
// out of a budget, just a wall-clock one. For a model that went quiet or a
// connection that broke it is wrong, and a client acting on it would shorten
// its prompt when it should have retried. So the mechanism travels beside the
// stop and the stop is no longer the only thing a client can read. The host
// names it; this codec knows the vocabulary and nothing about the host's own
// codes for its cuts.
func TestTruncationNamesItsCause(t *testing.T) {
	for _, probe := range []struct {
		code, cause string
	}{
		// The codes are whatever the host calls its cuts; this codec never
		// reads them, so these are made up.
		{code: "host_budget_elapsed", cause: llmprotocol.TruncationDeadline},
		{code: "host_model_went_quiet", cause: llmprotocol.TruncationSilence},
		{code: "host_no_first_byte", cause: llmprotocol.TruncationSilence},
		{code: "host_socket_closed", cause: llmprotocol.TruncationTransport},
	} {
		engine := NewBuiltinEngine()
		requested := true
		stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
			llmprotocol.StreamContext{
				Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
			})
		requireNoError(t, err)

		served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
		requireNoError(t, err)
		stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
		stream.SetTruncationCause(probe.cause)
		terminal, _, _, _ := stream.Finalize(llmprotocol.NewError(
			llmprotocol.ErrorUpstreamTimeout, probe.code, "the router ended this stream", nil))

		body := string(joinFrames(append(served, terminal...)))
		// The cause sits on the chunk that states the stop it qualifies: a
		// header cannot be sent once the stream has started, so a client
		// reading the stop finds the cause in the same frame.
		wantContains(t, body,
			`"finish_reason":"length"}],"rayline":{"truncated":"`+probe.cause+`"}`,
			"%s did not name its cause beside its stop", probe.code)
		wantContains(t, body, "data: [DONE]", "%s left the stream open", probe.code)
	}
}

// A turn that ended on its own states no cause. The member says the Router
// ended this turn; on a turn the model finished there is nothing to say, and a
// client must not read a truncation into an ordinary stop.
func TestANaturalTerminalNamesNoCause(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	var frames [][]byte
	for _, chunk := range []string{
		truncatedUpstreamChunk(llmprotocol.OpenAIChatV1),
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m",` +
			`"choices":[{"index":0,"delta":{},"finish_reason":"length"}],` +
			`"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n",
		"data: [DONE]\n\n",
	} {
		pushed, _, _, pushErr := stream.Push([]byte(chunk))
		requireNoError(t, pushErr)
		frames = append(frames, pushed...)
	}
	terminal, _, _, err := stream.Finalize(nil)
	requireNoError(t, err)

	body := string(joinFrames(append(frames, terminal...)))
	// A model that really did hit its output limit still says so, and says
	// only that.
	wantContains(t, body, `"finish_reason":"length"`)
	wantNotContains(t, body, "rayline",
		"a turn the model ended was reported as one the Router cut")
}

// A count the upstream stated before the output is a snapshot, not a settlement.
//
// Anthropic's message_start states output_tokens 1 before the first delta.
// Measured live 2026-10-26 with a 20 s deadline: a Chat client received 781
// content deltas and then a usage chunk saying completion_tokens 1, marked
// authoritative, because the stated count outranked the estimate. Both are
// lower bounds on what the upstream charged; after output has run past the
// statement, the larger one is the better one. The input leg stays as stated.
func TestAnOutputCountTheTurnRanPastIsRestatedFromTheEstimate(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	// message_start states the prompt and one output token; the text follows.
	served, _, _, err := stream.Push([]byte(
		"event: message_start\n" +
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",` +
			`"model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":37,"output_tokens":1}}}` +
			"\n\nevent: content_block_start\n" +
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` +
			"\n\nevent: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` +
			strings.Repeat("one two three four five six seven eight nine ten ", 40) + `"}}` + "\n\n"))
	requireNoError(t, err)
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(served, terminal...)))
	wantContains(t, body, `"finish_reason":"length"`)
	wantContains(t, body, `"completion_tokens":268`, "the snapshot was billed as the turn's output")
	wantNotContains(t, body, `"completion_tokens":1,`, "a count the turn ran past was kept as settled")
	wantContains(t, body, `"prompt_tokens":37`, "the stated prompt count was replaced by an estimate")
}

// A completion held back pending a clean end of stream does not make a
// snapshot fresh again. Chat's [DONE] completion carries the usage the stream
// had accumulated, whatever its age; only a count the upstream actually stated
// says the output leg is settled.
func TestAHeldCompletionDoesNotRefreshASnapshot(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	var frames [][]byte
	for _, chunk := range []string{
		// A snapshot: one output token, stated before the answer.
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,` +
			`"delta":{"role":"assistant","content":"one"},"finish_reason":null}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}` + "\n\n",
		// The answer runs on past it.
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,` +
			`"delta":{"content":"` + strings.Repeat(" two three four five", 30) + `"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		// The sentinel: the completion is held pending a clean end of stream.
		"data: [DONE]\n\n",
	} {
		pushed, _, _, pushErr := stream.Push([]byte(chunk))
		requireNoError(t, pushErr)
		frames = append(frames, pushed...)
	}
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(frames, terminal...)))
	wantContains(t, body, `"completion_tokens":268`, "the held completion made the snapshot authoritative again")
	wantNotContains(t, body, `"completion_tokens":1,`, "a count the turn ran past was billed as settled")
	// The stop the model reached still travels.
	wantContains(t, body, `"finish_reason":"stop"`)
}

// The same statement made after all the output is a settlement, and the
// estimate does not touch it, however large.
func TestAnOutputCountStatedAfterTheOutputStands(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(
		"event: message_start\n" +
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",` +
			`"model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":37,"output_tokens":1}}}` +
			"\n\nevent: content_block_start\n" +
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` +
			"\n\nevent: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ninety-one is"}}` +
			"\n\nevent: content_block_stop\n" + `data: {"type":"content_block_stop","index":0}` +
			"\n\nevent: message_delta\n" +
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}` +
			"\n\n"))
	requireNoError(t, err)
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(served, terminal...)))
	wantContains(t, body, `"completion_tokens":5`, "the upstream's settled count was overridden")
	wantNotContains(t, body, `"completion_tokens":268`, "an estimate replaced a settlement")
}

// Output the decoder only produces at the cut is not published.
//
// A transport read can end after a complete event and before the blank line
// that ends it; the framer holds the event and Finalize flushes it -- after
// the Router has counted the turn. Publishing it would deliver output the
// terminal's count does not cover (Codex review on the pull request). It is
// withheld, and a count or a stop the same flush carried is still honoured.
func TestOutputFlushedAtTheCutIsNotPublished(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
	requireNoError(t, err)
	// A complete event whose terminating blank line never arrives.
	held, _, _, err := stream.Push([]byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk",` +
		`"choices":[{"index":0,"delta":{"content":" UNCOUNTED TAIL"},"finish_reason":null}]}` + "\n"))
	requireNoError(t, err)
	wantEqual(t, 0, len(held), "an unterminated event was published before the framer closed it")

	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationDeadline)
	terminal, _, diagnostics, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(served, terminal...)))
	wantNotContains(t, body, "UNCOUNTED TAIL", "output decoded at the cut reached the client unbilled")
	wantContains(t, body, `"completion_tokens":268`)
	wantContains(t, body, "data: [DONE]")
	var said bool
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "stream.output" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			said = true
		}
	}
	wantTrue(t, said, "the withheld output was not named in the diagnostics: %v", diagnostics)
}

// A count the same flush carries is still the count that bills: the upstream's
// usage frame, cut off from its blank line, is a settlement and not output.
func TestACountFlushedAtTheCutStillBills(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
	requireNoError(t, err)
	_, _, _, err = stream.Push([]byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n"))
	requireNoError(t, err)

	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationDeadline)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(served, terminal...)))
	wantContains(t, body, `"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`,
		"a settlement flushed at the cut was replaced by the estimate")
}

// A Responses completion held back pending a clean end of stream is a fresh
// settlement, unlike Chat's sentinel: the provider's final count is on the
// resource itself. A transport cut after it keeps that count, and does not
// restate it from the estimate as if it were a snapshot.
func TestAHeldResponsesCompletionIsASettlement(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)

	item := `{"content":[{"annotations":[],"text":"ninety-one is","type":"output_text"}],"id":"item_1",` +
		`"role":"assistant","status":"completed","type":"message"}`
	var frames [][]byte
	for _, chunk := range []string{
		truncatedUpstreamChunk(llmprotocol.OpenAIResponsesV1),
		"event: response.output_text.done\n" + `data: {"type":"response.output_text.done","sequence_number":4,` +
			`"item_id":"item_1","output_index":0,"content_index":0,"text":"ninety-one is"}` + "\n\n",
		"event: response.content_part.done\n" + `data: {"type":"response.content_part.done","sequence_number":5,` +
			`"item_id":"item_1","output_index":0,"content_index":0,"part":{"annotations":[],"text":"ninety-one is","type":"output_text"}}` + "\n\n",
		"event: response.output_item.done\n" + `data: {"type":"response.output_item.done","sequence_number":6,` +
			`"output_index":0,"item":` + item + `}` + "\n\n",
		"event: response.completed\n" + `data: {"type":"response.completed","sequence_number":7,"response":{"id":"resp_1",` +
			`"object":"response","created_at":0,"model":"m","status":"completed","output":[` + item + `],` +
			`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}` + "\n\n",
	} {
		pushed, _, _, pushErr := stream.Push([]byte(chunk))
		requireNoError(t, pushErr)
		frames = append(frames, pushed...)
	}
	// The completion is held pending a clean end of stream; the body is cut.
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationTransport)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(frames, terminal...)))
	wantContains(t, body, `"output_tokens":5`, "the provider's settlement was restated from the estimate")
	wantNotContains(t, body, `"output_tokens":268`, "an estimate replaced a settlement")
}

// A settlement flushed at the cut reaches a Messages client too. The Messages
// encoder used to write the count the caller installed before finalization,
// so a usage frame cut off from its blank line was ignored and the client
// billed from the estimate.
func TestASettlementFlushedAtTheCutReachesAMessagesClient(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.AnthropicMessagesV1)))
	requireNoError(t, err)
	// The provider's final count, cut off from its blank line.
	_, _, _, err = stream.Push([]byte("event: content_block_stop\n" + `data: {"type":"content_block_stop","index":0}` +
		"\n\nevent: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":500}}` + "\n"))
	requireNoError(t, err)

	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationTransport)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(served, terminal...)))
	wantContains(t, body, `"output_tokens":500`, "the settlement flushed at the cut was ignored")
	wantNotContains(t, body, `"output_tokens":268`, "the client was billed from the estimate")
	wantContains(t, body, `"usage_source":"stream_estimate"`)
}

// A completion the transport cut off from its blank line still settles the
// turn. The decoder only produces it at Finalize; it is harvested as a held
// completion is, so its count, its stop and its tier reach the terminal.
func TestACompletionFlushedAtTheCutSettlesTheTurn(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)

	item := `{"content":[{"annotations":[],"text":"ninety-one is","type":"output_text"}],"id":"item_1",` +
		`"role":"assistant","status":"completed","type":"message"}`
	var frames [][]byte
	for _, chunk := range []string{
		truncatedUpstreamChunk(llmprotocol.OpenAIResponsesV1),
		"event: response.output_text.done\n" + `data: {"type":"response.output_text.done","sequence_number":4,` +
			`"item_id":"item_1","output_index":0,"content_index":0,"text":"ninety-one is"}` + "\n\n",
		"event: response.content_part.done\n" + `data: {"type":"response.content_part.done","sequence_number":5,` +
			`"item_id":"item_1","output_index":0,"content_index":0,"part":{"annotations":[],"text":"ninety-one is","type":"output_text"}}` + "\n\n",
		"event: response.output_item.done\n" + `data: {"type":"response.output_item.done","sequence_number":6,` +
			`"output_index":0,"item":` + item + `}` + "\n\n",
		// The completion, cut off from its blank line.
		"event: response.completed\n" + `data: {"type":"response.completed","sequence_number":7,"response":{"id":"resp_1",` +
			`"object":"response","created_at":0,"model":"m","status":"completed","output":[` + item + `],` +
			`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}` + "\n",
	} {
		pushed, _, _, pushErr := stream.Push([]byte(chunk))
		requireNoError(t, pushErr)
		frames = append(frames, pushed...)
	}
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationTransport)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(frames, terminal...)))
	wantContains(t, body, `"output_tokens":5`, "the flushed completion's count was discarded")
	wantNotContains(t, body, `"output_tokens":268`, "the turn was restated from the estimate")
	wantNotContains(t, body, `"max_output_tokens"`, "a finished turn was reported as stopped at a limit")
}

// The stop a Messages message_delta states survives a cut before message_stop.
// The delta carries the model's stop and its final count; a turn cut between
// the two had finished, and a Chat client is told the stop it reached.
func TestAStopStatedByAMessageDeltaSurvivesACut(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)

	served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.AnthropicMessagesV1) +
		"event: content_block_stop\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}` + "\n\n"))
	requireNoError(t, err)
	// No message_stop: the connection is cut first.
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationTransport)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(served, terminal...)))
	wantContains(t, body, `"finish_reason":"stop"`, "the stop the model reached was lost")
	wantNotContains(t, body, `"finish_reason":"length"`, "a finished answer was reported as cut short")
	wantContains(t, body, `"completion_tokens":5`, "the delta's settlement was replaced")
}

// A cut can land inside a JSON object or inside a multibyte character. The
// structural checks name those in their own words rather than as an EOF, and
// each is what a cut leaves behind: complete as far as it goes.
func TestACutInsideAnObjectOrACharacterStillBills(t *testing.T) {
	for name, tail := range map[string]string{
		"inside an object":   `data: {"id":"chatcmpl-1","object":"chat.completion.chu`,
		"inside a value":     `data: {"id":"chatcmpl-1"`,
		"inside a character": "data: {\"id\":\"chatcmpl-1\",\"object\":\"\xe2\x82",
	} {
		engine := NewBuiltinEngine()
		requested := true
		stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
			llmprotocol.StreamContext{
				Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
			})
		requireNoError(t, err)
		served, _, _, err := stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
		requireNoError(t, err)
		_, _, _, err = stream.Push([]byte(tail))
		requireNoError(t, err, "%s: the fragment was refused before the cut", name)
		stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
		stream.SetTruncationCause(llmprotocol.TruncationDeadline)
		terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
			llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
		requireNoError(t, err, "%s: a cut was reported as malformed upstream JSON", name)

		body := string(joinFrames(append(served, terminal...)))
		wantContains(t, body, `"completion_tokens":268`, "%s: the count was lost", name)
		wantContains(t, body, "data: [DONE]", "%s: the stream was left open", name)
		wantNotContains(t, body, "invalid_upstream", "%s: the fragment became the client's answer", name)
	}
}

// A byte that is simply invalid is not a cut, wherever the deadline fell. A
// character cut in half would have been whole had the bytes kept coming; 0xff
// never would, and a provider that sent it broke the protocol.
func TestAnInvalidByteAtTheCutIsStillAViolation(t *testing.T) {
	engine := NewBuiltinEngine()
	requested := true
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: t.Context(), Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)
	_, _, _, err = stream.Push([]byte(truncatedUpstreamChunk(llmprotocol.OpenAIChatV1)))
	requireNoError(t, err)
	_, _, _, err = stream.Push([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"\xff"))
	requireNoError(t, err)
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationDeadline)
	terminal, _, _, _ := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))

	body := string(joinFrames(terminal))
	wantContains(t, body, "invalid_upstream_utf8", "an invalid byte was passed off as a cut")
	wantNotContains(t, body, `"finish_reason":"length"`, "a malformed stream was presented as an ordinary truncation")
}

// An image generation open at the cut has no state the closer could
// synthesize, so it is left open rather than closed as something it is not.
// The terminal still arrives, with its count; it used to be replaced by a
// bare image_generation_item_invalid error that dropped the usage.
func TestAnOpenImageGenerationAtTheCutStillTerminates(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: t.Context()})
	requireNoError(t, err)

	var frames [][]byte
	for _, chunk := range []string{
		"event: response.created\n" + `data: {"type":"response.created","sequence_number":0,` +
			`"response":{"id":"resp_1","object":"response","created_at":0,"model":"m","status":"in_progress","output":[]}}` + "\n\n",
		"event: response.output_item.added\n" + `data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,` +
			`"item":{"type":"image_generation_call","id":"ig_1","status":"in_progress","result":null}}` + "\n\n",
		"event: response.image_generation_call.in_progress\n" + `data: {"type":"response.image_generation_call.in_progress",` +
			`"sequence_number":2,"item_id":"ig_1","output_index":0}` + "\n\n",
		"event: response.output_item.added\n" + `data: {"type":"response.output_item.added","sequence_number":3,"output_index":1,` +
			`"item":{"type":"message","id":"item_2","role":"assistant","status":"in_progress","content":[]}}` + "\n\n",
		"event: response.content_part.added\n" + `data: {"type":"response.content_part.added","sequence_number":4,"item_id":"item_2",` +
			`"output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}` + "\n\n",
		"event: response.output_text.delta\n" + `data: {"type":"response.output_text.delta","sequence_number":5,"item_id":"item_2",` +
			`"output_index":1,"content_index":0,"delta":"here is your"}` + "\n\n",
	} {
		pushed, _, _, pushErr := stream.Push([]byte(chunk))
		requireNoError(t, pushErr)
		frames = append(frames, pushed...)
	}
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationDeadline)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamTimeout, "stream_truncated", "deadline", nil))
	requireNoError(t, err, "an open image generation turned the cut into an error")

	body := string(joinFrames(append(frames, terminal...)))
	wantContains(t, body, `"type":"response.incomplete"`, "the turn did not terminate")
	wantContains(t, body, `"output_tokens":268`, "the count was dropped")
	wantNotContains(t, body, "image_generation_item_invalid")
	wantNotContains(t, body, "event: error")
	// The text item the cut landed in is still closed, as incomplete.
	wantContains(t, body, `"role":"assistant","status":"incomplete","type":"message"`)
}

// The drop that cannot be helped is loud.
//
// Chat omits its usage-only chunk unless the client requested it, so a cut
// turn's count has nowhere to go for a client that did not. That is the
// format's opt-in rather than a capability this codec lacks, but it is still a
// count that did not reach the client, and the silence around exactly this
// kind of drop is what hid the defect for two releases. So it is named, with
// the encoder that could not write it.
func TestADroppedTruncationCountIsLogged(t *testing.T) {
	events := captureWarnEvents(t)
	reportedUnwritableTruncationUsage.Range(func(key, _ any) bool {
		reportedUnwritableTruncationUsage.Delete(key)
		return true
	})

	// A client that did not ask for the usage chunk.
	body := truncatedStreamTerminal(t, llmprotocol.OpenAIChatV1, estimatedTruncationUsage(), false)

	// The turn still terminates validly for the client.
	wantContains(t, body, `"finish_reason":"length"`)
	wantContains(t, body, "data: [DONE]")
	wantNotContains(t, body, `"usage"`)

	var named bool
	for _, event := range events.all() {
		if event.name != "truncation_usage_unwritten" {
			continue
		}
		wantEqual(t, "chat", event.fields["encoder"], "the warning did not name the encoder")
		reason, _ := event.fields["reason"].(string)
		wantContains(t, reason, "include_usage",
			"the warning did not say why the count could not travel")
		named = true
	}
	wantTrue(t, named, "the warning did not name the encoder that dropped the count: %v", events.all())
}

// capturedEvent is one event the core emitted through its diagnostics sink.
type capturedEvent struct {
	level  slog.Level
	name   string
	fields map[string]interface{}
}

type capturedEvents struct {
	mu     sync.Mutex
	events []capturedEvent
}

func (captured *capturedEvents) all() []capturedEvent {
	captured.mu.Lock()
	defer captured.mu.Unlock()
	return append([]capturedEvent(nil), captured.events...)
}

// captureWarnEvents installs a sink that keeps every event the core emits for
// the rest of the test. The sink is process-global and cannot be removed, so
// the cleanup installs one that drops events rather than leaving this test's
// recorder attached to the next test.
func captureWarnEvents(t *testing.T) *capturedEvents {
	t.Helper()
	captured := &capturedEvents{}
	logging.SetEventSink(func(level slog.Level, _ string, event string, fields map[string]interface{}) {
		copied := make(map[string]interface{}, len(fields))
		for key, value := range fields {
			copied[key] = value
		}
		captured.mu.Lock()
		captured.events = append(captured.events, capturedEvent{level: level, name: event, fields: copied})
		captured.mu.Unlock()
	})
	t.Cleanup(func() {
		logging.SetEventSink(func(slog.Level, string, string, map[string]interface{}) {})
	})
	return captured
}

// The assertions below keep this file on the standard library alone, as the
// rest of this module is.

func requireNoError(t *testing.T, err error, msgAndArgs ...any) {
	t.Helper()
	if err != nil {
		t.Fatalf("%sunexpected error: %v", assertionPrefix(msgAndArgs), err)
	}
}

func requireError(t *testing.T, err error, msgAndArgs ...any) {
	t.Helper()
	if err == nil {
		t.Fatalf("%sexpected an error and got none", assertionPrefix(msgAndArgs))
	}
}

func requireErrorIs(t *testing.T, err, target error, msgAndArgs ...any) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("%serror %v is not %v", assertionPrefix(msgAndArgs), err, target)
	}
}

func wantContains(t *testing.T, body, substring string, msgAndArgs ...any) {
	t.Helper()
	if !strings.Contains(body, substring) {
		t.Errorf("%sbody does not contain %q:\n%s", assertionPrefix(msgAndArgs), substring, body)
	}
}

func wantNotContains(t *testing.T, body, substring string, msgAndArgs ...any) {
	t.Helper()
	if strings.Contains(body, substring) {
		t.Errorf("%sbody should not contain %q:\n%s", assertionPrefix(msgAndArgs), substring, body)
	}
}

func wantEqual(t *testing.T, want, got any, msgAndArgs ...any) {
	t.Helper()
	if want != got {
		t.Errorf("%sgot %v, want %v", assertionPrefix(msgAndArgs), got, want)
	}
}

func wantTrue(t *testing.T, condition bool, msgAndArgs ...any) {
	t.Helper()
	if !condition {
		t.Errorf("%sexpected true", assertionPrefix(msgAndArgs))
	}
}

// assertionPrefix renders an optional message and its arguments ahead of the
// assertion's own text, the way the common assertion libraries do.
func assertionPrefix(msgAndArgs []any) string {
	if len(msgAndArgs) == 0 {
		return ""
	}
	format, ok := msgAndArgs[0].(string)
	if !ok {
		return fmt.Sprint(msgAndArgs...) + ": "
	}
	return fmt.Sprintf(format, msgAndArgs[1:]...) + ": "
}
