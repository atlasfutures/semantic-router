package extproc

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// What a cut turn tells the client it cost.
//
// The Router already counts a truncated turn for its own telemetry, but the
// count stops there: the client is sent an error frame with no numbers on it,
// so the proxy in front of the cell bills nothing while the upstream billed
// for every token it generated. Measured on the dev cell 2026-09-04, one such
// turn cost $0.019 that nobody was charged for.
//
// So the error frame carries the count, beside the error rather than inside
// it, and says the count is an estimate. The proxy reads both.

type truncationFrame struct {
	Type  string `json:"type"`
	Usage *struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
	UsageSource string `json:"usage_source"`
	Error       *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// A turn cut with nothing but what the Router forwarded is billed from that,
// and the frame says the numbers are an estimate.
func TestTruncationFrameCarriesTheEstimatedCount(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	// 800 bytes of request text, which is 200 tokens at four bytes to a token.
	ctx.VSRContextTextBytes = 800

	body := streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))

	frame := truncationFrameFrom(t, body)
	if frame.Usage == nil {
		t.Fatalf("the cut turn told the client nothing about what it cost:\n%s", body)
	}
	if frame.Usage.InputTokens != 200 {
		t.Errorf("input_tokens = %d, want the 200 the Router measured", frame.Usage.InputTokens)
	}
	if frame.Usage.OutputTokens <= 0 {
		t.Errorf("output_tokens = %d, want what the stream carried", frame.Usage.OutputTokens)
	}
	if frame.UsageSource != "stream_estimate" {
		t.Errorf("usage_source = %q, want stream_estimate", frame.UsageSource)
	}
	// The estimate rides beside the error object, never inside it, so the
	// client's SDK parses the failure exactly as it did before.
	if bytes.Contains(body, []byte(`"message":"the router ended this stream at its own deadline, below the platform's","usage"`)) {
		t.Errorf("the estimate was written inside the error object:\n%s", body)
	}
	if frame.Error == nil || frame.Error.Type == "" {
		t.Errorf("the frame stopped naming the failure:\n%s", body)
	}
}

// When the upstream stated a prompt count before the cut, that number is the
// one the client is told -- the Router's own estimate of a prompt is low,
// because it omits tool schemas and images. The row is still an estimate: the
// turn did not finish, so nothing about it is a settlement.
func TestTruncationFrameKeepsAnAuthoritativePromptCount(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.StartTime = time.Now()
	ctx.VSRContextTextBytes = 800

	streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))
	streamDeadlineResponseBody(t, router, ctx, upstreamUsageChunk())

	ctx.StartTime = time.Now().Add(-600 * time.Second)
	body := streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(2))

	frame := truncationFrameFrom(t, body)
	if frame.Usage == nil {
		t.Fatalf("the cut turn told the client nothing about what it cost:\n%s", body)
	}
	if frame.Usage.InputTokens != 220 {
		t.Errorf("input_tokens = %d, want the 220 the upstream stated", frame.Usage.InputTokens)
	}
	if frame.Usage.OutputTokens != 16030 {
		t.Errorf("output_tokens = %d, want the 16030 the upstream stated", frame.Usage.OutputTokens)
	}
	if frame.UsageSource != "stream_estimate" {
		t.Errorf("usage_source = %q, want stream_estimate on a turn that did not finish", frame.UsageSource)
	}
}

// A turn that finished is a settlement, and settlements say nothing about
// estimates. Its usage rides where it always did, on message_delta.
func TestCompletedStreamCarriesNoUsageSource(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.StartTime = time.Now()

	var body []byte
	body = append(body, streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))...)
	body = append(body, streamDeadlineResponseBody(t, router, ctx, completedUpstreamChunk())...)
	body = append(body, endOfStreamResponseBody(t, router, ctx)...)

	if bytes.Contains(body, []byte("usage_source")) {
		t.Fatalf("a completed turn was marked as an estimate:\n%s", body)
	}
	if !bytes.Contains(body, []byte("message_delta")) {
		t.Fatalf("the completed turn published no terminal usage:\n%s", body)
	}
}

// An upstream that fails is not a turn the Router cut. It has no estimate to
// offer -- the failure came from the provider, and the provider said what it
// cost or said nothing.
func TestUpstreamErrorFrameCarriesNoUsage(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.StartTime = time.Now()

	var body []byte
	body = append(body, streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))...)
	body = append(body, streamDeadlineResponseBody(t, router, ctx, upstreamErrorChunk())...)

	if !bytes.Contains(body, []byte(`"type":"error"`)) {
		t.Fatalf("the upstream failure did not reach the client:\n%s", body)
	}
	if bytes.Contains(body, []byte("usage_source")) {
		t.Fatalf("a provider failure was marked as the Router's estimate:\n%s", body)
	}
	frame := truncationFrameFrom(t, body)
	if frame.Usage != nil {
		t.Fatalf("a provider failure carried a count the Router invented:\n%s", body)
	}
}

func truncationFrameFrom(t *testing.T, body []byte) truncationFrame {
	t.Helper()
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var frame truncationFrame
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
			continue
		}
		if frame.Type == "error" {
			return frame
		}
	}
	t.Fatalf("no error frame reached the client:\n%s", body)
	return truncationFrame{}
}

func endOfStreamResponseBody(t *testing.T, router *OpenAIRouter, ctx *RequestContext) []byte {
	t.Helper()
	return responseBodyMutationBytes(router.handleSemanticStreamingResponseBody(nil, true, ctx))
}

// completedUpstreamChunk ends the turn the way a provider that finished does.
func completedUpstreamChunk() string {
	return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
		`"model":"xiaomi/mimo-v2.5-pro","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":220,"completion_tokens":12,"total_tokens":232}}` + "\n\n" +
		"data: [DONE]\n\n"
}

// upstreamErrorChunk is a provider failure mid-stream, not a cut.
func upstreamErrorChunk() string {
	return `data: {"error":{"message":"mock provider stream failed","type":"server_error",` +
		`"param":null,"code":"provider_overloaded"}}` + "\n\n"
}
