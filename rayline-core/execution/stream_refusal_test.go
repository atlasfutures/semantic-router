package execution

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

// The runtimes' context-window refusals, as their servers write them.
const (
	// TensorFold: 400 on a buffered request; the same error object as a
	// Chat stream's first event after the role chunk.
	tensorFoldContextError = `{"message":"This server's maximum context length is 1048576 tokens, but the rendered prompt has 1048000 tokens and requests 4096 reply tokens, which exceeds the context window. Reduce the prompt to at most 1044480 prompt tokens or request at most 576 reply tokens, including chat template and thinking tokens.","type":"invalid_request_error","param":null,"code":"context_length_exceeded"}`
	// vLLM's engine-side check, raised past the renderer's own.
	vllmModelLengthError = `{"error":{"message":"The decoder prompt (length 40000) is longer than the maximum model length of 32768. Make sure that ` + "`max_model_len`" + ` is no smaller than the number of text tokens.","type":"BadRequestError","param":null,"code":400}}`
)

func TestRuntimeContextRefusalRecognisesEachRuntime(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want bool
	}{
		{"tensorfold chat", `{"error":` + tensorFoldContextError + `}`, true},
		{"tensorfold messages", `{"type":"error","error":{"type":"invalid_request_error","message":"This server's maximum context length is 1048576 tokens, but the rendered prompt has 1048577 tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation."}}`, true},
		{"vllm renderer", `{"error":{"message":"This model's maximum context length is 32768 tokens. However, your prompt contains 40000 input tokens, which exceeds this limit even without any output tokens. Please reduce the length of the input prompt.","type":"BadRequestError","param":"input_tokens","code":400}}`, true},
		{"vllm max_tokens", `{"error":{"message":"Input length (40000) exceeds model's maximum context length (32768).","type":"BadRequestError","param":null,"code":400}}`, true},
		{"vllm engine", vllmModelLengthError, true},
		{"vllm engine, prompt fills the window", `{"error":{"message":"The decoder prompt (length 32768) plus the number of requested output tokens (at least 1) is longer than the maximum model length of 32768.","type":"BadRequestError","param":null,"code":400}}`, true},
		{"other 400", `{"error":{"message":"temperature must be at most 2","type":"BadRequestError","param":null,"code":400}}`, false},
	} {
		if got := runtimeContextRefusal(http.StatusBadRequest, []byte(tc.body)); got != tc.want {
			t.Errorf("%s: %v", tc.name, got)
		}
	}
}

func TestStreamPeekFindsARefusalBeforeOutputAndReplaysEverythingElse(t *testing.T) {
	for _, tc := range []struct {
		name   string
		stream string
		want   bool
	}{
		{"chat after the role chunk", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\ndata: {\"error\":" + tensorFoldContextError + "}\n\n", true},
		{"chat with no role chunk", "data: {\"error\":" + tensorFoldContextError + "}\n\ndata: [DONE]\n\n", true},
		{"messages", "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"role\":\"assistant\",\"content\":[]}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"This server's maximum context length is 1048576 tokens, but the rendered prompt has 1048577 tokens.\"}}\n\n", true},
		{"responses", "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\nevent: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"r\"}}\n\nevent: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"r\",\"status\":\"failed\",\"error\":{\"code\":\"invalid_prompt\",\"message\":\"This server's maximum context length is 1048576 tokens, but the rendered prompt has 1048577 tokens.\"}}}\n\n", true},
		{"output first", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\ndata: {\"error\":" + tensorFoldContextError + "}\n\n", false},
		{"keepalive before the refusal", ": keepalive\n\ndata: {\"error\":" + tensorFoldContextError + "}\n\n", true},
		{"another error", "data: {\"error\":{\"message\":\"engine fault\",\"type\":\"server_error\"}}\n\n", false},
		{"ordinary stream", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ready\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", false},
		{"cut short", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}", false},
		{"output naming the marker, error null", "data: {\"error\":null,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"exceed_context_size_error\"},\"finish_reason\":null}]}\n\n", false},
		{"flat responses error", "event: error\ndata: {\"type\":\"error\",\"sequence_number\":0,\"code\":\"context_length_exceeded\",\"message\":\"too long\"}\n\n", true},
		{"comment inside the refusal", "event: error\n: trace\ndata: {\"error\":" + tensorFoldContextError + "}\n\n", true},
		{"CR separators", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\r\rdata: {\"error\":" + tensorFoldContextError + "}\r\r", true},
		{"BOM first", "\ufeffdata: {\"error\":" + tensorFoldContextError + "}\n\n", true},
		{"no blank line at EOF", "data: {\"error\":" + tensorFoldContextError + "}", true},
		{"repeated event field", "event: x\nevent: response.failed\ndata: {\"response\":{\"error\":{\"code\":\"invalid_prompt\",\"message\":\"This server's maximum context length is 8 tokens\"}}}\n\n", true},
		{"first event over the limit", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"" + strings.Repeat("x", 2*streamPeekLimit) + "\"},\"finish_reason\":null}]}\n\n", false},
	} {
		refused, body := peekStreamContextRejection(io.NopCloser(strings.NewReader(tc.stream)))
		if refused != tc.want {
			t.Errorf("%s: refused=%v", tc.name, refused)
		}
		if replayed, err := io.ReadAll(body); err != nil || string(replayed) != tc.stream {
			t.Errorf("%s: replayed %q, %v", tc.name, replayed, err)
		}
	}
}

func TestStreamPeekStopsWaitingOnASilentStream(t *testing.T) {
	// A role chunk, then a long silent prefill: the peek gives the stream on
	// after streamPeekWait, and the read it left pending is relayed.
	reader, writer := io.Pipe()
	opening := "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
	rest := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ready\"},\"finish_reason\":\"stop\"}]}\n\n"
	go func() {
		_, _ = io.WriteString(writer, opening)
		time.Sleep(streamPeekWait + 500*time.Millisecond)
		_, _ = io.WriteString(writer, rest)
		_ = writer.Close()
	}()
	started := time.Now()
	refused, body := peekStreamContextRejection(reader)
	if refused || time.Since(started) > streamPeekWait+250*time.Millisecond {
		t.Fatalf("refused=%v after %s", refused, time.Since(started))
	}
	replayed, err := io.ReadAll(body)
	if err != nil || string(replayed) != opening+rest {
		t.Fatalf("replayed %q, %v", replayed, err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
}

type failingBody struct{ io.Reader }

func (failingBody) Close() error { return nil }

func TestStreamPeekDecidesOnACompleteEventWithTheStreamStillOpen(t *testing.T) {
	for _, separator := range []string{"\r\r", "\n\n", "\r\n\r\n"} {
		reader, writer := io.Pipe()
		event := "data: {\"error\":" + tensorFoldContextError + "}" + separator
		go func() { _, _ = io.WriteString(writer, event) }()
		started := time.Now()
		refused, body := peekStreamContextRejection(reader)
		if !refused || time.Since(started) > streamPeekWait/2 {
			t.Errorf("%q: refused=%v after %s", separator, refused, time.Since(started))
		}
		_ = writer.Close()
		_ = body.Close()
	}
}

func TestStreamPeekKeepsTheBodysReadError(t *testing.T) {
	stream := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ready\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	_, body := peekStreamContextRejection(failingBody{io.MultiReader(strings.NewReader(stream), iotest.ErrReader(io.ErrUnexpectedEOF))})
	replayed, err := io.ReadAll(body)
	if string(replayed) != stream || err != io.ErrUnexpectedEOF {
		t.Fatalf("replayed %q, %v", replayed, err)
	}
}

func TestLocalStreamedContextRefusalReconsultsToCloud(t *testing.T) {
	// A local runtime that opens the stream with 200 and refuses the prompt
	// as its first event (TensorFold) re-consults before any output, as a
	// 400 refusal does.
	localCalls := 0
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		localCalls++
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"error\":"+tensorFoldContextError+"}\n\n")
	}))
	defer local.Close()
	cloudCalls := 0
	cloud := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cloudCalls++
		_, _ = io.Copy(io.Discard, r.Body)
		modelPinHeaders(w.Header())
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ready\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":1,\"total_tokens\":21}}\n\ndata: [DONE]\n\n")
	}))
	defer cloud.Close()
	config := capacityEstimateConfig(t, local.URL, cloud.URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: fitPolicy}
	_, httpServer := v2Server(t, config, fixture)
	turn := `{"model":"arc-1.0","stream":true,"max_tokens":64,` + v2Tools + `,"messages":[` + userTask + `]}`
	status, header, response := executeWithHeaders(t, httpServer.URL, turn, "local-stream-refuses")
	if status != 200 || header.Get("X-Rayline-Executed-Action-Id") == localActionID || !strings.Contains(string(response), "ready") || strings.Contains(string(response), "maximum context length") {
		t.Fatalf("status %d executed %q: %.400s", status, header.Get("X-Rayline-Executed-Action-Id"), response)
	}
	lookups := fixture.snapshot()
	if len(lookups) != 2 || lookups[1].Execution.Turn.Reconsult == nil || *lookups[1].Execution.Turn.Reconsult != "capacity" {
		t.Fatalf("lookups %d", len(lookups))
	}
	if fact := lookups[1].Execution.Targets[0].Capacity; fact == nil || fact.Fits || fact.Provenance != "provider_reported" {
		t.Fatalf("refusal fact %s", mustJSON(fact))
	}
	if localCalls != 1 || cloudCalls != 1 {
		t.Fatalf("local=%d cloud=%d", localCalls, cloudCalls)
	}
}
