package protocolcodec

import (
	"bytes"
	"testing"
)

const providerCreditsText = "can only afford 54095. To increase, visit https://openrouter.ai/settings/keys/0123abcd"

func pushInSmallChunks(t *testing.T, filter PublicStreamFilter, input []byte) []byte {
	t.Helper()
	var output []byte
	for offset := 0; offset < len(input); {
		end := offset + 1 + offset%13
		if end > len(input) {
			end = len(input)
		}
		chunk, err := filter.Push(input[offset:end])
		if err != nil {
			t.Fatal(err)
		}
		output = append(output, chunk...)
		offset = end
	}
	final, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	return append(output, final...)
}

func TestChatPublicStreamRestatesProviderErrorFrame(t *testing.T) {
	input := []byte(
		"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"c1\",\"error\":{\"code\":402,\"message\":\"" + providerCreditsText + "\",\"metadata\":{\"raw\":\"provider body\"}}," +
			"\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\n" +
			"data: [DONE]\n\n",
	)
	for _, filter := range []PublicStreamFilter{NewChatUsageStreamFilter(1 << 20), NewChatPassthroughStreamFilter(1 << 20)} {
		output := pushInSmallChunks(t, filter, input)
		for _, leaked := range []string{"openrouter.ai", "54095", "metadata", "provider body"} {
			if bytes.Contains(output, []byte(leaked)) {
				t.Fatalf("public stream leaked %q: %s", leaked, output)
			}
		}
		for _, kept := range []string{`"content":"hi"`, `"message":"model service unavailable"`, `"finish_reason":"error"`, "data: [DONE]"} {
			if !bytes.Contains(output, []byte(kept)) {
				t.Fatalf("public stream lost %q: %s", kept, output)
			}
		}
	}
}

func TestAnthropicPublicStreamPassesFramesAndRestatesErrorEvent(t *testing.T) {
	content := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"an \\\"error\\\" here\"}}\n\n"
	input := []byte(content +
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"prompt is too long; see https://example.com/limits\"},\"request_id\":\"req_1\"}\n\n")
	output := pushInSmallChunks(t, NewAnthropicPublicStreamFilter(1<<20), input)
	if !bytes.HasPrefix(output, []byte(content)) {
		t.Fatalf("non-error frames were changed: %s", output)
	}
	errorFrame := string(output[len(content):])
	want := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"prompt is too long; see [redacted]\"},\"request_id\":\"req_1\"}\n\n"
	if errorFrame != want {
		t.Fatalf("error frame = %q, want %q", errorFrame, want)
	}

	billing := []byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"billing_error\",\"message\":\"" + providerCreditsText + "\"}}\n\n")
	output = pushInSmallChunks(t, NewAnthropicPublicStreamFilter(1<<20), billing)
	if bytes.Contains(output, []byte("54095")) || !bytes.Contains(output, []byte("model service unavailable")) {
		t.Fatalf("account error leaked: %s", output)
	}
}
