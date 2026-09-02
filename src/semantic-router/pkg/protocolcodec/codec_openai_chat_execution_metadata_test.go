package protocolcodec

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const vLLMChatResponseExtensionsFixture = `{
  "id":"chatcmpl-1","object":"chat.completion","created":7,
  "model":"provider-model","service_tier":"default","system_fingerprint":"fp_1",
  "prompt_logprobs":null,"prompt_token_ids":[11,12],"prompt_text":null,"kv_transfer_params":{},
  "ec_transfer_params":null,"metrics":null,
  "choices":[{
    "index":0,"finish_reason":"stop","stop_reason":106,"token_ids":[13],"logprobs":null,
    "routed_experts":null,
    "message":{"role":"assistant","content":"OK","refusal":null,"annotations":null,
      "audio":null,"function_call":null,"tool_calls":[],"reasoning":null}
  }],
  "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3,
    "prompt_tokens_details":{"cached_tokens":1,"audio_tokens":0},
    "completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,
      "reasoning_tokens":0,"rejected_prediction_tokens":0}}
}`

const legacyVLLMChatResponseExtensionsFixture = `{
  "id":"chatcmpl-legacy","object":"chat.completion","created":7,"model":"provider-model",
  "do_remote_decode":false,"do_remote_prefill":false,"remote_block_ids":null,
  "remote_engine_id":"","remote_host":"","remote_port":0,
  "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"OK"}}],
  "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
}`

func TestOpenAIChatResponseAcceptsClosedVLLMExecutionMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	translated, translateErr := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		[]byte(vLLMChatResponseExtensionsFixture),
		func(response *llmprotocol.Response) error {
			response.Model = "public-model"
			return nil
		},
	)
	if translateErr != nil {
		t.Fatalf("TranslateResponse() error = %v", translateErr)
	}
	if translated.Response.Model != "public-model" || translated.Response.Usage.Total.Value == nil ||
		*translated.Response.Usage.Total.Value != 3 {
		t.Fatalf("translated response = %+v", translated.Response)
	}
}

func TestOpenAIChatResponseAcceptsLegacyVLLMKVTransferMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	translated, translateErr := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		[]byte(legacyVLLMChatResponseExtensionsFixture),
		nil,
	)
	if translateErr != nil {
		t.Fatalf("TranslateResponse() error = %v", translateErr)
	}
	if translated.Response.Model != "provider-model" || len(translated.Response.Output) != 1 {
		t.Fatalf("translated response = %+v", translated.Response)
	}
	assertDiagnosticFields(t, translated.Diagnostics, "kv_transfer")
}

func TestOpenAIChatResponseAcceptsProviderTokenizedToolArguments(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-tool","object":"chat.completion","created":7,"model":"provider-model",
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
			"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}","TokenizedArguments":["{","}"]}}]}}],
		"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
	}`)
	translated, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		raw,
		nil,
	)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if translated.Response.StopReason != llmprotocol.StopToolCall || len(translated.Response.Output) != 1 {
		t.Fatalf("translated response = %+v", translated.Response)
	}
	assertDiagnosticFields(t, translated.Diagnostics, "choices.message.tool_calls.function.TokenizedArguments")
}

func TestOpenAIChatRequestRejectsProviderTokenizedToolArguments(t *testing.T) {
	raw := []byte(`{
		"model":"client-model","messages":[{"role":"user","content":"lookup"},{"role":"assistant","tool_calls":[{
			"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}","TokenizedArguments":["{","}"]}
		}]}]
	}`)
	_, err := NewBuiltinEngine().TranslateRequest(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		raw,
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported_messages_tool_calls_function_tokenized_arguments") {
		t.Fatalf("TranslateRequest() error = %v", err)
	}
}

func TestOpenAIChatResponseExtensionEnvelopeRoundTripAndUnknownDrop(t *testing.T) {
	engine := NewBuiltinEngine()
	raw := []byte(vLLMChatResponseExtensionsFixture)
	translated, translateErr := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, raw, nil)
	if translateErr != nil {
		t.Fatalf("TranslateResponse() error = %v", translateErr)
	}
	if !bytes.Equal(translated.Body, raw) {
		t.Fatal("same-format response envelope was not replayed byte-for-byte")
	}

	// A member the provider adds after this contract was written no longer
	// fails the response. It is dropped, and it does not reach the client.
	future := bytes.Replace(raw, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{},"future_field":true`), 1)
	dropped, translateErr := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		future,
		func(*llmprotocol.Response) error { return nil },
	)
	if translateErr != nil {
		t.Fatalf("future field was refused: %v", translateErr)
	}
	if bytes.Contains(dropped.Body, []byte("future_field")) {
		t.Fatalf("dropped body still carries the unknown member: %s", dropped.Body)
	}
}

func TestOpenAIChatResponseRejectsUnsupportedNonNullOutputExtensions(t *testing.T) {
	engine := NewBuiltinEngine()
	for field, replacement := range map[string]string{
		"audio":         `"audio":{"id":"audio_1","data":"AA==","expires_at":9,"transcript":"OK"}`,
		"function_call": `"function_call":{"name":"legacy","arguments":"{}"}`,
	} {
		t.Run(field, func(t *testing.T) {
			raw := strings.Replace(vLLMChatResponseExtensionsFixture, `"`+field+`":null`, replacement, 1)
			_, translateErr := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1,
				llmprotocol.OpenAIChatV1,
				[]byte(raw),
				func(*llmprotocol.Response) error { return nil },
			)
			if translateErr == nil || !strings.Contains(translateErr.Error(), "unsupported_upstream") {
				t.Fatalf("non-null %s error = %v", field, translateErr)
			}
		})
	}
}

func TestOpenAIChatResponseRejectsInvalidClosedExecutionMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := map[string]string{
		"non_null_prompt_logprobs": strings.Replace(vLLMChatResponseExtensionsFixture, `"prompt_logprobs":null`, `"prompt_logprobs":[]`, 1),
		"non_null_prompt_text":     strings.Replace(vLLMChatResponseExtensionsFixture, `"prompt_text":null`, `"prompt_text":"private prompt"`, 1),
		"non_null_ec_transfer":     strings.Replace(vLLMChatResponseExtensionsFixture, `"ec_transfer_params":null`, `"ec_transfer_params":{}`, 1),
		"non_null_metrics":         strings.Replace(vLLMChatResponseExtensionsFixture, `"metrics":null`, `"metrics":{}`, 1),
		"non_null_routed_experts":  strings.Replace(vLLMChatResponseExtensionsFixture, `"routed_experts":null`, `"routed_experts":[]`, 1),
		"unknown_service_tier":     strings.Replace(vLLMChatResponseExtensionsFixture, `"service_tier":"default"`, `"service_tier":"future"`, 1),
		"non_scalar_stop_reason":   strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":true`, 1),
		"fractional_stop_reason":   strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":1.5`, 1),
		"exponent_stop_reason":     strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":1e3`, 1),
		"oversized_fingerprint": strings.Replace(
			vLLMChatResponseExtensionsFixture,
			`"system_fingerprint":"fp_1"`,
			`"system_fingerprint":"`+strings.Repeat("x", 257)+`"`,
			1,
		),
	}
	for name, raw := range tests {
		t.Run(name, func(t *testing.T) {
			_, translateErr := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1,
				llmprotocol.OpenAIChatV1,
				[]byte(raw),
				func(*llmprotocol.Response) error { return nil },
			)
			if translateErr == nil {
				t.Fatal("invalid closed metadata was accepted")
			}
		})
	}
}

// A nested member the contract does not name is dropped at whatever depth the
// provider puts it. The closed shapes above stay closed for values.
func TestOpenAIChatResponseDropsNestedUnknownExecutionMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, raw := range map[string]string{
		"nested_kv_field": strings.Replace(vLLMChatResponseExtensionsFixture, `"kv_transfer_params":{}`, `"kv_transfer_params":{"future":true}`, 1),
		"nested_usage_field": strings.Replace(
			vLLMChatResponseExtensionsFixture,
			`"prompt_tokens_details":{"cached_tokens":1,"audio_tokens":0}`,
			`"prompt_tokens_details":{"cached_tokens":1,"audio_tokens":0,"future":1}`,
			1,
		),
	} {
		t.Run(name, func(t *testing.T) {
			dropped, translateErr := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1,
				llmprotocol.OpenAIChatV1,
				[]byte(raw),
				func(*llmprotocol.Response) error { return nil },
			)
			if translateErr != nil {
				t.Fatalf("nested unknown member was refused: %v", translateErr)
			}
			if bytes.Contains(dropped.Body, []byte(`"future"`)) {
				t.Fatalf("dropped body still carries the unknown member: %s", dropped.Body)
			}
		})
	}
}

func TestOpenAIChatStreamAcceptsClosedVLLMExecutionMetadataAndDropsFutureFields(t *testing.T) {
	engine := NewBuiltinEngine()
	context := llmprotocol.StreamContext{PublicModel: "public-model", ProviderModel: "provider-model"}
	stream, streamErr := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if streamErr != nil {
		t.Fatal(streamErr)
	}
	frame := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7," +
		"\"model\":\"provider-model\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp_1\"," +
		"\"prompt_logprobs\":null,\"prompt_token_ids\":[11],\"prompt_text\":null,\"kv_transfer_params\":{}," +
		"\"ec_transfer_params\":null,\"metrics\":null," +
		"\"choices\":[{\"index\":0,\"finish_reason\":null,\"stop_reason\":null,\"token_ids\":[12]," +
		"\"logprobs\":null,\"routed_experts\":null,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\",\"audio\":null,\"function_call\":null}}]}\n\n")
	frames, events, _, pushErr := stream.Push(frame)
	if pushErr != nil {
		t.Fatalf("Push() error = %v", pushErr)
	}
	for _, event := range events {
		if event.Model != "public-model" {
			t.Fatalf("event model = %q, want public-model", event.Model)
		}
	}
	if bytes.Contains(bytes.Join(frames, nil), []byte("provider-model")) ||
		!bytes.Contains(bytes.Join(frames, nil), []byte("public-model")) {
		t.Fatalf("rewritten frames = %q", bytes.Join(frames, nil))
	}

	future := bytes.Replace(frame, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{},"future_field":true`), 1)
	assertChatStreamAcceptedWithoutField(t, engine, context, future, "future_field")
	sameModelContext := llmprotocol.StreamContext{PublicModel: "same-model", ProviderModel: "same-model"}
	assertChatStreamAcceptedWithoutField(t, engine, sameModelContext, future, "future_field")
	// This synthetic frame carries no terminal event, so accumulating it fails
	// on its own account. The point is that the unknown member no longer
	// changes the outcome: both bodies now fail the same way.
	baseline := decodeChatStreamError(t, engine, frame, sameModelContext)
	withFuture := decodeChatStreamError(t, engine, future, sameModelContext)
	if fmt.Sprint(withFuture) != fmt.Sprint(baseline) {
		t.Fatalf("DecodeResponseStream error = %v, want the same failure as the body without the unknown member (%v)", withFuture, baseline)
	}

	nestedFuture := bytes.Replace(frame, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{"future":true}`), 1)
	assertChatStreamAcceptedWithoutField(t, engine, context, nestedFuture, `"future"`)
	withLogprobs := bytes.Replace(frame, []byte(`"logprobs":null`), []byte(`"logprobs":{"content":[]}`), 1)
	assertChatStreamRejected(t, engine, context, withLogprobs, "unsupported_upstream_stream_logprobs")
}

func TestOpenAIChatStreamAcceptsProviderTokenizedToolArguments(t *testing.T) {
	engine := NewBuiltinEngine()
	context := llmprotocol.StreamContext{PublicModel: "public-model", ProviderModel: "provider-model"}
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, context)
	if err != nil {
		t.Fatal(err)
	}
	frame := []byte("data: {\"id\":\"chatcmpl-tool\",\"object\":\"chat.completion.chunk\",\"created\":7," +
		"\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"finish_reason\":null,\"delta\":{" +
		"\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\"," +
		"\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\",\"TokenizedArguments\":[\"{\",\"}\"]}}]}}]}\n\n")
	_, events, diagnostics, pushErr := stream.Push(frame)
	if pushErr != nil {
		t.Fatalf("Push() error = %v", pushErr)
	}
	if len(events) == 0 {
		t.Fatal("Push() emitted no neutral events")
	}
	assertDiagnosticFields(t, diagnostics, "choices.delta.tool_calls.function.TokenizedArguments")
}

func decodeChatStreamError(
	t *testing.T,
	engine *Engine,
	frame []byte,
	context llmprotocol.StreamContext,
) error {
	t.Helper()
	_, _, err := engine.DecodeResponseStream(
		llmprotocol.OpenAIChatV1,
		append(append([]byte(nil), frame...), []byte("data: [DONE]\n\n")...),
		context,
	)
	return err
}

// assertChatStreamAcceptedWithoutField pins the tolerant half of the stream
// contract: the frame is accepted and the member the contract does not name is
// gone from what the client receives.
func assertChatStreamAcceptedWithoutField(
	t *testing.T,
	engine *Engine,
	context llmprotocol.StreamContext,
	payload []byte,
	field string,
) {
	t.Helper()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, pushErr := stream.Push(payload)
	if pushErr != nil {
		t.Fatalf("stream error = %v, want the unknown member dropped", pushErr)
	}
	if bytes.Contains(bytes.Join(frames, nil), []byte(field)) {
		t.Fatalf("rewritten frames still carry %s: %q", field, bytes.Join(frames, nil))
	}
}

func assertChatStreamRejected(
	t *testing.T,
	engine *Engine,
	context llmprotocol.StreamContext,
	payload []byte,
	errorCode string,
) {
	t.Helper()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, pushErr := stream.Push(payload); pushErr == nil || !strings.Contains(pushErr.Error(), errorCode) {
		t.Fatalf("stream error = %v, want %s", pushErr, errorCode)
	}
}
