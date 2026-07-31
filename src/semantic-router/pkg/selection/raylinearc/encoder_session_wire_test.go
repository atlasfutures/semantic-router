/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRetainedSessionClientUsesVersionedDirectWire(t *testing.T) {
	t.Parallel()
	var observed map[string]any
	server := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			if request.URL.Path != "/v1/rayline/arc/session/pooling" {
				t.Fatalf("path = %q", request.URL.Path)
			}
			if err := json.NewDecoder(request.Body).Decode(&observed); err != nil {
				t.Fatal(err)
			}
			writeSessionEncoderResponse(t, writer, nil)
		},
	))
	defer server.Close()

	config := validEncoderClientConfig(server.URL)
	config.ServingRung = encoderServingRungB
	config.RequiredCapabilities = []string{
		"chunked_causal_mean",
		"resumable_causal_mean",
	}
	config.RetainedSession = true
	client, err := NewEncoderClient(config)
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Encode(
		t.Context(),
		strings.Repeat("a", 64),
		[]Turn{{Role: "user", Text: "public synthetic input"}},
	)
	if err != nil {
		t.Fatal(err)
	}

	if observed["schema_version"] != encoderSessionRequestSchema {
		t.Fatalf("schema_version = %v", observed["schema_version"])
	}
	if _, ok := observed["task"]; ok {
		t.Fatal("retained-session request unexpectedly used plugin envelope")
	}
	if result.RetainedPrefixTokens != 12 || result.AppendedTokens != 4 {
		t.Fatalf("retained/appended = %d/%d", result.RetainedPrefixTokens, result.AppendedTokens)
	}
	if result.CachedPrefixTokens != 0 || result.SessionAction != "appended" || result.SessionRevision != 2 {
		t.Fatalf("unexpected session result: %+v", result)
	}
}

func TestRetainedSessionClientRejectsInvalidAccounting(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			writeSessionEncoderResponse(t, writer, func(data map[string]any) {
				data["appended_tokens"] = 5
			})
		},
	))
	defer server.Close()

	config := validEncoderClientConfig(server.URL)
	config.ServingRung = encoderServingRungB
	config.RequiredCapabilities = []string{
		"chunked_causal_mean",
		"resumable_causal_mean",
	}
	config.RetainedSession = true
	client, err := NewEncoderClient(config)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Encode(
		t.Context(),
		strings.Repeat("b", 64),
		[]Turn{{Role: "user", Text: "public synthetic input"}},
	)
	assertEncoderFailureClass(t, err, EncoderFailureContract)
}

func writeSessionEncoderResponse(
	t *testing.T,
	writer http.ResponseWriter,
	mutate func(map[string]any),
) {
	t.Helper()
	embedding := make([]float64, encoderDimension)
	embedding[0] = 1
	data := map[string]any{
		"schema_version":         encoderSessionResponseSchema,
		"embedding":              embedding,
		"serialized_tokens":      16,
		"full_history_tokens":    16,
		"truncated_tokens":       0,
		"retained_prefix_tokens": 12,
		"appended_tokens":        4,
		"session_action":         "appended",
		"session_revision":       2,
		"serializer_version":     SerializationName,
		"model":                  "Qwen/Qwen3.5-0.8B",
		"model_revision":         "2fc06364715b967f1860aea9cf38778875588b17",
		"tokenizer_revision":     "2fc06364715b967f1860aea9cf38778875588b17",
		"tokenizer_sha256":       EncoderTokenizerSHA256,
		"eos_token_id":           EncoderEOSTokenID,
		"engine_build_id":        "vllm@immutable-build",
		"io_plugin_version":      "rayline-arc-io@0.1.0",
		"pooling_capabilities": []string{
			"chunked_causal_mean",
			"resumable_causal_mean",
		},
	}
	if mutate != nil {
		mutate(data)
	}
	writer.Header().Set("content-type", "application/json")
	if err := json.NewEncoder(writer).Encode(data); err != nil {
		t.Fatal(err)
	}
}
