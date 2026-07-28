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
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestEncoderClientUsesStructuredHashedRequestAndValidatesReadiness(t *testing.T) {
	rawEpisodeID := "private-episode-id"
	var observed map[string]any
	server := httptest.NewServer(captureEncoderRequest(
		t,
		rawEpisodeID,
		&observed,
	))
	defer server.Close()

	client, err := NewEncoderClient(validEncoderClientConfig(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Encode(
		context.Background(),
		HashEpisodeID(rawEpisodeID),
		[]Turn{{Role: "user", Text: "hello"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	assertValidEncoderResult(t, result)
	assertStructuredEncoderRequest(t, observed, rawEpisodeID)
}

func captureEncoderRequest(
	t *testing.T,
	rawEpisodeID string,
	observed *map[string]any,
) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/pooling" {
			t.Errorf("path = %q, want /pooling", request.URL.Path)
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(body), rawEpisodeID) {
			t.Fatal("raw episode ID crossed the encoder boundary")
		}
		if err := json.Unmarshal(body, observed); err != nil {
			t.Fatal(err)
		}
		writeEncoderResponse(t, writer, nil)
	})
}

func assertValidEncoderResult(t *testing.T, result *EncoderResult) {
	t.Helper()
	if len(result.Embedding) != 1024 || result.SerializedTokens != 16 {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func assertStructuredEncoderRequest(
	t *testing.T,
	observed map[string]any,
	rawEpisodeID string,
) {
	t.Helper()
	if observed["task"] != "plugin" {
		t.Fatalf("task = %#v, want plugin", observed["task"])
	}
	data, ok := observed["data"].(map[string]any)
	if !ok {
		t.Fatalf("missing request data: %#v", observed)
	}
	if data["episode_id_hash"] != HashEpisodeID(rawEpisodeID) {
		t.Fatalf("episode hash = %#v", data["episode_id_hash"])
	}
	if data["serializer_version"] != SerializationName {
		t.Fatalf("serializer = %#v", data["serializer_version"])
	}
}

func TestEncoderClientRejectsContractDrift(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(map[string]any)
		class  EncoderFailureClass
	}{
		{
			name: "dimension",
			mutate: func(data map[string]any) {
				data["embedding"] = []float64{1}
			},
			class: EncoderFailureContract,
		},
		{
			name: "nonfinite",
			mutate: func(data map[string]any) {
				values := make([]float64, 1024)
				values[3] = math.MaxFloat64
				data["embedding"] = values
			},
			class: EncoderFailureContract,
		},
		{
			name: "build",
			mutate: func(data map[string]any) {
				data["engine_build_id"] = "vllm@wrong-build"
			},
			class: EncoderFailureContract,
		},
		{
			name: "tokenizer",
			mutate: func(data map[string]any) {
				data["tokenizer_sha256"] = strings.Repeat("b", 64)
			},
			class: EncoderFailureContract,
		},
		{
			name: "capability",
			mutate: func(data map[string]any) {
				data["pooling_capabilities"] = []string{"other"}
			},
			class: EncoderFailureContract,
		},
		{
			name: "count invariant",
			mutate: func(data map[string]any) {
				data["serialized_tokens"] = 100
				data["full_history_tokens"] = 10
				data["truncated_tokens"] = 0
			},
			class: EncoderFailureContract,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(
				writer http.ResponseWriter,
				_ *http.Request,
			) {
				writeEncoderResponse(t, writer, test.mutate)
			}))
			defer server.Close()
			client, err := NewEncoderClient(validEncoderClientConfig(server.URL))
			if err != nil {
				t.Fatal(err)
			}
			_, err = client.Encode(
				context.Background(),
				strings.Repeat("a", 64),
				[]Turn{{Role: "user", Text: "hello"}},
			)
			assertEncoderFailureClass(t, err, test.class)
		})
	}
}

func TestEncoderClientDoesNotRetryAfterHTTPResponse(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		_ *http.Request,
	) {
		calls.Add(1)
		writer.WriteHeader(http.StatusServiceUnavailable)
		_, _ = writer.Write([]byte("private upstream failure"))
	}))
	defer server.Close()
	cfg := validEncoderClientConfig(server.URL)
	cfg.MaxRetries = 3
	client, err := NewEncoderClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Encode(
		context.Background(),
		strings.Repeat("a", 64),
		[]Turn{{Role: "user", Text: "hello"}},
	)
	assertEncoderFailureClass(t, err, EncoderFailureStatus)
	if calls.Load() != 1 {
		t.Fatalf("calls = %d, want 1", calls.Load())
	}
	if strings.Contains(err.Error(), "private upstream failure") {
		t.Fatal("encoder error exposed the response body")
	}
}

func TestEncoderClientRetriesOnlyPreResponseTransportFailure(t *testing.T) {
	var calls atomic.Int32
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if calls.Add(1) == 1 {
			return nil, errors.New("connection failed before headers")
		}
		recorder := httptest.NewRecorder()
		writeEncoderResponse(t, recorder, nil)
		return recorder.Result(), nil
	})
	cfg := validEncoderClientConfig("http://encoder.test")
	cfg.MaxRetries = 1
	client, err := newEncoderClient(cfg, &http.Client{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Encode(
		context.Background(),
		strings.Repeat("a", 64),
		[]Turn{{Role: "user", Text: "hello"}},
	); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

func TestEncoderClientHonorsOneTotalRequestBudget(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(
		_ http.ResponseWriter,
		_ *http.Request,
	) {
		time.Sleep(100 * time.Millisecond)
	}))
	defer server.Close()
	cfg := validEncoderClientConfig(server.URL)
	cfg.ConnectTimeout = 5 * time.Millisecond
	cfg.TotalTimeout = 25 * time.Millisecond
	client, err := NewEncoderClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	_, err = client.Encode(
		context.Background(),
		strings.Repeat("a", 64),
		[]Turn{{Role: "user", Text: "hello"}},
	)
	assertEncoderFailureClass(t, err, EncoderFailureTimeout)
	if time.Since(started) > time.Second {
		t.Fatal("encoder total request budget was not enforced")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func validEncoderClientConfig(baseURL string) EncoderClientConfig {
	return EncoderClientConfig{
		BaseURL:               baseURL,
		Model:                 "Qwen/Qwen3.5-0.8B",
		ModelRevision:         "2fc06364715b967f1860aea9cf38778875588b17",
		TokenizerRevision:     "2fc06364715b967f1860aea9cf38778875588b17",
		TokenizerSHA256:       "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
		EOSTokenID:            248046,
		ExpectedBuildID:       "vllm@immutable-build",
		ExpectedPluginVersion: "rayline-arc-io@0.1.0",
		SerializerVersion:     SerializationName,
		RequiredCapabilities:  []string{"all_plugin_mean"},
		ConnectTimeout:        time.Second,
		TotalTimeout:          time.Second,
		MaxRetries:            0,
	}
}

func writeEncoderResponse(
	t *testing.T,
	writer http.ResponseWriter,
	mutate func(map[string]any),
) {
	t.Helper()
	embedding := make([]float64, 1024)
	embedding[0] = 1
	data := map[string]any{
		"embedding":            embedding,
		"serialized_tokens":    16,
		"full_history_tokens":  16,
		"truncated_tokens":     0,
		"cached_prefix_tokens": 0,
		"serializer_version":   SerializationName,
		"model":                "Qwen/Qwen3.5-0.8B",
		"model_revision":       "2fc06364715b967f1860aea9cf38778875588b17",
		"tokenizer_revision":   "2fc06364715b967f1860aea9cf38778875588b17",
		"tokenizer_sha256":     "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
		"eos_token_id":         248046,
		"engine_build_id":      "vllm@immutable-build",
		"io_plugin_version":    "rayline-arc-io@0.1.0",
		"pooling_capabilities": []string{"all_plugin_mean"},
	}
	if mutate != nil {
		mutate(data)
	}
	writer.Header().Set("content-type", "application/json")
	if err := json.NewEncoder(writer).Encode(map[string]any{"data": data}); err != nil {
		t.Fatal(err)
	}
}

func assertEncoderFailureClass(
	t *testing.T,
	err error,
	want EncoderFailureClass,
) {
	t.Helper()
	var failure *EncoderFailure
	if !errors.As(err, &failure) {
		t.Fatalf("error = %v, want EncoderFailure", err)
	}
	if failure.Class != want {
		t.Fatalf("class = %q, want %q", failure.Class, want)
	}
}
