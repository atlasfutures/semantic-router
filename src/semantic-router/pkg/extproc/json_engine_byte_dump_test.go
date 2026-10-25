//go:build !windows && cgo

package extproc

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// jsonEngineByteDumpEnv names a directory that receives, for every captured
// client body in the corpora below, the bytes the router writes: the
// provider-bound body (or the refusal it answers), and the policy decide
// request. Two runs, one per encoding/json engine (Go 1.25's v1 and Go 1.27's
// v2-backed v1), are diffed by tools/scripts/json-v2-byte-diff.sh: the engine
// change must not move a byte. Without the variable the test is skipped.
const jsonEngineByteDumpEnv = "VSR_JSON_BYTE_DUMP_OUT"

func TestJSONEngineByteDump(t *testing.T) {
	out := os.Getenv(jsonEngineByteDumpEnv)
	if out == "" {
		t.Skip("set " + jsonEngineByteDumpEnv + " to dump router output bytes")
	}
	corpus := jsonEngineCorpus(t)
	if len(corpus) == 0 {
		t.Fatal("no corpus bodies found")
	}
	dispatched := 0
	defer func() { t.Logf("%d of %d requests reached the policy decide", dispatched, 4*len(corpus)) }()
	for _, glmFormat := range []string{"anthropic", "openai"} {
		router, fake := v5RouterForCaptures(t)
		if glmFormat != "anthropic" {
			router, fake = v5RouterWith(t, glmFormat, func(config string) string {
				const card = "    - name: glm\n      modality: text\n"
				return strings.Replace(config, card, card+"      capabilities: [tool_result_images]\n", 1)
			})
		}
		for _, action := range []struct{ name, id string }{{"glm-up", v5GLMUp}, {"opus", v5OpusAction}} {
			for _, entry := range corpus {
				name := filepath.Join("glm-"+glmFormat, action.name, entry.name)
				episode := strings.NewReplacer("/", "-", ".", "-").Replace(name)
				before := len(fake.receivedBodies())
				written, client := jsonEngineTurn(t, router, fake, action.id, episode, entry.path, entry.body, "ok")
				writeJSONEngineDump(t, out, name+".provider", written)
				if client != nil {
					writeJSONEngineDump(t, out, name+".client", client)
					// The same turn on its own episode, answered by a
					// provider error instead.
					_, failed := jsonEngineTurn(t, router, fake, action.id, episode+"-error", entry.path, entry.body, "error")
					writeJSONEngineDump(t, out, name+".client-error", failed)
				}
				if bodies := fake.receivedBodies(); len(bodies) > before {
					writeJSONEngineDump(t, out, name+".decide", bodies[len(bodies)-1])
				}
				// A capture the router refuses as an unknown model never
				// reached the dispatch path the dump is for.
				if len(fake.receivedBodies()) == before && strings.Contains(string(written), "is not available") {
					t.Errorf("%s: the capture's model was not routed on the auto alias: %.200s", name, written)
				}
				if len(fake.receivedBodies()) > before {
					dispatched++
				}
			}
		}
	}
}

var jsonEngineModelMember = regexp.MustCompile(`"model"\s*:\s*"[^"]*"`)

type jsonEngineCorpusEntry struct {
	name, path string
	body       []byte
}

// jsonEngineCorpus is every captured client request body the repo keeps: the
// thinking-control corpus (Claude Code, Chat and Responses captures) and the
// protocol codec's request goldens. Each is routed on the auto alias.
func jsonEngineCorpus(t *testing.T) []jsonEngineCorpusEntry {
	t.Helper()
	var corpus []jsonEngineCorpusEntry
	add := func(name, path, file string) {
		body, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		// The first model member, however the capture is formatted, names the
		// auto alias.
		if location := jsonEngineModelMember.FindIndex(body); location != nil {
			body = append(append(append([]byte{}, body[:location[0]]...), `"model":"auto"`...), body[location[1]:]...)
		}
		corpus = append(corpus, jsonEngineCorpusEntry{name: name, path: path, body: body})
	}
	paths := map[string]string{"messages": "/v1/messages", "chat": "/v1/chat/completions", "responses": "/v1/responses"}
	for format, path := range paths {
		files, _ := filepath.Glob(filepath.Join(v5Golden, format, "*", "client-*.json"))
		for _, file := range files {
			add(filepath.Join("thinkingcontrol", format, filepath.Base(filepath.Dir(file)), filepath.Base(file)), path, file)
		}
	}
	codec := "../protocolcodec/testdata/golden/request"
	files, _ := filepath.Glob(filepath.Join(codec, "*-in.json"))
	for _, file := range files {
		base := filepath.Base(file)
		path := ""
		switch {
		case strings.Contains(base, "-chat-"):
			path = paths["chat"]
		case strings.Contains(base, "-responses-"):
			path = paths["responses"]
		case strings.Contains(base, "-anthropic-"):
			path = paths["messages"]
		default:
			continue
		}
		add(filepath.Join("protocolcodec", base), path, file)
	}
	sort.Slice(corpus, func(i, j int) bool { return corpus[i].name < corpus[j].name })
	return corpus
}

// jsonEngineTurn runs one request through the request phases and returns
// what the router writes: the provider body, or an immediate response's
// status and body.
func jsonEngineTurn(t *testing.T, router *OpenAIRouter, fake *fakePolicyService, action, episode, path string, client []byte, reply string) ([]byte, []byte) {
	t.Helper()
	fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		for _, offered := range request.Selection.AvailableActionIDs {
			if offered == action {
				return action
			}
		}
		if len(request.Selection.AvailableActionIDs) > 0 {
			return request.Selection.AvailableActionIDs[0]
		}
		return action
	})
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "json-engine-" + episode,
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"}, {Key: ":path", Value: path},
			{Key: "content-type", Value: "application/json"}, {Key: "x-rayline-session", Value: episode},
		}},
	}}
	response, err := router.handleRequestHeaders(headers, ctx)
	if err != nil {
		return []byte("headers error: " + err.Error()), nil
	}
	// What the client receives: processRequestHeaders and processRequestBody
	// encode an immediate response in the client's protocol before sending.
	response = router.encodeImmediateResponseForClient(response, ctx)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return []byte(fmt.Sprintf("headers immediate %d\n%s", immediate.GetStatus().GetCode(), immediate.GetBody())), nil
	}
	response, err = router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: client, EndOfStream: true},
	}, ctx)
	if err != nil {
		var ok bool
		if response, ok = router.processBodyRoutingError(err, ctx); !ok {
			return []byte("body error: " + err.Error()), nil
		}
	}
	response = router.encodeImmediateResponseForClient(response, ctx)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return []byte(fmt.Sprintf("immediate %d\n%s", immediate.GetStatus().GetCode(), immediate.GetBody())), nil
	}
	written := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	return written, jsonEngineResponse(router, ctx, reply)
}

// jsonEngineResponse answers a dispatched turn with a provider reply in the
// provider's wire format -- a success, buffered or streamed as the client
// asked, or a rate-limit error -- and returns every body the router sends the
// client for it.
func jsonEngineResponse(router *OpenAIRouter, ctx *RequestContext, reply string) []byte {
	var out bytes.Buffer
	switch ctx.TargetFormat {
	case llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1:
	default:
		fmt.Fprintf(&out, "no fixture for provider format %q\n", ctx.TargetFormat)
		return out.Bytes()
	}
	headers, body := jsonEngineHeaders("200", "application/json"), extProcResponseFixture(ctx.TargetFormat)
	switch {
	case reply == "error":
		headers, body = jsonEngineHeaders("429", "application/json"), extProcTransportErrorFixture(ctx.TargetFormat)
	case ctx.ExpectStreamingResponse:
		headers, body = jsonEngineHeaders("200", "text/event-stream"), extProcStreamFixture(ctx.TargetFormat)
	}
	response, err := router.handleResponseHeaders(headers, ctx)
	if err != nil {
		fmt.Fprintf(&out, "response headers error: %v\n", err)
		return out.Bytes()
	}
	writeJSONEngineResponse(&out, "headers", response)
	stream := NewMockStream(nil)
	if err := router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
	}, ctx); err != nil {
		fmt.Fprintf(&out, "response body error: %v\n", err)
	}
	for _, sent := range stream.Responses {
		writeJSONEngineResponse(&out, "body", sent)
	}
	finalizeSelectionProcessTerminal(ctx)
	return out.Bytes()
}

func jsonEngineHeaders(status, contentType string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: status}, {Key: "content-type", Value: contentType},
		}},
	}}
}

// writeJSONEngineResponse writes the bodies a processing response carries:
// an immediate response, a body mutation, or a streamed chunk.
func writeJSONEngineResponse(out *bytes.Buffer, phase string, response *ext_proc.ProcessingResponse) {
	if response == nil {
		return
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		fmt.Fprintf(out, "--- %s immediate %d\n%s\n", phase, immediate.GetStatus().GetCode(), immediate.GetBody())
		return
	}
	common := response.GetResponseBody().GetResponse()
	if common == nil {
		common = response.GetResponseHeaders().GetResponse()
	}
	mutation := common.GetBodyMutation()
	if mutation == nil {
		return
	}
	if streamed := mutation.GetStreamedResponse(); streamed != nil {
		fmt.Fprintf(out, "--- %s streamed eos=%v\n%s\n", phase, streamed.GetEndOfStream(), streamed.GetBody())
		return
	}
	fmt.Fprintf(out, "--- %s body clear=%v\n%s\n", phase, mutation.GetClearBody(), mutation.GetBody())
}

func writeJSONEngineDump(t *testing.T, root, name string, body []byte) {
	t.Helper()
	path := filepath.Join(root, "extproc", name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
}
