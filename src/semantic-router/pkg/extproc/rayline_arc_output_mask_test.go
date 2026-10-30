package extproc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// outputMaskRouter is a basket shaped like the prod one: two arms capped at
// 128,000, one at 262,144, and one whose card declares no output limit.
func outputMaskRouter() (*OpenAIRouter, []config.ModelRef) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"claude": {MaxOutputTokens: 128000},
				"gpt":    {MaxOutputTokens: 128000},
				"large":  {MaxOutputTokens: 262144},
				"open":   {},
			},
		},
	}}
	return router, []config.ModelRef{{Model: "claude"}, {Model: "gpt"}, {Model: "large"}, {Model: "open"}}
}

func decodeOutputMaskRequest(t *testing.T, format llmprotocol.WireFormat, body string) *llmprotocol.Request {
	t.Helper()
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, []byte(body))
	if err != nil {
		t.Fatalf("decode %s: %v", format, err)
	}
	return &request
}

// Each wire format's output field reaches the mask through the codec's
// normalized allowance, and a request with no limit masks nothing.
func TestRaylineARCOverOutputArmsPerWireFormat(t *testing.T) {
	t.Parallel()
	router, refs := outputMaskRouter()
	above := []bool{true, true, false, false}
	cases := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		tokens int64
		want   []bool
	}{
		{"anthropic under every limit", llmprotocol.AnthropicMessagesV1,
			`{"model":"auto","max_tokens":64000,"messages":[{"role":"user","content":"hi"}]}`, 64000, nil},
		{"anthropic above claude and gpt", llmprotocol.AnthropicMessagesV1,
			`{"model":"auto","max_tokens":200000,"messages":[{"role":"user","content":"hi"}]}`, 200000, above},
		{"anthropic thinking budget inside max_tokens", llmprotocol.AnthropicMessagesV1,
			`{"model":"auto","max_tokens":200000,"thinking":{"type":"enabled","budget_tokens":150000},"messages":[{"role":"user","content":"hi"}]}`, 200000, above},
		{"chat max_tokens", llmprotocol.OpenAIChatV1,
			`{"model":"auto","max_tokens":200000,"messages":[{"role":"user","content":"hi"}]}`, 200000, above},
		{"chat max_completion_tokens", llmprotocol.OpenAIChatV1,
			`{"model":"auto","max_completion_tokens":200000,"messages":[{"role":"user","content":"hi"}]}`, 200000, above},
		{"chat under every limit", llmprotocol.OpenAIChatV1,
			`{"model":"auto","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`, 4096, nil},
		{"chat no limit", llmprotocol.OpenAIChatV1,
			`{"model":"auto","messages":[{"role":"user","content":"hi"}]}`, 0, nil},
		{"responses max_output_tokens", llmprotocol.OpenAIResponsesV1,
			`{"model":"auto","max_output_tokens":200000,"input":"hi"}`, 200000, above},
		{"responses no limit", llmprotocol.OpenAIResponsesV1,
			`{"model":"auto","input":"hi"}`, 0, nil},
		{"above every declared limit", llmprotocol.OpenAIResponsesV1,
			`{"model":"auto","max_output_tokens":300000,"input":"hi"}`, 300000, []bool{true, true, true, false}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := decodeOutputMaskRequest(t, tc.format, tc.body)
			tokens := requestedOutputTokens(request)
			if tokens != tc.tokens {
				t.Fatalf("requestedOutputTokens = %d, want %d", tokens, tc.tokens)
			}
			if got := router.overOutputArms(refs, tokens); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("overOutputArms(%d) = %v, want %v", tokens, got, tc.want)
			}
		})
	}
}

// A thinking budget counts as output when it is the larger number, a limit
// the router wrote is not the caller's, and a router-raised allowance reads
// back the caller's own.
func TestRaylineARCRequestedOutputTokensSources(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		request *llmprotocol.Request
		want    int64
	}{
		{"nil request", nil, 0},
		{"budget larger than the allowance", &llmprotocol.Request{
			Sampling:              llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(1000)},
			ReasoningBudgetTokens: llmprotocol.Int64(200000),
		}, 200000},
		{"budget without an allowance", &llmprotocol.Request{ReasoningBudgetTokens: llmprotocol.Int64(150000)}, 150000},
		{"router-set card default", &llmprotocol.Request{
			Sampling:                 llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(262144)},
			RouterSetMaxOutputTokens: true,
		}, 0},
		{"router-raised allowance", &llmprotocol.Request{
			Sampling:              llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(262144)},
			ClientMaxOutputTokens: llmprotocol.Int64(8000),
		}, 8000},
	}
	for _, tc := range cases {
		if got := requestedOutputTokens(tc.request); got != tc.want {
			t.Fatalf("%s: requestedOutputTokens = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// The mask reaches the selection context from the neutral request, and the
// request context keeps it for the selection log.
func TestRaylineARCSelectionContextCarriesTheOutputMask(t *testing.T) {
	t.Parallel()
	router, refs := outputMaskRouter()
	algorithm := &config.AlgorithmConfig{
		Type: config.RaylineARCAlgorithmType,
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode: config.RaylineARCEpisodeConfig{IDHeader: "x-rayline-session"},
		},
	}
	request := testNeutralRequest("auto", "write a long report")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(200000)
	requestContext := &RequestContext{Headers: map[string]string{}, SemanticRequest: request}
	selectionContext := router.buildRaylineARCSelectionContext(algorithm, requestContext, refs, raylineARCEpisodeEphemeral)
	if selectionContext.PreparationFailure != "" {
		t.Fatalf("preparation failed: %s", selectionContext.PreparationFailure)
	}
	want := []bool{true, true, false, false}
	if selectionContext.RequestedOutputTokens != 200000 || !reflect.DeepEqual(selectionContext.OverOutputArms, want) {
		t.Fatalf("output = %d over %v, want 200000 and %v", selectionContext.RequestedOutputTokens, selectionContext.OverOutputArms, want)
	}
	if !reflect.DeepEqual(requestContext.RaylineARCOverOutputArms, want) {
		t.Fatalf("request context mask = %v, want %v", requestContext.RaylineARCOverOutputArms, want)
	}
}

func outputSelectionContext(state *raylinearc.EpisodeState, overOutputArms []bool) *selection.SelectionContext {
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.RequestedOutputTokens = 200000
	selectionContext.RaylineARC.OverOutputArms = overOutputArms
	return selectionContext
}

// An over-output arm is masked where it stands, like an over-context one.
func TestRaylineARCSelectorExcludesArmsUnderTheRequestedOutput(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())
	if _, err := selector.Select(context.Background(), outputSelectionContext(state, []bool{true, false})); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{true, false}) {
		t.Fatalf("scorer exclusion = %v, want the over-output arm excluded", scorer.excluded)
	}
}

// A turn above every arm's output limit fails closed before the encoder, with
// its own class, as the same 503 the over-context refusal answers and a
// message that names the cause.
func TestRaylineARCSelectorRefusesOutputAboveEveryArm(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	encoder := visionEncoder()
	selector := armedVisionSelector(visionScorer(), encoder)
	_, err = selector.Select(context.Background(), outputSelectionContext(state, []bool{true, true}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != arcFailureNoOutputArm {
		t.Fatalf("Select() error = %v, want class %s", err, arcFailureNoOutputArm)
	}
	if encoder.calls != 0 {
		t.Fatalf("encoder calls = %d, want 0", encoder.calls)
	}
	if failure.contended() {
		t.Fatalf("%s must answer 503, not 429", arcFailureNoOutputArm)
	}
	response := (&OpenAIRouter{}).authoritativeSelectionFailureResponse(
		&modelSelectionFailure{algorithm: "rayline_arc", class: arcFailureNoOutputArm}, &RequestContext{},
	)
	if response == nil {
		t.Fatal("no_output_arm produced no admission response")
	}
	immediate := response.GetImmediateResponse()
	if status := int(immediate.GetStatus().GetCode()); status != http.StatusServiceUnavailable {
		t.Fatalf("no_output_arm status = %d, want %d", status, http.StatusServiceUnavailable)
	}
	if !strings.Contains(string(immediate.GetBody()), "max_tokens exceeds every arm's output limit") {
		t.Fatalf("no_output_arm body = %s, want the output-limit message", immediate.GetBody())
	}
}

// The selection log carries the requested output and the masked ordinals,
// and a turn that stated no limit carries neither.
func TestRaylineARCSelectionLogCarriesTheOutputMask(t *testing.T) {
	logs := captureLogs(t)
	trace := &selection.RaylineARCTrace{SelectedArm: 2}
	observeRaylineARCSelection(&RequestContext{
		RequestID:                       "r",
		RaylineARCRequestedOutputTokens: 200000,
		RaylineARCOverOutputArms:        []bool{true, true, false, false},
	}, trace)
	fields := findLogEvent(t, logs, "rayline_arc_selection")
	if fields["requested_output_tokens"] != int64(200000) {
		t.Fatalf("requested_output_tokens = %#v, want 200000", fields["requested_output_tokens"])
	}
	if got := fields["over_output_arms_count"]; got != int64(2) {
		t.Fatalf("over_output_arms_count = %#v, want 2", got)
	}
	if got := fmt.Sprint(fields["over_output_arms"]); got != "[0 1]" {
		t.Fatalf("over_output_arms = %s, want [0 1]", got)
	}
	logs.TakeAll()
	observeRaylineARCSelection(&RequestContext{RequestID: "r"}, trace)
	fields = findLogEvent(t, logs, "rayline_arc_selection")
	if _, present := fields["requested_output_tokens"]; present {
		t.Fatalf("requested_output_tokens present without a limit: %v", fields)
	}
}

// The refusal keeps its message through the client's final encoding, in
// every wire format, and stays a 503.
func TestRaylineARCNoOutputArmMessageSurvivesClientEncoding(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		router := &OpenAIRouter{}
		ctx := routingTestContext(format, testNeutralRequest("auto", "hi"))
		response := router.authoritativeSelectionFailureResponse(
			&modelSelectionFailure{algorithm: "rayline_arc", class: arcFailureNoOutputArm}, ctx,
		)
		encoded := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse()
		if status := int(encoded.GetStatus().GetCode()); status != http.StatusServiceUnavailable {
			t.Fatalf("%s: status = %d, want 503", format, status)
		}
		if !ctx.ImmediateResponseEncoded || !strings.Contains(string(encoded.GetBody()), "max_tokens exceeds every arm's output limit") {
			t.Fatalf("%s: encoded body = %s, want the output-limit message", format, encoded.GetBody())
		}
	}
}
