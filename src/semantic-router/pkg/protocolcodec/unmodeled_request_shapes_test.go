package protocolcodec

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// This file demonstrates a routing-surface gap, not a codec defect.
//
// The router decodes every inbound body into the neutral request before it
// routes it, and the decode is strict: a field, a content block or an input
// item the neutral model does not name refuses the whole request with a
// protocol error, which the ExtProc boundary answers as HTTP 400 "invalid
// inference request" (pkg/extproc/processor_protocol_contract.go:119). That is
// correct for a translated request, because the target format cannot carry
// what the neutral model cannot hold. It is not correct for a request that the
// router only routes: the client and the provider both speak the same wire
// format, the router changes the destination and not the payload, and the two
// endpoints agree on a field the router happens not to model.
//
// The fixture holds fifteen request bodies taken from live Anthropic Messages
// and OpenAI Responses traffic. Each one is refused at ingress today. Each case
// records the exact code and message the codec answers, the field or block that
// causes it, and the group it belongs to:
//
//	A  the neutral model does not name the field, block or input item
//	B  the neutral model is narrower than the source API for a named field
//	C  the router validates payload bytes it never reads
//
// The only edit to the recorded bodies is that "model", and "max_tokens" for
// Anthropic, are set on every case, because both API surfaces require them.
//
// TestUnmodeledRequestShapesMatchRecordedDisposition pins what each shape does
// today: a case with a recorded code is still refused with exactly that code,
// and a case with no recorded code routes. A shape may only move between the
// two by editing the fixture, so a codec change can never move one silently.
// TestUnmodeledRequestShapesAreRoutable states the destination: every shape
// routes. It fails once per shape that has not been carried yet.
type unmodeledShapeFixture struct {
	SchemaVersion string                 `json:"schema_version"`
	Cases         []unmodeledRequestCase `json:"cases"`
}

type unmodeledRequestCase struct {
	ID      string                 `json:"id"`
	Surface llmprotocol.WireFormat `json:"surface"`
	Group   string                 `json:"group"`
	Trigger string                 `json:"trigger"`
	Note    string                 `json:"note"`
	// CurrentCode and CurrentMessage are the protocol error the codec answers
	// today. They are empty once the shape routes, and are asserted either way
	// so a codec change shows up here rather than silently moving a shape
	// between routable and refused.
	CurrentCode    string          `json:"current_code"`
	CurrentMessage string          `json:"current_message"`
	Request        json.RawMessage `json:"request"`
}

func readUnmodeledRequestShapes(t *testing.T) []unmodeledRequestCase {
	t.Helper()
	path := filepath.Join("testdata", "unmodeled_request_shapes.v1.json")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var fixture unmodeledShapeFixture
	if err := json.Unmarshal(body, &fixture); err != nil {
		t.Fatalf("invalid fixture %s: %v", path, err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatalf("fixture %s holds no cases", path)
	}
	return fixture.Cases
}

// TestUnmodeledRequestShapesMatchRecordedDisposition records the present
// behaviour of every shape. It passes at every commit; the fixture moves, not
// the assertion.
func TestUnmodeledRequestShapesMatchRecordedDisposition(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, testCase := range readUnmodeledRequestShapes(t) {
		t.Run(testCase.ID, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequestForMutation(testCase.Surface, testCase.Request)
			if testCase.CurrentCode == "" {
				if err != nil {
					t.Fatalf("%s is recorded as routable but was refused: %v", testCase.ID, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("%s now decodes; the recorded rejection %q is stale", testCase.ID, testCase.CurrentCode)
			}
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) {
				t.Fatalf("%s failed with a non-protocol error: %v", testCase.ID, err)
			}
			if protocolError.Code != testCase.CurrentCode {
				t.Fatalf("%s: code is %q, fixture records %q", testCase.ID, protocolError.Code, testCase.CurrentCode)
			}
			if protocolError.Message != testCase.CurrentMessage {
				t.Fatalf("%s: message is %q, fixture records %q", testCase.ID, protocolError.Message, testCase.CurrentMessage)
			}
		})
	}
}

// TestUnmodeledRequestShapesAreRoutable states the expected behaviour: a body
// the router only routes reaches its destination, and the parts the neutral
// model does not name survive the round trip rather than refusing the request.
//
// A case passes once its shape is carried rather than refused.
func TestUnmodeledRequestShapesAreRoutable(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, testCase := range readUnmodeledRequestShapes(t) {
		t.Run(testCase.ID, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequestForMutation(testCase.Surface, testCase.Request)
			if err == nil {
				return
			}
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) {
				t.Fatalf("%s failed with a non-protocol error: %v", testCase.ID, err)
			}
			t.Fatalf(
				"%s is refused at ingress and answers HTTP 400.\n"+
					"  surface: %s\n"+
					"  group:   %s\n"+
					"  cause:   %s\n"+
					"  code:    %s\n"+
					"  message: %s",
				testCase.ID, testCase.Surface, testCase.Group, testCase.Trigger,
				protocolError.Code, protocolError.Message,
			)
		})
	}
}
