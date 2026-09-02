package extproc

import (
	"fmt"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

// immediateStatusMappers are the three copies of one mapping. They must agree,
// because a caller cannot see which one built its response.
var immediateStatusMappers = map[string]func(int) typev3.StatusCode{
	"statusCodeToImmediateResponseCode": statusCodeToImmediateResponseCode,
	"statusCodeToEnum":                  statusCodeToEnum,
	"statusCodeToEnumForResponseAPI":    statusCodeToEnumForResponseAPI,
}

// TestImmediateResponseStatusNeverAnswers200ToAFailure is the defect.
//
// 200 is the one answer no failure may carry. A client reads it as success and
// parses the body as a result, so a mapping that falls back to OK turns a
// failed request into a silently wrong answer rather than a visible error.
//
// 499 is the reachable case: the router raises it on a client cancel
// (processor_req_body_prepare.go). The rest are statuses the router or its
// protocol-error table already name.
func TestImmediateResponseStatusNeverAnswers200ToAFailure(t *testing.T) {
	failures := []int{
		0,   // no status at all
		409, // named by immediateProtocolError
		410,
		418,
		499, // client cancel, raised by the request-body path
		504, // named by immediateProtocolError
		599,
	}
	for name, mapper := range immediateStatusMappers {
		for _, statusCode := range failures {
			t.Run(fmt.Sprintf("%s/%d", name, statusCode), func(t *testing.T) {
				if got := mapper(statusCode); got == typev3.StatusCode_OK {
					t.Fatalf(
						"%s(%d) = OK: a failed request was answered 200",
						name,
						statusCode,
					)
				}
			})
		}
	}
}

// TestImmediateResponseStatusPreservesEveryCodeEnvoyDefines pins the default.
//
// The Envoy enum's values are the HTTP codes themselves, so any status Envoy
// defines can be carried through unchanged. Nothing needs a lookup table, and
// a table is exactly what leaves a status unmapped.
func TestImmediateResponseStatusPreservesEveryCodeEnvoyDefines(t *testing.T) {
	for name, mapper := range immediateStatusMappers {
		for value := range typev3.StatusCode_name {
			if value == int32(typev3.StatusCode_Empty) {
				continue
			}
			statusCode := int(value)
			t.Run(fmt.Sprintf("%s/%d", name, statusCode), func(t *testing.T) {
				want := typev3.StatusCode(value)
				if got := mapper(statusCode); got != want {
					t.Fatalf("%s(%d) = %v, want %v", name, statusCode, got, want)
				}
			})
		}
	}
}

// TestImmediateResponseStatusDegradesUndefinedCodesWithinTheirClass pins what
// happens to a status Envoy has no enum value for. Envoy validates this field
// as defined_only and rejects 0, so such a status cannot be sent as itself. It
// must degrade to the nearest defined code of the same class, never across the
// 4xx/5xx line and never to a success.
func TestImmediateResponseStatusDegradesUndefinedCodesWithinTheirClass(t *testing.T) {
	tests := []struct {
		statusCode int
		want       typev3.StatusCode
	}{
		{statusCode: 499, want: typev3.StatusCode_BadRequest},
		{statusCode: 418, want: typev3.StatusCode_BadRequest},
		{statusCode: 599, want: typev3.StatusCode_BadGateway},
		{statusCode: 509, want: typev3.StatusCode_BadGateway},
		{statusCode: 0, want: typev3.StatusCode_InternalServerError},
		{statusCode: 42, want: typev3.StatusCode_InternalServerError},
		{statusCode: 700, want: typev3.StatusCode_InternalServerError},
	}
	for name, mapper := range immediateStatusMappers {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s/%d", name, tt.statusCode), func(t *testing.T) {
				if got := mapper(tt.statusCode); got != tt.want {
					t.Fatalf("%s(%d) = %v, want %v", name, tt.statusCode, got, tt.want)
				}
			})
		}
	}
}

// TestCanceledRequestAnswersAClientErrorNotSuccess is the same defect seen from
// the caller. The router raises 499 for a cancelled request and the protocol
// error table already carries a request_canceled body for it, so a 200 here
// ships an error body under a success status.
func TestCanceledRequestAnswersAClientErrorNotSuccess(t *testing.T) {
	router := &OpenAIRouter{}
	response := router.createErrorResponse(499, "request canceled")
	got := response.GetImmediateResponse().GetStatus().GetCode()
	if got == typev3.StatusCode_OK {
		t.Fatal("a cancelled request was answered 200 with an error body")
	}
	if got != typev3.StatusCode_BadRequest {
		t.Fatalf("cancelled request status = %v, want BadRequest", got)
	}
}
