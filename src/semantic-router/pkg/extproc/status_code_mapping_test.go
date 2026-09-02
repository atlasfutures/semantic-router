package extproc

import (
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

// statusMappers are the three copies of one HTTP-status-to-Envoy-enum mapping.
// A caller cannot see which one built its response, so they must agree.
var statusMappers = map[string]func(int) typev3.StatusCode{
	"statusCodeToImmediateResponseCode": statusCodeToImmediateResponseCode,
	"statusCodeToEnum":                  statusCodeToEnum,
	"statusCodeToEnumForResponseAPI":    statusCodeToEnumForResponseAPI,
}

// failureStatuses are statuses the router already raises or names, none of
// which is a success. 499 is raised on a client cancel
// (processor_req_body_prepare.go:55). 409, 504 and the rest are named by
// immediateProtocolError (processor_protocol_contract.go:361).
var failureStatuses = []int{0, 402, 408, 409, 410, 499, 501, 504, 507}

// TestImmediateResponseStatusNeverAnswers200ToAFailure is the ask.
//
// Each mapping falls back to StatusCode_OK for a status its table omits, so a
// failed request is answered HTTP 200. A client reads 200 as success and
// parses the error body as a result, which is worse than any wrong error code.
//
// 499 is the reachable case. The request-body path raises it when the client
// cancels, and the protocol error table already builds a request_canceled body
// for it, so that body ships today under a 200.
func TestImmediateResponseStatusNeverAnswers200ToAFailure(t *testing.T) {
	for name, mapper := range statusMappers {
		for _, statusCode := range failureStatuses {
			if got := mapper(statusCode); got == typev3.StatusCode_OK {
				t.Errorf(
					"%s(%d) = OK: a failed request is answered HTTP 200",
					name,
					statusCode,
				)
			}
		}
	}
}

// TestEveryStatusEnvoyDefinesSurvivesTheMapping is the second half of the ask.
//
// The Envoy enum's values are the HTTP status codes themselves, so no table is
// needed to carry a status Envoy defines. A table only decides which statuses
// are lost, and these three lose a different set each.
func TestEveryStatusEnvoyDefinesSurvivesTheMapping(t *testing.T) {
	for name, mapper := range statusMappers {
		for value := range typev3.StatusCode_name {
			if value == int32(typev3.StatusCode_Empty) {
				continue
			}
			want := typev3.StatusCode(value)
			if got := mapper(int(value)); got != want {
				t.Errorf("%s(%d) = %v, want %v", name, value, got, want)
			}
		}
	}
}

// TestUnmappedStatusesAnswer200Today pins the present behaviour so a change to
// it shows up in the diff rather than only in the test above.
func TestUnmappedStatusesAnswer200Today(t *testing.T) {
	for name, mapper := range statusMappers {
		for _, statusCode := range failureStatuses {
			if got := mapper(statusCode); got != typev3.StatusCode_OK {
				t.Errorf(
					"%s(%d) = %v: the recorded behaviour has changed",
					name,
					statusCode,
					got,
				)
			}
		}
	}
}
