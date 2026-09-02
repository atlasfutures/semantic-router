package extproc

import (
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

// statusCodeToImmediateResponseCode names the status of an ImmediateResponse.
// It is the same mapping as everywhere else, and deliberately not a second
// copy of it: the tables that used to live here and in the Response API path
// each omitted a different set of statuses, and every omission answered 200.
func statusCodeToImmediateResponseCode(statusCode int) typev3.StatusCode {
	return statusCodeToEnum(statusCode)
}
