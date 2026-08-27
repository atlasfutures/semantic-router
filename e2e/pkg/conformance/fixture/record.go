package fixture

import (
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
)

// ObservedRequest is one raw inbound provider request, captured before anything is
// replayed. It is the evidence a case asserts the provider boundary against, so it
// keeps the body exactly as it arrived rather than a re-encoded form of it.
//
// Body is JSON-encoded as base64, which survives non-UTF-8 bytes intact, so an
// out-of-process reader of the observed endpoint sees the same bytes an in-process
// caller does.
type ObservedRequest struct {
	Method string `json:"method"`
	Path   string `json:"path"`
	// Headers are the request headers as net/http parsed them, under canonical
	// names. Go moves Host onto the request itself, so it is the one header a
	// provider expectation cannot name.
	Headers http.Header `json:"headers"`
	Body    []byte      `json:"body"`
	// Mismatch names why the request failed the script's expect block. It is empty
	// when the request matched and the script was replayed.
	Mismatch string `json:"mismatch,omitempty"`
}

func capture(r *http.Request, body []byte) ObservedRequest {
	return ObservedRequest{
		Method:  r.Method,
		Path:    r.URL.Path,
		Headers: r.Header.Clone(),
		Body:    body,
	}
}

// matchExpectation reports why observed fails the script's expect block, or "" when
// it matches. Every failing field is named at once so an author fixes one route,
// not three in sequence.
func matchExpectation(script *conformance.ReplayScript, observed ObservedRequest) string {
	if script == nil {
		return "no replay script is loaded"
	}

	var problems []string
	if script.Expect.Method != observed.Method {
		problems = append(problems, fmt.Sprintf("method %q, want %q", observed.Method, script.Expect.Method))
	}
	if script.Expect.Path != observed.Path {
		problems = append(problems, fmt.Sprintf("path %q, want %q", observed.Path, script.Expect.Path))
	}
	problems = append(problems, headerProblems(script.Expect.Headers, observed.Headers)...)

	if len(problems) == 0 {
		return ""
	}
	return "provider request does not match expect: " + strings.Join(problems, "; ")
}

// headerProblems compares only the headers the expectation names; an unnamed header
// is free. The result is sorted because map iteration order is not stable and a
// mismatch message is both compared in tests and read in failure output.
func headerProblems(want map[string]string, got http.Header) []string {
	var problems []string
	for name, value := range want {
		if actual := got.Get(name); actual != value {
			problems = append(problems, fmt.Sprintf("header %s = %q, want %q", name, actual, value))
		}
	}
	sort.Strings(problems)
	return problems
}
