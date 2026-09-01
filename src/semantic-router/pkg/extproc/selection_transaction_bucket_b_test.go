//go:build vsr_next_bucket_b

// Parked until Bucket B re-seats the selection-transaction commit on the
// upstream response-header path. Build with -tags vsr_next_bucket_b once
// finalizeSelectionResponseHeaders is wired into handleResponseHeaders again.

package extproc

import (
	"errors"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func TestResponseHeaderCommitFailureBecomesTyped503(t *testing.T) {
	transaction := &recordingSelectionTransaction{
		commitErr: errors.New("private coordinator response"),
	}
	ctx := &RequestContext{
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			transaction,
		),
	}
	response, err := (&OpenAIRouter{}).handleResponseHeaders(
		&ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{
						{Key: ":status", Value: "200"},
					},
				},
			},
		},
		ctx,
	)
	if err != nil {
		t.Fatal(err)
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil ||
		int(immediate.GetStatus().GetCode()) != 503 ||
		!strings.Contains(
			string(immediate.Body),
			"Rayline remote routing unavailable",
		) {
		t.Fatalf("response = %#v", response)
	}
	finalizeSelectionProcessTerminal(ctx)
	if transaction.commits != 1 || transaction.aborts != 1 {
		t.Fatalf("terminal calls = %#v", transaction)
	}
}
