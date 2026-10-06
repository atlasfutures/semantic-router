package extproc

import (
	"errors"
	"testing"
)

// A selection transaction failure is logged with the turn's request id, so it
// joins the turn's other records and the gateway's route.
func TestSelectionTransactionFailureCarriesTheRequestID(t *testing.T) {
	logs := captureLogs(t)
	recordSelectionLifecycleFailure(&RequestContext{RequestID: "req-selection-failed"}, "prepare", errors.New("policy refused"))
	if got := findLogEvent(t, logs, "selection_transaction_failed")["request_id"]; got != "req-selection-failed" {
		t.Fatalf("selection_transaction_failed request_id = %v, want the turn's id", got)
	}
}
