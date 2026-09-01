package extproc

import (
	"context"
	"errors"
	"testing"
)

type recordingSelectionTransaction struct {
	validateErr error
	commitErr   error
	abortErr    error
	settleErr   error

	validates int
	commits   int
	aborts    int
	settles   int
	status    int
	reason    string
	outcome   selectionActualOutcome
}

func (transaction *recordingSelectionTransaction) ValidateDispatch(
	context.Context,
) error {
	transaction.validates++
	return transaction.validateErr
}

func (transaction *recordingSelectionTransaction) CommitOnHeaders(
	_ context.Context,
	status int,
) error {
	transaction.commits++
	transaction.status = status
	return transaction.commitErr
}

func (transaction *recordingSelectionTransaction) Abort(
	_ context.Context,
	reason string,
) error {
	transaction.aborts++
	transaction.reason = reason
	return transaction.abortErr
}

func (transaction *recordingSelectionTransaction) Settle(
	_ context.Context,
	outcome selectionActualOutcome,
) error {
	transaction.settles++
	transaction.outcome = outcome
	return transaction.settleErr
}

func TestSelectionTransactionOwnerRunsEachTerminalOperationOnce(
	t *testing.T,
) {
	transaction := &recordingSelectionTransaction{}
	ctx := &RequestContext{
		UpstreamStatusCode: 200,
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			transaction,
		),
	}
	if err := selectionDispatchAllowed(ctx); err != nil {
		t.Fatal(err)
	}
	if err := finalizeSelectionResponseHeaders(ctx, true); err != nil {
		t.Fatal(err)
	}
	inputTokens := 17
	outcome := selectionActualOutcome{
		OutcomeClass: "success",
		StatusCode:   200,
		InputTokens:  &inputTokens,
	}
	finalizeSelectionSettlement(ctx, outcome)
	finalizeSelectionSettlement(ctx, outcome)
	finalizeSelectionProcessTerminal(ctx)

	if transaction.validates != 1 ||
		transaction.commits != 1 ||
		transaction.aborts != 0 ||
		transaction.settles != 1 {
		t.Fatalf(
			"lifecycle calls validate=%d commit=%d abort=%d settle=%d",
			transaction.validates,
			transaction.commits,
			transaction.aborts,
			transaction.settles,
		)
	}
	if transaction.status != 200 ||
		transaction.outcome.InputTokens == nil ||
		*transaction.outcome.InputTokens != inputTokens {
		t.Fatalf("terminal facts = %#v", transaction)
	}
}

func TestSelectionTransactionOwnerAbortsWithoutCommitOnNon2xx(
	t *testing.T,
) {
	transaction := &recordingSelectionTransaction{}
	ctx := &RequestContext{
		UpstreamStatusCode: 503,
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			transaction,
		),
	}
	if err := finalizeSelectionResponseHeaders(ctx, false); err != nil {
		t.Fatal(err)
	}
	finalizeSelectionProcessTerminal(ctx)
	if transaction.aborts != 1 ||
		transaction.commits != 0 ||
		transaction.settles != 0 ||
		transaction.reason != "upstream_non_2xx" {
		t.Fatalf("terminal calls = %#v", transaction)
	}
}

func TestSelectionDispatchFailureRemainsPreCommit(t *testing.T) {
	transaction := &recordingSelectionTransaction{
		validateErr: errors.New("private transport detail"),
	}
	ctx := &RequestContext{
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			transaction,
		),
	}
	if err := selectionDispatchAllowed(ctx); err == nil {
		t.Fatal("expected dispatch validation failure")
	}
	finalizeSelectionAbort(ctx, "dispatch_transport")
	finalizeSelectionProcessTerminal(ctx)
	if transaction.validates != 1 ||
		transaction.aborts != 1 ||
		transaction.commits != 0 {
		t.Fatalf("terminal calls = %#v", transaction)
	}
}

func TestSelectionProcessTerminalSettlesCommittedBrokenStream(
	t *testing.T,
) {
	transaction := &recordingSelectionTransaction{}
	ctx := &RequestContext{
		UpstreamStatusCode:  200,
		IsStreamingResponse: true,
		StreamingAborted:    true,
		SelectionTransaction: newSelectionTransactionOwner(
			configRaylineARC,
			transaction,
		),
	}
	if err := finalizeSelectionResponseHeaders(ctx, true); err != nil {
		t.Fatal(err)
	}
	finalizeSelectionProcessTerminal(ctx)
	if transaction.aborts != 0 ||
		transaction.settles != 1 ||
		transaction.outcome.OutcomeClass != "stream_error" ||
		transaction.outcome.InputTokens != nil ||
		transaction.outcome.OutputTokens != nil {
		t.Fatalf("terminal calls = %#v", transaction)
	}
}

func TestSelectionSettlementPreservesAbsentUsageEvidence(
	t *testing.T,
) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{UpstreamStatusCode: 200}
	outcome := router.selectionOutcomeForUsage(
		ctx,
		responseUsageMetrics{},
		0,
		"success",
	)
	if outcome.InputTokens != nil ||
		outcome.OutputTokens != nil ||
		outcome.CostUSD != nil ||
		outcome.LatencyMS != nil {
		t.Fatalf("unknown evidence was fabricated: %#v", outcome)
	}
}
