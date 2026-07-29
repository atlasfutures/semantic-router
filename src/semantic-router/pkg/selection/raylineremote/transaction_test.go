package raylineremote

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

//nolint:cyclop // The assertions cover the lease keeper's asynchronous failure boundary.
func TestLeaseKeeperFailureIsConsumedBeforeDispatch(t *testing.T) {
	var renewCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		switch request.URL.Path {
		case "/v1/route/prepare":
			writeJSON(t, writer, validPrepareResponse())
		case "/v1/route/renew":
			renewCalls.Add(1)
			writer.WriteHeader(http.StatusGone)
			writeJSON(t, writer, map[string]any{
				"schema_version": TransactionSchemaVersion,
				"error": map[string]any{
					"code":    "selection_lease_expired",
					"message": "expired",
				},
			})
		default:
			writer.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	client, err := NewClient(validClientConfig(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	// Use a short test interval after validating the public configuration
	// bounds; production configuration remains at the advertised 30 seconds.
	client.config.LeaseTTL = 15 * time.Millisecond
	transaction, _, err := client.Prepare(
		context.Background(),
		validPrepareInput(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if leaseErr := transaction.StartLeaseKeeper(ctx); leaseErr != nil {
		t.Fatal(leaseErr)
	}
	deadline := time.Now().Add(time.Second)
	for renewCalls.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if renewCalls.Load() == 0 {
		t.Fatal("lease keeper did not renew")
	}
	// Let the keeper publish its bounded failure after the HTTP response.
	for {
		transaction.mu.Lock()
		failed := transaction.leaseFailure != nil
		transaction.mu.Unlock()
		if failed || time.Now().After(deadline) {
			break
		}
		time.Sleep(time.Millisecond)
	}
	err = transaction.ValidateDispatch(context.Background())
	if !IsFailureClass(err, FailureLease) {
		t.Fatalf("error = %v, want lease failure", err)
	}
}

func TestTransactionRejectsInvalidLocalTransitions(t *testing.T) {
	fixture := newProtocolFixture(t)
	defer fixture.server.Close()
	transaction, _, err := fixture.client(t).Prepare(
		context.Background(),
		validPrepareInput(t),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := transaction.CommitOnHeaders(
		context.Background(),
		http.StatusBadGateway,
	); !IsFailureClass(err, FailureRequest) {
		t.Fatalf("non-2xx commit error = %v", err)
	}
	if err := transaction.Settle(
		context.Background(),
		ActualOutcome{
			OutcomeClass: OutcomeSuccess,
			StatusCode:   http.StatusOK,
		},
	); !IsFailureClass(err, FailureState) {
		t.Fatalf("pre-commit settlement error = %v", err)
	}
	if err := transaction.Abort(
		context.Background(),
		AbortReason("private-reason"),
	); !IsFailureClass(err, FailureRequest) {
		t.Fatalf("invalid abort error = %v", err)
	}
}

func TestOutcomeValidationPreservesUnknownEvidence(t *testing.T) {
	if err := validateOutcome(ActualOutcome{
		OutcomeClass: OutcomeUnknown,
		StatusCode:   0,
	}); err != nil {
		t.Fatal(err)
	}
	negative := -1
	if err := validateOutcome(ActualOutcome{
		OutcomeClass: OutcomeSuccess,
		StatusCode:   http.StatusOK,
		InputTokens:  &negative,
	}); !IsFailureClass(err, FailureRequest) {
		t.Fatalf("negative token error = %v", err)
	}
}
