package raylineremote

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"sync"
	"time"
)

type transactionState string

const (
	transactionPrepared  transactionState = "prepared"
	transactionCommitted transactionState = "committed"
	transactionAborted   transactionState = "aborted"
	transactionSettled   transactionState = "settled"
)

type transactionWireRequest struct {
	SchemaVersion string `json:"schema_version"`
	DecisionID    string `json:"decision_id"`
	Receipt       string `json:"receipt"`
}

type commitWireRequest struct {
	transactionWireRequest
	StatusCode int `json:"status_code"`
}

type abortWireRequest struct {
	transactionWireRequest
	Reason AbortReason `json:"reason"`
}

type settleWireRequest struct {
	transactionWireRequest
	Outcome ActualOutcome `json:"outcome"`
}

type stateWireResponse struct {
	SchemaVersion  string `json:"schema_version"`
	DecisionID     string `json:"decision_id"`
	Receipt        string `json:"receipt"`
	State          string `json:"state"`
	RouteCallIndex int    `json:"route_call_index"`
	LeaseExpiresAt string `json:"lease_expires_at,omitempty"`
}

// Transaction is a request-scoped capability. Its identifiers are kept
// private so callers cannot accidentally add receipts to logs or traces.
type Transaction struct {
	client         *Client
	decisionID     string
	receipt        string
	selectedWorker string
	routeCallIndex int

	mu             sync.Mutex
	state          transactionState
	commitStatus   int
	abortReason    AbortReason
	outcomeDigest  [sha256.Size]byte
	leaseExpiresAt time.Time
	leaseFailure   error
	keeperCancel   context.CancelFunc
	keeperDone     chan struct{}
}

func newTransaction(
	client *Client,
	decisionID string,
	receipt string,
	selectedWorker string,
	routeCallIndex int,
	leaseExpiresAt time.Time,
) *Transaction {
	return &Transaction{
		client:         client,
		decisionID:     decisionID,
		receipt:        receipt,
		selectedWorker: selectedWorker,
		routeCallIndex: routeCallIndex,
		state:          transactionPrepared,
		leaseExpiresAt: leaseExpiresAt,
	}
}

func (transaction *Transaction) SelectedWorker() string {
	if transaction == nil {
		return ""
	}
	return transaction.selectedWorker
}

func (transaction *Transaction) RouteCallIndex() int {
	if transaction == nil {
		return -1
	}
	return transaction.routeCallIndex
}

// StartLeaseKeeper periodically renews the receipt until Commit or Abort.
// Exactly one keeper may own a transaction.
func (transaction *Transaction) StartLeaseKeeper(parent context.Context) error {
	if transaction == nil || transaction.client == nil {
		return newFailure(
			FailureRequest,
			"renew",
			"transaction_required",
			0,
		)
	}
	transaction.mu.Lock()
	defer transaction.mu.Unlock()
	if transaction.state != transactionPrepared {
		return newFailure(
			FailureState,
			"renew",
			"not_prepared",
			0,
		)
	}
	if transaction.keeperCancel != nil {
		return nil
	}
	ctx, cancel := context.WithCancel(parent)
	done := make(chan struct{})
	transaction.keeperCancel = cancel
	transaction.keeperDone = done
	interval := transaction.client.config.LeaseTTL / 3
	go transaction.keepLease(ctx, done, interval)
	return nil
}

func (transaction *Transaction) keepLease(
	ctx context.Context,
	done chan struct{},
	interval time.Duration,
) {
	defer close(done)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if _, err := transaction.renew(ctx); err != nil {
				transaction.recordLeaseKeeperFailure(ctx)
				return
			}
		}
	}
}

func (transaction *Transaction) recordLeaseKeeperFailure(
	ctx context.Context,
) {
	if ctx.Err() != nil {
		return
	}
	transaction.mu.Lock()
	defer transaction.mu.Unlock()
	if transaction.leaseFailure == nil {
		transaction.leaseFailure = newFailure(
			FailureLease,
			"renew",
			"keeper_failed",
			0,
		)
	}
}

// ValidateDispatch consumes any asynchronous lease loss and performs a
// synchronous renewal immediately before provider dispatch.
func (transaction *Transaction) ValidateDispatch(
	ctx context.Context,
) error {
	if transaction == nil {
		return newFailure(
			FailureRequest,
			"renew",
			"transaction_required",
			0,
		)
	}
	transaction.mu.Lock()
	if transaction.state != transactionPrepared {
		transaction.mu.Unlock()
		return newFailure(
			FailureState,
			"renew",
			"not_prepared",
			0,
		)
	}
	leaseFailure := transaction.leaseFailure
	transaction.mu.Unlock()
	if leaseFailure != nil {
		return leaseFailure
	}
	_, err := transaction.renew(ctx)
	return err
}

func (transaction *Transaction) renew(
	ctx context.Context,
) (time.Time, error) {
	transaction.mu.Lock()
	defer transaction.mu.Unlock()
	if transaction.state != transactionPrepared {
		return time.Time{}, newFailure(
			FailureState,
			"renew",
			"not_prepared",
			0,
		)
	}
	if transaction.leaseFailure != nil {
		return time.Time{}, transaction.leaseFailure
	}
	request := transaction.baseRequest()
	var response stateWireResponse
	if err := transaction.client.post(
		ctx,
		"renew",
		"/v1/route/renew",
		request,
		&response,
	); err != nil {
		return time.Time{}, err
	}
	if err := transaction.validateStateResponse(
		response,
		"prepared",
	); err != nil {
		return time.Time{}, err
	}
	expiresAt, err := time.Parse(
		time.RFC3339Nano,
		response.LeaseExpiresAt,
	)
	if err != nil || !expiresAt.After(transaction.client.now()) {
		return time.Time{}, newFailure(
			FailureLease,
			"renew",
			"invalid_lease",
			0,
		)
	}
	transaction.leaseExpiresAt = expiresAt
	return expiresAt, nil
}

func (transaction *Transaction) CommitOnHeaders(
	ctx context.Context,
	statusCode int,
) error {
	if transaction == nil || transaction.client == nil {
		return newFailure(
			FailureRequest,
			"commit",
			"transaction_required",
			0,
		)
	}
	if statusCode < 200 || statusCode > 299 {
		return newFailure(
			FailureRequest,
			"commit",
			"invalid_status_code",
			0,
		)
	}
	transaction.stopLeaseKeeper()
	transaction.mu.Lock()
	defer transaction.mu.Unlock()
	alreadyCommitted, err := transaction.validateCommitTransition(statusCode)
	if err != nil || alreadyCommitted {
		return err
	}
	request := commitWireRequest{
		transactionWireRequest: transaction.baseRequest(),
		StatusCode:             statusCode,
	}
	var response stateWireResponse
	if err := transaction.client.post(
		ctx,
		"commit",
		"/v1/route/commit",
		request,
		&response,
	); err != nil {
		return err
	}
	if response.State != "committed" &&
		response.State != "settled" {
		return newFailure(
			FailureContract,
			"commit",
			"invalid_state",
			0,
		)
	}
	if err := transaction.validateStateResponse(
		response,
		response.State,
	); err != nil {
		return err
	}
	transaction.state = transactionCommitted
	transaction.commitStatus = statusCode
	return nil
}

func (transaction *Transaction) validateCommitTransition(
	statusCode int,
) (bool, error) {
	if transaction.leaseFailure != nil {
		return false, transaction.leaseFailure
	}
	if transaction.state == transactionCommitted ||
		transaction.state == transactionSettled {
		if transaction.commitStatus == statusCode {
			return true, nil
		}
		return false, newFailure(
			FailureState,
			"commit",
			"status_conflict",
			0,
		)
	}
	if transaction.state != transactionPrepared {
		return false, newFailure(
			FailureState,
			"commit",
			"not_prepared",
			0,
		)
	}
	return false, nil
}

func (transaction *Transaction) Abort(
	ctx context.Context,
	reason AbortReason,
) error {
	if transaction == nil || transaction.client == nil {
		return newFailure(
			FailureRequest,
			"abort",
			"transaction_required",
			0,
		)
	}
	if !validAbortReason(reason) {
		return newFailure(
			FailureRequest,
			"abort",
			"invalid_reason",
			0,
		)
	}
	transaction.stopLeaseKeeper()
	transaction.mu.Lock()
	defer transaction.mu.Unlock()
	if transaction.state == transactionAborted {
		if transaction.abortReason == reason {
			return nil
		}
		return newFailure(
			FailureState,
			"abort",
			"reason_conflict",
			0,
		)
	}
	if transaction.state != transactionPrepared {
		return newFailure(
			FailureState,
			"abort",
			"not_prepared",
			0,
		)
	}
	request := abortWireRequest{
		transactionWireRequest: transaction.baseRequest(),
		Reason:                 reason,
	}
	var response stateWireResponse
	if err := transaction.client.post(
		ctx,
		"abort",
		"/v1/route/abort",
		request,
		&response,
	); err != nil {
		return err
	}
	if response.State != "aborted" &&
		response.State != "expired" {
		return newFailure(
			FailureContract,
			"abort",
			"invalid_state",
			0,
		)
	}
	if err := transaction.validateStateResponse(
		response,
		response.State,
	); err != nil {
		return err
	}
	transaction.state = transactionAborted
	transaction.abortReason = reason
	return nil
}

func (transaction *Transaction) Settle(
	ctx context.Context,
	outcome ActualOutcome,
) error {
	if transaction == nil || transaction.client == nil {
		return newFailure(
			FailureRequest,
			"settle",
			"transaction_required",
			0,
		)
	}
	if err := validateOutcome(outcome); err != nil {
		return err
	}
	encodedOutcome, err := json.Marshal(outcome)
	if err != nil {
		return newFailure(
			FailureRequest,
			"settle",
			"encode",
			0,
		)
	}
	outcomeDigest := sha256.Sum256(encodedOutcome)
	transaction.mu.Lock()
	defer transaction.mu.Unlock()
	if transaction.state == transactionSettled {
		if transaction.outcomeDigest == outcomeDigest {
			return nil
		}
		return newFailure(
			FailureState,
			"settle",
			"outcome_conflict",
			0,
		)
	}
	if transaction.state != transactionCommitted {
		return newFailure(
			FailureState,
			"settle",
			"not_committed",
			0,
		)
	}
	request := settleWireRequest{
		transactionWireRequest: transaction.baseRequest(),
		Outcome:                outcome,
	}
	var response stateWireResponse
	if err := transaction.client.post(
		ctx,
		"settle",
		"/v1/route/settle",
		request,
		&response,
	); err != nil {
		return err
	}
	if err := transaction.validateStateResponse(
		response,
		"settled",
	); err != nil {
		return err
	}
	transaction.state = transactionSettled
	transaction.outcomeDigest = outcomeDigest
	return nil
}

func (transaction *Transaction) baseRequest() transactionWireRequest {
	return transactionWireRequest{
		SchemaVersion: TransactionSchemaVersion,
		DecisionID:    transaction.decisionID,
		Receipt:       transaction.receipt,
	}
}

func (transaction *Transaction) validateStateResponse(
	response stateWireResponse,
	expectedState string,
) error {
	if response.SchemaVersion != TransactionSchemaVersion ||
		response.DecisionID != transaction.decisionID ||
		response.Receipt != transaction.receipt ||
		response.RouteCallIndex != transaction.routeCallIndex ||
		response.State != expectedState {
		return newFailure(
			FailureContract,
			expectedState,
			"invalid_response",
			0,
		)
	}
	return nil
}

func (transaction *Transaction) stopLeaseKeeper() {
	if transaction == nil {
		return
	}
	transaction.mu.Lock()
	cancel := transaction.keeperCancel
	done := transaction.keeperDone
	transaction.keeperCancel = nil
	transaction.keeperDone = nil
	transaction.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
}

func IsFailureClass(err error, class FailureClass) bool {
	var failure *Failure
	return errors.As(err, &failure) && failure.Class == class
}
