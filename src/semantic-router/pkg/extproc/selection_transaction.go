package extproc

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylineremote"
)

type selectionActualOutcome struct {
	OutcomeClass string
	StatusCode   int
	InputTokens  *int
	OutputTokens *int
	LatencyMS    *float64
	CostUSD      *float64
	ErrorType    *string
}

type selectionTransaction interface {
	ValidateDispatch(context.Context) error
	CommitOnHeaders(context.Context, int) error
	Abort(context.Context, string) error
	Settle(context.Context, selectionActualOutcome) error
}

type selectionTransactionOwner struct {
	kind        string
	transaction selectionTransaction

	mu        sync.Mutex
	committed bool
	aborted   bool
	settled   bool
}

func newSelectionTransactionOwner(
	kind string,
	transaction selectionTransaction,
) *selectionTransactionOwner {
	return &selectionTransactionOwner{
		kind:        kind,
		transaction: transaction,
	}
}

func (owner *selectionTransactionOwner) validateDispatch(
	ctx context.Context,
) error {
	if owner == nil || owner.transaction == nil {
		return errors.New("selection transaction is unavailable")
	}
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.committed || owner.aborted {
		return errors.New("selection transaction is already terminal")
	}
	return owner.transaction.ValidateDispatch(ctx)
}

func (owner *selectionTransactionOwner) commitOnHeaders(
	ctx context.Context,
	statusCode int,
) (bool, error) {
	if owner == nil || owner.transaction == nil {
		return false, errors.New("selection transaction is unavailable")
	}
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.committed {
		return false, nil
	}
	if owner.aborted {
		return false, errors.New("selection transaction was aborted")
	}
	if err := owner.transaction.CommitOnHeaders(
		ctx,
		statusCode,
	); err != nil {
		return false, err
	}
	owner.committed = true
	return true, nil
}

func (owner *selectionTransactionOwner) abort(
	ctx context.Context,
	class string,
) (bool, error) {
	if owner == nil || owner.transaction == nil {
		return false, nil
	}
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.committed || owner.aborted {
		return false, nil
	}
	if err := owner.transaction.Abort(ctx, class); err != nil {
		return false, err
	}
	owner.aborted = true
	return true, nil
}

func (owner *selectionTransactionOwner) settle(
	ctx context.Context,
	outcome selectionActualOutcome,
) (bool, error) {
	if owner == nil || owner.transaction == nil {
		return false, nil
	}
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if !owner.committed || owner.aborted || owner.settled {
		return false, nil
	}
	if err := owner.transaction.Settle(ctx, outcome); err != nil {
		return false, err
	}
	owner.settled = true
	return true, nil
}

func (owner *selectionTransactionOwner) isCommitted() bool {
	if owner == nil {
		return false
	}
	owner.mu.Lock()
	defer owner.mu.Unlock()
	return owner.committed
}

type raylineARCSelectionTransactionAdapter struct {
	transaction *raylineARCEpisodeTransaction
	request     *RequestContext
}

func (adapter *raylineARCSelectionTransactionAdapter) ValidateDispatch(
	context.Context,
) error {
	if adapter == nil || adapter.transaction == nil ||
		!adapter.transaction.dispatchAllowed() {
		return ErrRaylineARCEpisodeLeaseLost
	}
	return nil
}

func (adapter *raylineARCSelectionTransactionAdapter) CommitOnHeaders(
	ctx context.Context,
	_ int,
) error {
	return adapter.transaction.commit(ctx, adapter.request)
}

func (adapter *raylineARCSelectionTransactionAdapter) Abort(
	ctx context.Context,
	class string,
) error {
	return adapter.transaction.abort(ctx, class)
}

func (*raylineARCSelectionTransactionAdapter) Settle(
	context.Context,
	selectionActualOutcome,
) error {
	return nil
}

type raylineRemoteSelectionTransactionAdapter struct {
	transaction *raylineremote.Transaction
}

func (adapter *raylineRemoteSelectionTransactionAdapter) ValidateDispatch(
	ctx context.Context,
) error {
	if adapter == nil || adapter.transaction == nil {
		return errors.New("remote selection transaction is unavailable")
	}
	return adapter.transaction.ValidateDispatch(ctx)
}

func (adapter *raylineRemoteSelectionTransactionAdapter) CommitOnHeaders(
	ctx context.Context,
	statusCode int,
) error {
	return adapter.transaction.CommitOnHeaders(ctx, statusCode)
}

func (adapter *raylineRemoteSelectionTransactionAdapter) Abort(
	ctx context.Context,
	class string,
) error {
	return adapter.transaction.Abort(ctx, remoteAbortReason(class))
}

func (adapter *raylineRemoteSelectionTransactionAdapter) Settle(
	ctx context.Context,
	outcome selectionActualOutcome,
) error {
	return adapter.transaction.Settle(
		ctx,
		raylineremote.ActualOutcome{
			OutcomeClass: remoteOutcomeClass(outcome.OutcomeClass),
			StatusCode:   outcome.StatusCode,
			InputTokens:  outcome.InputTokens,
			OutputTokens: outcome.OutputTokens,
			LatencyMS:    outcome.LatencyMS,
			CostUSD:      outcome.CostUSD,
			ErrorType:    outcome.ErrorType,
		},
	)
}

func remoteAbortReason(class string) raylineremote.AbortReason {
	switch {
	case strings.Contains(class, "cancel"):
		return raylineremote.AbortClientCancelled
	case strings.Contains(class, "non_2xx"):
		return raylineremote.AbortProviderNon2xx
	case strings.Contains(class, "transport"):
		return raylineremote.AbortProviderNetwork
	case strings.Contains(class, "dispatch"):
		return raylineremote.AbortDispatchFailure
	case strings.Contains(class, "selection"):
		return raylineremote.AbortSelectionFailed
	default:
		return raylineremote.AbortRequestFailure
	}
}

func remoteOutcomeClass(class string) raylineremote.OutcomeClass {
	switch class {
	case "success":
		return raylineremote.OutcomeSuccess
	case "upstream_error":
		return raylineremote.OutcomeUpstreamError
	case "client_cancelled":
		return raylineremote.OutcomeClientCancelled
	case "stream_error":
		return raylineremote.OutcomeStreamError
	default:
		return raylineremote.OutcomeUnknown
	}
}

func bindRaylineARCSelectionTransaction(ctx *RequestContext) {
	if ctx == nil || ctx.RaylineARCTransaction == nil {
		return
	}
	ctx.SelectionTransaction = newSelectionTransactionOwner(
		configRaylineARC,
		&raylineARCSelectionTransactionAdapter{
			transaction: ctx.RaylineARCTransaction,
			request:     ctx,
		},
	)
}

// ensureSelectionTransactionBound preserves the ARC lifecycle contract for
// legacy/internal callers that construct RequestContext directly. Production
// ARC selection binds eagerly, but every terminal seam must still recognize
// the prepared lease rather than silently shortcut or leave it pending.
func ensureSelectionTransactionBound(ctx *RequestContext) {
	if ctx == nil || ctx.SelectionTransaction != nil ||
		ctx.RaylineARCTransaction == nil {
		return
	}
	bindRaylineARCSelectionTransaction(ctx)
}

func bindRaylineRemoteSelectionTransaction(
	ctx *RequestContext,
	transaction *raylineremote.Transaction,
) error {
	if ctx == nil || transaction == nil ||
		ctx.SelectionTransaction != nil {
		return errors.New("remote selection transaction binding failed")
	}
	ctx.SelectionTransaction = newSelectionTransactionOwner(
		configRaylineRemote,
		&raylineRemoteSelectionTransactionAdapter{
			transaction: transaction,
		},
	)
	return nil
}

const (
	configRaylineARC    = "rayline_arc"
	configRaylineRemote = "rayline_remote"
)

func selectionDispatchAllowed(ctx *RequestContext) error {
	ensureSelectionTransactionBound(ctx)
	if ctx == nil || ctx.SelectionTransaction == nil {
		return nil
	}
	validateContext, cancel := context.WithTimeout(
		context.Background(),
		episodeFinalizeTimeout,
	)
	defer cancel()
	return ctx.SelectionTransaction.validateDispatch(validateContext)
}

func finalizeSelectionAbort(
	ctx *RequestContext,
	class string,
) {
	ensureSelectionTransactionBound(ctx)
	if ctx == nil || ctx.SelectionTransaction == nil {
		return
	}
	finalizeContext, cancel := context.WithTimeout(
		context.Background(),
		episodeFinalizeTimeout,
	)
	defer cancel()
	performed, err := ctx.SelectionTransaction.abort(
		finalizeContext,
		class,
	)
	if err != nil {
		logSelectionTransactionFailure(
			ctx.SelectionTransaction.kind,
			"abort",
			err,
		)
		return
	}
	if performed {
		recordSelectionTransactionSuccess(
			ctx.SelectionTransaction.kind,
			"abort",
			class,
		)
	}
}

func finalizeSelectionResponseHeaders(
	ctx *RequestContext,
	successful bool,
) error {
	ensureSelectionTransactionBound(ctx)
	if ctx == nil || ctx.SelectionTransaction == nil {
		return nil
	}
	finalizeTimeout := episodeFinalizeTimeout
	if ctx.RaylineARCTransaction != nil {
		finalizeTimeout = ctx.RaylineARCTransaction.finalizeTimeout()
	}
	finalizeContext, cancel := context.WithTimeout(
		context.Background(),
		finalizeTimeout,
	)
	defer cancel()
	if !successful {
		performed, err := ctx.SelectionTransaction.abort(
			finalizeContext,
			"upstream_non_2xx",
		)
		if err == nil && performed {
			recordSelectionTransactionSuccess(
				ctx.SelectionTransaction.kind,
				"abort",
				"upstream_non_2xx",
			)
		}
		return err
	}
	performed, err := ctx.SelectionTransaction.commitOnHeaders(
		finalizeContext,
		ctx.UpstreamStatusCode,
	)
	if err == nil && performed {
		recordSelectionTransactionSuccess(
			ctx.SelectionTransaction.kind,
			"commit",
			"success",
		)
	}
	return err
}

func finalizeSelectionSettlement(
	ctx *RequestContext,
	outcome selectionActualOutcome,
) {
	ensureSelectionTransactionBound(ctx)
	if ctx == nil || ctx.SelectionTransaction == nil {
		return
	}
	settleContext, cancel := context.WithTimeout(
		context.Background(),
		episodeFinalizeTimeout,
	)
	defer cancel()
	performed, err := ctx.SelectionTransaction.settle(
		settleContext,
		outcome,
	)
	if err != nil {
		// Settlement is post-commit observation. It must be visible, but cannot
		// replace a successful provider response or rerun selection.
		logSelectionTransactionFailure(
			ctx.SelectionTransaction.kind,
			"settle",
			err,
		)
		return
	}
	if performed {
		recordSelectionTransactionSuccess(
			ctx.SelectionTransaction.kind,
			"settle",
			outcome.OutcomeClass,
		)
	}
}

func finalizeSelectionProcessTerminal(ctx *RequestContext) {
	ensureSelectionTransactionBound(ctx)
	if ctx == nil || ctx.SelectionTransaction == nil {
		return
	}
	if !ctx.SelectionTransaction.isCommitted() {
		finalizeSelectionAbort(ctx, processAbortClass(ctx))
		return
	}
	defaultOutcome := "unknown"
	if ctx.StreamingAborted {
		defaultOutcome = "stream_error"
	}
	finalizeSelectionSettlement(
		ctx,
		selectionOutcomeFromContext(ctx, defaultOutcome),
	)
}

func processAbortClass(ctx *RequestContext) string {
	if ctx == nil {
		return "process_terminal"
	}
	if ctx.SelectionSettlement.OutcomeClass != "" {
		return ctx.SelectionSettlement.OutcomeClass
	}
	if ctx.StreamingAborted {
		return "stream_error"
	}
	return "process_terminal"
}

func selectionOutcomeFromContext(
	ctx *RequestContext,
	defaultClass string,
) selectionActualOutcome {
	outcome := ctx.SelectionSettlement
	if outcome.OutcomeClass == "" {
		outcome.OutcomeClass = defaultClass
	}
	if outcome.StatusCode == 0 {
		outcome.StatusCode = ctx.UpstreamStatusCode
	}
	if outcome.LatencyMS == nil && !ctx.StartTime.IsZero() {
		latency := float64(time.Since(ctx.StartTime).Milliseconds())
		outcome.LatencyMS = &latency
	}
	return outcome
}

func logSelectionTransactionFailure(
	kind string,
	stage string,
	err error,
) {
	logging.ComponentErrorEvent(
		"extproc",
		"selection_transaction_failed",
		map[string]interface{}{
			"algorithm":     kind,
			"stage":         stage,
			"failure_class": boundedSelectionTransactionFailure(err),
		},
	)
	if kind == configRaylineRemote {
		metrics.RecordRaylineRemoteFailure(
			stage,
			boundedSelectionTransactionFailure(err),
		)
	}
}

func recordSelectionTransactionSuccess(
	kind string,
	operation string,
	outcome string,
) {
	if kind == configRaylineRemote {
		metrics.RecordRaylineRemoteTransaction(
			operation,
			outcome,
		)
	}
}

func recordSelectionLifecycleFailure(
	ctx *RequestContext,
	stage string,
	err error,
) {
	kind := ""
	if ctx != nil && ctx.SelectionTransaction != nil {
		kind = ctx.SelectionTransaction.kind
	}
	logSelectionTransactionFailure(kind, stage, err)
}

func selectionUnavailableMessage(ctx *RequestContext) string {
	if ctx != nil &&
		ctx.SelectionTransaction != nil &&
		ctx.SelectionTransaction.kind == configRaylineRemote {
		return "Rayline remote routing unavailable"
	}
	return "Rayline ARC routing unavailable"
}

func boundedSelectionTransactionFailure(err error) string {
	var remoteFailure *raylineremote.Failure
	switch {
	case err == nil:
		return ""
	case errors.As(err, &remoteFailure):
		return string(remoteFailure.Class)
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, ErrRaylineARCEpisodeLeaseLost):
		return "lease_lost"
	default:
		return "transaction"
	}
}
