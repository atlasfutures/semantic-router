package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) selectWithCandidateSelector(
	selCtx *selection.SelectionContext,
	algorithm *config.AlgorithmConfig,
	ctx *RequestContext,
	defaultCandidateModelRef *config.ModelRef,
	method selection.SelectionMethod,
	selector selection.Selector,
) (*config.ModelRef, string, error) {
	selectionContext := modelSelectionContext(ctx)
	result, err := selector.Select(selectionContext, selCtx)
	if err != nil {
		return r.handleSelectionFallback(
			algorithm,
			authoritativeSelectionFailureClass(algorithm, err),
			string(method),
			selCtx,
			nil,
			defaultCandidateModelRef,
			ctx,
			"[ModelSelection] Selection failed: %v, using default candidate",
			err,
		)
	}
	if err := selection.ValidateSelectionResult(selCtx, result); err != nil {
		return r.handleSelectionFallback(
			algorithm,
			"invalid_result",
			string(method),
			selCtx,
			result,
			defaultCandidateModelRef,
			ctx,
			"[ModelSelection] Invalid selection result: %v, using default candidate",
			err,
		)
	}

	selectedModelRef := selectedModelRefFromResult(selCtx, result)
	if selectedModelRef == nil {
		return r.handleSelectionFallback(
			algorithm,
			"unknown_model",
			string(method),
			selCtx,
			result,
			defaultCandidateModelRef,
			ctx,
			"[ModelSelection] Selected model %s not found in candidates, using default candidate",
			result.SelectedModel,
		)
	}
	if err := bindRaylineARCDispatchContract(
		selector,
		algorithm,
		result,
		selectedModelRef,
		ctx,
	); err != nil {
		return nil, "", err
	}
	if err := bindRaylineRemoteDispatchContract(
		selCtx,
		algorithm,
		result,
		selectedModelRef,
		ctx,
	); err != nil {
		return nil, "", err
	}
	return r.completeModelSelection(
		selCtx,
		algorithm,
		ctx,
		method,
		result,
		selectedModelRef,
	)
}

func modelSelectionContext(ctx *RequestContext) context.Context {
	if ctx != nil && ctx.TraceContext != nil {
		return ctx.TraceContext
	}
	return context.Background()
}

func bindRaylineARCDispatchContract(
	selector selection.Selector,
	algorithm *config.AlgorithmConfig,
	result *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) error {
	if !raylineARCSelection(algorithm) {
		return nil
	}
	dispatchProvider, ok := selector.(raylineARCWorkerProvider)
	if !ok || result.RaylineARC == nil {
		return selectionFailureForAlgorithm(algorithm, "dispatch_contract")
	}
	worker, found := dispatchProvider.Worker(result.RaylineARC.SelectedArm)
	if !found || worker.ID != selectedModelRef.Model {
		return selectionFailureForAlgorithm(algorithm, "dispatch_contract")
	}
	if ctx == nil {
		return selectionFailureForAlgorithm(algorithm, "dispatch_context")
	}
	ctx.RaylineARCDispatch = &worker
	return nil
}
