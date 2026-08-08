package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

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
