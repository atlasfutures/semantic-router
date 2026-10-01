package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// validateAcceptedFormatsCodec is validateModelDemand's codec check. The
// demand-only paths that call it know no client format, so they cannot know
// which of the model's accepted formats (#123) a request would go out in. The
// check therefore passes when any accepted format's codec preserves the
// demand: permissive at selection, exact at dispatch, where
// rejectDispatchCapabilityMismatch qualifies the format actually chosen. The
// same rule #123 applies to config-time checks. When no format passes, the
// first format's error is returned, as before #123.
func (r *OpenAIRouter) validateAcceptedFormatsCodec(
	requirements *config.CandidateRequirements,
	model string,
	demand selection.CandidateDemand,
) error {
	var firstErr error
	for _, apiFormat := range r.Config.GetModelAcceptedFormats(model) {
		err := r.validateFormatCodec(requirements, model, apiFormat, demand)
		if err == nil {
			return nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (r *OpenAIRouter) validateFormatCodec(
	requirements *config.CandidateRequirements,
	model string,
	apiFormat string,
	demand selection.CandidateDemand,
) error {
	format, err := wireFormatForModel(apiFormat)
	if err != nil {
		return fmt.Errorf("%w: %w", selection.ErrNoEligibleCandidates, err)
	}
	supported, ok := r.codecCapabilitiesForFormat(format)
	if !ok {
		return fmt.Errorf("%w: model %q has no request codec", selection.ErrNoEligibleCandidates, model)
	}
	return selection.ValidateCandidateCodec(requirements, model, supported, demand)
}
