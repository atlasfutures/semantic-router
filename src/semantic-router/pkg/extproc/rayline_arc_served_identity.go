package extproc

import (
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// raylineARCServedIdentity is what a gateway needs to bill a Rayline ARC turn
// at the model that served it (configurable ARC, arc-1.0: the served model's
// own catalog card, no ARC fee): the provider model id the turn was
// dispatched to, the policy package that decided it and the cost weight it
// was decided at. Every member is a fact the router holds; a member it does
// not hold stays empty and its header is not sent.
type raylineARCServedIdentity struct {
	// model is the provider model id read back off the final provider body
	// (DispatchedOutbound.Model), not the arm name: an arm is named by the
	// package (kimi-k3@thinking-off), and its rate card by the provider id
	// (moonshotai/kimi-k3). A flex re-send dispatches the same provider
	// model (flexResendPlan refuses a twin of another), so it names the
	// served model either way.
	model         string
	packageSHA256 string
	costWeight    *float64
}

// raylineARCServedIdentityOf reads the identity of an ARC turn that reached
// its provider. A turn outside ARC, or one that never dispatched a body,
// has none.
func raylineARCServedIdentityOf(ctx *RequestContext) (raylineARCServedIdentity, bool) {
	if ctx == nil || ctx.VSRRaylineARC == nil || ctx.DispatchedOutbound == nil {
		return raylineARCServedIdentity{}, false
	}
	model := strings.TrimSpace(ctx.DispatchedOutbound.Model)
	if model == "" {
		return raylineARCServedIdentity{}, false
	}
	trace := ctx.VSRRaylineARC
	identity := raylineARCServedIdentity{model: model, packageSHA256: trace.PolicyPackageSHA256}
	if weight := trace.PolicyCostWeight; weight != nil {
		echoed := *weight
		identity.costWeight = &echoed
	}
	return identity, true
}

// addRaylineARCServedIdentity publishes the served identity of an ARC turn:
// x-vsr-served-model, x-vsr-policy-package and x-vsr-policy-cost-weight.
func (builder *responseHeaderMutationBuilder) addRaylineARCServedIdentity(ctx *RequestContext) {
	identity, ok := raylineARCServedIdentityOf(ctx)
	if !ok {
		return
	}
	builder.addString(headers.VSRServedModel, identity.model)
	builder.addString(headers.VSRPolicyPackage, identity.packageSHA256)
	if identity.costWeight != nil && *identity.costWeight >= 0 {
		// Full precision: V = 1 / cost weight, and 0.2222 is not 1/4.5.
		builder.addString(headers.VSRPolicyCostWeight, strconv.FormatFloat(*identity.costWeight, 'g', -1, 64))
	}
}

// addRaylineARCServedEndpoint names the endpoint a successful ARC turn was
// served by (x-vsr-served-endpoint), when its call was pinned to one.
func (builder *responseHeaderMutationBuilder) addRaylineARCServedEndpoint(ctx *RequestContext) {
	if _, ok := raylineARCServedIdentityOf(ctx); !ok {
		return
	}
	builder.addString(headers.VSRServedEndpoint, ctx.RaylineARCDispatchedEndpoint)
}
