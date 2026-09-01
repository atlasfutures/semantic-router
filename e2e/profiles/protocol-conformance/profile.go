package protocolconformance

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/protocol-conformance/values.yaml"

var (
	resourceManifests = []string{
		"e2e/profiles/protocol-conformance/gateway-resources/backend.yaml",
		"e2e/profiles/protocol-conformance/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "conformance-fixture-system", Name: "conformance-fixture"},
	}
)

// Profile implements the protocol-conformance test profile.
//
// It replaces the usual model backend with the programmable provider fixture, so
// a test can state exactly what the provider must observe and exactly what it
// replays. That is what makes both wire boundaries assertable at once: the request
// the router emitted, and the response the router returned.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new protocol-conformance profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "protocol-conformance",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "protocol-conformance"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Runs the versioned protocol-conformance corpus against a programmable provider fixture and compares both wire boundaries"
}

// Setup deploys the shared gateway stack and the provider fixture.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and the provider fixture.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.ProtocolConformanceContract
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
