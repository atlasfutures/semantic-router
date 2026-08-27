package protocolconformance

import "github.com/vllm-project/semantic-router/e2e/pkg/framework"

// LocalImages returns the fixture images required by this profile.
//
// The build context is the e2e module root because the image carries both the
// fixture binary and the fixture tree: a replay script's file references resolve
// on the fixture's own filesystem, not on the test runner's.
func LocalImages() []framework.LocalImageBuild {
	return []framework.LocalImageBuild{
		{
			Dockerfile:   "e2e/testing/conformance-fixture/Dockerfile",
			Tag:          "ghcr.io/vllm-project/semantic-router/conformance-fixture:e2e-test",
			BuildContext: "e2e",
		},
	}
}
