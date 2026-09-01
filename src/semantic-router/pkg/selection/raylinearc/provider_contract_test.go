/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import "testing"

func TestManifestAcceptsBoundedProviderOrderWithoutAutomaticFallback(t *testing.T) {
	manifest := syntheticManifest()
	manifest.Workers[0].OpenRouterProviderOrder = []string{
		manifest.Workers[0].OpenRouterProviderSlug,
		"backup-provider",
	}
	manifest.Workers[0].OpenRouterAllowFallbacks = false

	if err := validateWorkerProviderContract(&manifest.Workers[0]); err != nil {
		t.Fatalf("bounded provider order was rejected: %v", err)
	}
}
