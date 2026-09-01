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

import (
	"os"
	"testing"
)

// TestConfiguredPrivateRuntime is an opt-in deployment preflight. Public CI
// has no private artifact and skips it; a launch packet sets the environment
// variable after staging its generated worker-double runtime.
func TestConfiguredPrivateRuntime(t *testing.T) {
	runtimeDir := os.Getenv("RAYLINE_ARC_PRIVATE_ARTIFACT_DIR")
	if runtimeDir == "" {
		t.Skip("RAYLINE_ARC_PRIVATE_ARTIFACT_DIR is not configured")
	}
	runtime, err := LoadRuntime(runtimeDir)
	if err != nil {
		t.Fatalf("load configured private ARC runtime: %v", err)
	}
	if len(runtime.WorkerIDs()) < 2 {
		t.Fatal("configured private ARC runtime has fewer than two workers")
	}
	contract := runtime.EncoderContract()
	if contract.Model == "" || contract.Revision == "" ||
		contract.Dimension <= 0 || contract.Serialization != SerializationName {
		t.Fatalf("configured private ARC encoder contract is invalid: %#v", contract)
	}
	parity := runtime.HeadParity()
	if parity.Cases <= 0 || parity.SelectionParity != 1 {
		t.Fatalf("configured private ARC head parity is invalid: %#v", parity)
	}
}
