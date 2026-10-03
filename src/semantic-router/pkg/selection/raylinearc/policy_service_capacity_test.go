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

// A capacity refusal's detail keeps only the contract's two counts, as
// non-negative integers; free-form service text never travels further.
func TestCapacityRefusalDetailKeepsOnlyItsCounts(t *testing.T) {
	kept := capacityRefusalDetail(map[string]any{
		"token_count": 300000.0, "max_tokens": 262144.0, "note": "x",
	})
	if len(kept) != 2 || kept["token_count"] != 300000 || kept["max_tokens"] != 262144 {
		t.Fatalf("kept %v", kept)
	}
	if dropped := capacityRefusalDetail(map[string]any{"token_count": -1.0, "max_tokens": 1.5}); len(dropped) != 0 {
		t.Fatalf("kept invalid counts %v", dropped)
	}
}
