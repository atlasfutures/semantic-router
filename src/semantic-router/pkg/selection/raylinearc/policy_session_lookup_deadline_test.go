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
	"context"
	"errors"
	"testing"
	"time"
)

// A lookup binding past its deadline starts no session and sends no decide,
// and says so with ErrPolicyLookupDeadline.
func TestLookupBindingRefusesPastItsDeadline(t *testing.T) {
	backend := &PolicySessionBackend{config: PolicySessionConfig{IdleTimeout: time.Minute}}
	binding := backend.BindLookup(nil, time.Now().Add(-time.Millisecond))
	if _, err := binding.Decide(context.Background(), PolicyDecisionRequest{}); !errors.Is(err, ErrPolicyLookupDeadline) {
		t.Fatalf("err = %v, want the lookup deadline", err)
	}
	if binding.current != nil || binding.sessions != 0 {
		t.Fatal("a session was started past the deadline")
	}
}
