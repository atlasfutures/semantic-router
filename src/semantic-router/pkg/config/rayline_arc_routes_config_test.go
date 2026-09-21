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

package config

import "testing"

// The deadline is bounded from both ends. Longer than the encoder's own
// timeout cannot bound anything; shorter than a cold encode guarantees a 504
// on the first request after a scale-up.
func TestRoutesAPIDeadlineIsBoundedBothWays(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		deadline int
		valid    bool
	}{
		"unset takes the default": {deadline: 0, valid: true},
		"the shipped budget":      {deadline: 1500, valid: true},
		"at the floor":            {deadline: minRaylineARCRoutesDeadlineMS, valid: true},
		"below the floor":         {deadline: minRaylineARCRoutesDeadlineMS - 1, valid: false},
		"at the ceiling":          {deadline: maxRaylineARCRoutesDeadlineMS, valid: true},
		"above the ceiling":       {deadline: maxRaylineARCRoutesDeadlineMS + 1, valid: false},
		"negative":                {deadline: -1, valid: false},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			err := validateRaylineARCRoutesAPIConfig(RaylineARCRoutesAPIConfig{
				Enabled:    true,
				DeadlineMS: testCase.deadline,
			})
			if testCase.valid && err != nil {
				t.Fatalf("deadline_ms=%d rejected: %v", testCase.deadline, err)
			}
			if !testCase.valid && err == nil {
				t.Fatalf("deadline_ms=%d accepted, want rejected", testCase.deadline)
			}
		})
	}
}

// The label is published to callers and read back in support, so it stays in
// a character set that survives a URL, a log line and a shell.
func TestRoutesAPICheckpointLabelCharacterSet(t *testing.T) {
	t.Parallel()
	for label, valid := range map[string]bool{
		"":               true,
		"arc-c82-dev":    true,
		"arc_2026_09_12": true,
		"a1":             true,
		"Arc-Dev":        false,
		"arc dev":        false,
		"arc/dev":        false,
		"-leading-dash":  false,
	} {
		t.Run(label, func(t *testing.T) {
			t.Parallel()
			err := validateRaylineARCRoutesAPIConfig(RaylineARCRoutesAPIConfig{
				Enabled:         true,
				CheckpointLabel: label,
			})
			if valid && err != nil {
				t.Fatalf("label %q rejected: %v", label, err)
			}
			if !valid && err == nil {
				t.Fatalf("label %q accepted, want rejected", label)
			}
		})
	}
}

// Zero selects the shipped default rather than meaning "no deadline", which
// is the reading that would leave a lookup inheriting the encoder's own
// minutes-long budget.
func TestRoutesAPIEffectiveDeadlineSubstitutesTheDefault(t *testing.T) {
	t.Parallel()
	unset := RaylineARCRoutesAPIConfig{Enabled: true}
	if got := unset.EffectiveDeadline().Milliseconds(); got != DefaultRaylineARCRoutesDeadlineMS {
		t.Fatalf("EffectiveDeadline() = %dms, want the %dms default", got, DefaultRaylineARCRoutesDeadlineMS)
	}
	set := RaylineARCRoutesAPIConfig{Enabled: true, DeadlineMS: 800}
	if got := set.EffectiveDeadline().Milliseconds(); got != 800 {
		t.Fatalf("EffectiveDeadline() = %dms, want 800ms", got)
	}
}

// include_tool_names changes what the selector is asked on every routed turn,
// so its zero value has to be off.
func TestToolNamesAndRoutesAPIDefaultOff(t *testing.T) {
	t.Parallel()
	var zero RaylineARCAlgorithmConfig
	if zero.IncludeToolNames {
		t.Fatal("include_tool_names defaults on, want off")
	}
	if zero.RoutesAPI.Enabled {
		t.Fatal("routes_api defaults on, want off")
	}
	if zero.RoutesAPI.EpisodeWrites {
		t.Fatal("episode_writes defaults on, want off")
	}
}
