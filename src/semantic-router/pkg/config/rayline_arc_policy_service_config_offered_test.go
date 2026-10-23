package config

import (
	"strings"
	"testing"
)

// offered_action_ids names bound actions only, each once.
func TestRaylineARCOfferedActionIDs(t *testing.T) {
	_, decision := policyDispatchFixture()
	policy := decision.Algorithm.RaylineARC.PolicyService
	bound := policy.Bindings[0].ActionID
	for name, tc := range map[string]struct {
		ids  []string
		want string
	}{
		"none":    {nil, ""},
		"bound":   {[]string{bound}, ""},
		"unbound": {[]string{strings.Repeat("f", 64)}, "no binding binds"},
		"twice":   {[]string{bound, bound}, "twice"},
	} {
		policy.OfferedActionIDs = tc.ids
		err := validateRaylineARCOfferedActionIDs(policy)
		if (tc.want == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tc.want)) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}
