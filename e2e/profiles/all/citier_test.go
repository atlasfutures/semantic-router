package all

import (
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
)

// TestCITiersNameRunnableWork keeps the CI tier table honest against the live
// registries. CI derives its -tests argument from this table, so a renamed
// testcase or a retired profile would otherwise only surface as a failed CI job.
func TestCITiersNameRunnableWork(t *testing.T) {
	cadences := []testmatrix.Cadence{testmatrix.CadencePR, testmatrix.CadenceNightly}

	for _, profile := range testmatrix.CITierProfiles() {
		if _, ok := framework.LookupProfileRegistration(profile); !ok {
			t.Errorf("CI tier table names unregistered profile %q", profile)
			continue
		}
		for _, cadence := range cadences {
			for _, name := range testmatrix.CITier(profile, cadence) {
				if _, ok := pkgtestcases.Get(name); !ok {
					t.Errorf("profile %q %s tier names unregistered test case %q", profile, cadence, name)
				}
			}
		}
	}
}

// TestPRTierIsNeverWiderThanNightly pins the direction of the split: a pull
// request may run less than the scheduled run, never more.
func TestPRTierIsNeverWiderThanNightly(t *testing.T) {
	for _, profile := range testmatrix.CITierProfiles() {
		pr := testmatrix.CITier(profile, testmatrix.CadencePR)
		nightly := testmatrix.CITier(profile, testmatrix.CadenceNightly)
		if nightly != nil && len(pr) > len(nightly) {
			t.Errorf("profile %q runs %d cases on a PR but %d nightly", profile, len(pr), len(nightly))
		}
	}
}
