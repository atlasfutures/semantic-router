package config

import (
	"strings"
	"testing"
)

// The readiness wait is bounded at load: a wait no shipped ext_proc
// message_timeout could cover is refused.
func TestRaylineARCReadinessWaitIsBoundedAtLoad(t *testing.T) {
	err := validateRaylineARCAlgorithmConfig(&RaylineARCAlgorithmConfig{ReadinessWaitSeconds: maxRaylineARCReadinessWaitSeconds + 1})
	if err == nil || !strings.Contains(err.Error(), "readiness_wait_seconds cannot exceed 300") {
		t.Fatalf("a %d s wait loaded: %v", maxRaylineARCReadinessWaitSeconds+1, err)
	}
	if DefaultRaylineARCReadinessWaitSeconds >= 60 {
		t.Fatalf("the default wait %d s reaches the shipped 60 s ext_proc message_timeout", DefaultRaylineARCReadinessWaitSeconds)
	}
}
