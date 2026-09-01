package testcases

import (
	"strings"
	"testing"
)

// The blocker this file exists for: testcases used to accept "0% only" bars,
// where a single correct case out of N satisfied a whole suite. Each test below
// pins one way that bar used to let a regression through.

func TestRequireAccuracyFloorRejectsSingleCorrectCase(t *testing.T) {
	// The exact shape DPC-106 removes: 1 of 261 correct used to pass.
	err := requireAccuracyFloor("domain classification", 1, 261, 60.0)
	if err == nil {
		t.Fatal("expected a single correct case out of 261 to fail the floor")
	}
	if !strings.Contains(err.Error(), "1/261") {
		t.Fatalf("verdict must name the counts, got %v", err)
	}
	if !strings.Contains(err.Error(), "60.00%") {
		t.Fatalf("verdict must name the floor it missed, got %v", err)
	}
}

// An emptied or unreadable case list must not read as a vacuous pass: 0/0 is
// 0 correct out of 0 executed, which asserts nothing at all.
func TestRequireAccuracyFloorRejectsZeroCases(t *testing.T) {
	err := requireAccuracyFloor("plugin chain", 0, 0, 60.0)
	if err == nil {
		t.Fatal("expected zero executed cases to fail")
	}
	if !strings.Contains(err.Error(), "nothing was asserted") {
		t.Fatalf("zero-case verdict must say nothing was asserted, got %v", err)
	}
}

func TestRequireAccuracyFloorRejectsTotalFailure(t *testing.T) {
	if err := requireAccuracyFloor("pii detection", 0, 12, 65.0); err == nil {
		t.Fatal("expected 0/12 to fail")
	}
}

// Just below the floor must fail; the floor itself must pass. The boundary is
// the whole point of stating a minimum, so it is pinned in both directions.
func TestRequireAccuracyFloorBoundary(t *testing.T) {
	if err := requireAccuracyFloor("jailbreak detection", 6, 10, 70.0); err == nil {
		t.Fatal("expected 60% to fail a 70% floor")
	}
	if err := requireAccuracyFloor("jailbreak detection", 7, 10, 70.0); err != nil {
		t.Fatalf("expected exactly the floor to pass, got %v", err)
	}
}

func TestRequireAccuracyFloorAcceptsFullPass(t *testing.T) {
	if err := requireAccuracyFloor("rule conditions", 6, 6, 80.0); err != nil {
		t.Fatalf("expected a full pass to succeed, got %v", err)
	}
}

func TestRequireAccuracyFloorNamesTheTestcase(t *testing.T) {
	err := requireAccuracyFloor("mcp http classification", 1, 10, 60.0)
	if err == nil || !strings.Contains(err.Error(), "mcp http classification") {
		t.Fatalf("verdict must name the failing testcase, got %v", err)
	}
}
