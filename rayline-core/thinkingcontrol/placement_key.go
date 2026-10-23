package thinkingcontrol

import (
	"fmt"
	"strconv"
)

// PlacementKey names a placer: one worker's transcript in one format
// under one control shape. Pathfinder fixes base, budget and lever presence
// per episode, so a shape change on a worker starts its own placer.
func PlacementKey(worker, format string, control Control) string {
	budget := "-"
	if control.BudgetTokens != nil {
		budget = strconv.FormatInt(*control.BudgetTokens, 10)
	}
	return fmt.Sprintf("%s|%s|%s|%s|%t", worker, format, control.Native, budget, control.Instruction != nil)
}
