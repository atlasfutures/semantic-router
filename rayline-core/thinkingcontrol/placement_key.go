package thinkingcontrol

import "fmt"

// PlacementKey names a placer: one worker's transcript in one format. It
// names no control: the items a placer wrote are part of the transcript the
// provider has cached, so every action the policy draws on that worker,
// whatever its base, budget or lever presence, must resume the same placer
// and replay the same items. A placer per control shape renders one
// conversation as several transcripts that diverge at the first item, and
// the provider's prompt cache misses from there on every switch.
func PlacementKey(worker, format string) string {
	return fmt.Sprintf("%s|%s", worker, format)
}
