package config

import (
	"encoding/json"
	"testing"
)

// A model-unit stage one offers candidates with no effort; any effort, a
// missing offer or an unreadable part is not one (memex-desktop #7417).
func TestStageOneByModelReadsTheOfferedCandidates(t *testing.T) {
	for raw, want := range map[string]bool{
		`{"offered":[{"model":"a","effort":null},{"model":"b","effort":null}]}`:   true,
		`{"offered":[{"model":"a","effort":"high"},{"model":"b","effort":null}]}`: false,
		`{"offered":[]}`:                           false,
		`{"boundaries":["task_start"]}`:            false,
		`{"offered":[{"model":"","effort":null}]}`: false,
		`[1,2]`: false,
	} {
		if got := stageOneByModel(json.RawMessage(raw)); got != want {
			t.Errorf("stageOneByModel(%s) = %v, want %v", raw, got, want)
		}
	}
	if stageOneByModel(nil) {
		t.Error("an absent stage_one is not a model-unit stage one")
	}
}
