package selection

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// RaylineARCSelectionContext carries only the structured, privacy-safe inputs
// required by the artifact-owned ARC selector. It is nil for every other
// algorithm.
type RaylineARCSelectionContext struct {
	EpisodeIDHash      string
	Turns              []raylinearc.Turn
	State              *raylinearc.EpisodeState
	InputTokens        int
	PreparationFailure string
}

// RaylineARCTrace records bounded, privacy-safe artifact policy diagnostics.
type RaylineARCTrace struct {
	// ArtifactID and ArtifactRevision hold SHA256-derived hashes of the
	// deployment-private artifact identity, never the raw pins.
	ArtifactID           string
	ArtifactRevision     string
	EncoderRevision      string
	EpisodeIDHash        string
	SelectedArm          int
	PreviousArm          *int
	RawScores            []float32
	AdjustedScores       []float32
	SwitchCostUSD        []float64
	CacheMissTokens      []int
	Stayed               bool
	UpgradeExemptions    []bool
	StayUpgradeExempted  bool
	SerializedTokens     int
	FullHistoryTokens    int
	TruncatedTokens      int
	CachedPrefixTokens   int
	RetainedPrefixTokens int
	AppendedTokens       int
	SessionAction        string
	SessionRevision      int
	EncoderLatency       time.Duration
	EncoderReplicaIndex  int
	EncoderAttempts      int
	EncoderFailover      bool
	// EncoderReplicaID and EncoderVisitedReplicaIDs are internal transaction
	// state. They must never be logged or exported as metric labels.
	EncoderReplicaID         string
	EncoderVisitedReplicaIDs []string
}
