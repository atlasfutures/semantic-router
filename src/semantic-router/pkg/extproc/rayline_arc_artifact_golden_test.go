//go:build !windows && cgo

package extproc

import "math"

// fixtureArmAxis is the value one arm embedding lands on after layer
// normalization, given a pre-norm vector of one 1 and the rest zeros. The
// golden scores are written in terms of it.
func fixtureArmAxis() float64 {
	mean := 1.0 / float64(fixtureArmDimension)
	variance := ((1-mean)*(1-mean) + float64(fixtureArmDimension-1)*mean*mean) /
		float64(fixtureArmDimension)
	return (1 - mean) / math.Sqrt(variance+1e-5)
}

type fixtureGoldenCase struct {
	ID                 string    `json:"id"`
	Embedding          []float32 `json:"embedding"`
	PreviousModelIndex int64     `json:"previous_model_index"`
	RouteCallIndex     uint64    `json:"route_call_index"`
	Scores             []float32 `json:"scores"`
	SelectedIndex      int       `json:"selected_index"`
}

type fixtureGoldenFile struct {
	SchemaVersion    string              `json:"schema_version"`
	CheckpointSHA256 string              `json:"checkpoint_sha256"`
	ScoreTolerance   float32             `json:"score_tolerance"`
	Pool             []string            `json:"pool"`
	Cases            []fixtureGoldenCase `json:"cases"`
}

// The golden scores are derived, not measured.
//
// With these weights the head collapses to one expression. The history vector
// is L2-normalized and carries a single 1 on the routing axis; the first
// backbone row sums that axis and the candidate arm's first component, the
// second row negates the same sum, and the two ReLU rows are added back
// together at the head, which yields |history[routingAxis] + armAxis|. The arm
// component is +armAxis for the first worker and -armAxis for the second,
// because their residual rows are +1 and -1, so the two scores are armAxis+1
// and armAxis-1, and the larger one names the selected arm. fixtureArmAxis is
// the post-LayerNorm value of a vector holding a single 1, which is why the
// closed form needs no floating-point replay of the network.
func fixtureGolden(checkpointDigest string) fixtureGoldenFile {
	arm := float32(fixtureArmAxis())
	return fixtureGoldenFile{
		SchemaVersion:    "rayline.mtrouter-head-golden.v1",
		CheckpointSHA256: checkpointDigest,
		ScoreTolerance:   0.001,
		Pool:             []string{fixtureWorkerA, fixtureWorkerB},
		Cases: []fixtureGoldenCase{
			{
				ID:                 "route-a",
				Embedding:          fixtureEmbedding(1),
				PreviousModelIndex: -1,
				RouteCallIndex:     0,
				Scores:             []float32{arm + 1, arm - 1},
				SelectedIndex:      0,
			},
			{
				ID:                 "route-b",
				Embedding:          fixtureEmbedding(-1),
				PreviousModelIndex: -1,
				RouteCallIndex:     0,
				Scores:             []float32{arm - 1, arm + 1},
				SelectedIndex:      1,
			},
		},
	}
}

func fixtureEmbedding(sign float32) []float32 {
	embedding := make([]float32, fixtureHistoryDimension)
	embedding[fixtureRoutingAxis] = sign
	return embedding
}
