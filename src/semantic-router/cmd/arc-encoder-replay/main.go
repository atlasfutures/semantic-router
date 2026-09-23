// Command arc-encoder-replay replays a prod-shaped ARC turn trace against a
// set of retained-session encoder replicas through raylinearc.EncoderPool, the
// same client, rendezvous placement and single-remap failover the router uses.
// It measures the encoder fleet alone: no provider call and no admission gate.
//
// Trace: JSONL of {"conv": int, "t_ms": int, "prompt": int}, sorted by t_ms.
// Each conversation is one episode. A turn's encoder context is
// min(prompt*ratio, max) tokens of deterministic per-episode text; a turn
// that grows the context appends to the history, and a turn that shrinks it
// (compaction) starts a new history, which the encoder must rebuild.
//
// Credentials come from RAYLINE_ARC_MODAL_KEY and RAYLINE_ARC_MODAL_SECRET.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type traceTurn struct {
	Conv   int   `json:"conv"`
	TMS    int64 `json:"t_ms"`
	Prompt int   `json:"prompt"`
}

type outcome struct {
	Conv         int     `json:"conv"`
	OffsetS      float64 `json:"offset_s"`
	LagS         float64 `json:"lag_s"`
	Seconds      float64 `json:"seconds"`
	TargetTokens int     `json:"target_tokens"`
	Serialized   int     `json:"serialized_tokens"`
	Appended     int     `json:"appended_tokens"`
	Action       string  `json:"session_action"`
	Replica      int     `json:"replica_index"`
	Failover     bool    `json:"failover"`
	Error        string  `json:"error,omitempty"`
	ErrorClass   string  `json:"error_class,omitempty"`
}

// words is a fixed vocabulary; ~1.3 Qwen tokens per word on average.
var words = strings.Fields(`system request response handler module config
router encoder session token context history buffer cache queue worker
latency throughput replica cluster deploy region service endpoint payload
schema migration index query transaction commit rollback snapshot lease
fence retry backoff timeout circuit budget metric alert trace span log
function parameter return value error panic recover channel select mutex
file path directory stream reader writer parse format encode decode vector
matrix tensor kernel batch prefill decode attention layer weight gradient`)

const tokensPerWord = 1.3

type episode struct {
	hash     string
	rng      *rand.Rand
	history  []raylinearc.Turn
	tokens   int
	affinity raylinearc.EncoderAffinity
	mu       sync.Mutex
}

func (e *episode) text(tokens int) string {
	n := int(float64(tokens)/tokensPerWord) + 1
	var b strings.Builder
	b.Grow(n * 8)
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(words[e.rng.Intn(len(words))])
	}
	return b.String()
}

func main() {
	tracePath := flag.String("trace", "", "trace JSONL")
	replicasFlag := flag.String("replicas", "", "comma-separated id=url")
	speed := flag.Float64("speed", 1, "time compression factor")
	duration := flag.Duration("duration", 5*time.Minute, "replay window (wall clock)")
	ratio := flag.Float64("ratio", 0.6, "encoder tokens per provider prompt token")
	maxTokens := flag.Int("max-tokens", 240000, "cap on encoder context tokens")
	label := flag.String("label", "run", "episode id namespace")
	out := flag.String("out", "", "per-turn JSONL output")
	flag.Parse()

	turns := readTrace(*tracePath)
	pool := buildPool(*replicasFlag)
	episodes := map[int]*episode{}
	for _, t := range turns {
		if _, ok := episodes[t.Conv]; !ok {
			episodes[t.Conv] = &episode{
				hash: raylinearc.HashEpisodeID(fmt.Sprintf("replay-%s-%d", *label, t.Conv)),
				rng:  rand.New(rand.NewSource(int64(t.Conv) + 7)),
			}
		}
	}

	var (
		results []outcome
		resMu   sync.Mutex
		wg      sync.WaitGroup
	)
	start := time.Now()
	windowMS := int64(float64(duration.Milliseconds()) * *speed)
	for _, t := range turns {
		if t.TMS > windowMS {
			break
		}
		due := start.Add(time.Duration(float64(t.TMS)/(*speed)) * time.Millisecond)
		time.Sleep(time.Until(due))
		wg.Add(1)
		go func(t traceTurn, due time.Time) {
			defer wg.Done()
			ep := episodes[t.Conv]
			// A conversation's turns are serial, as in production.
			ep.mu.Lock()
			defer ep.mu.Unlock()
			target := int(float64(t.Prompt) * *ratio)
			if target > *maxTokens {
				target = *maxTokens
			}
			if target < 64 {
				target = 64
			}
			if target < ep.tokens || len(ep.history) == 0 {
				ep.history = []raylinearc.Turn{{Role: "user", Text: ep.text(target)}}
			} else if delta := target - ep.tokens; delta > 0 {
				ep.history = append(ep.history,
					raylinearc.Turn{Role: "assistant", Text: ep.text(delta / 2)},
					raylinearc.Turn{Role: "user", Text: ep.text(delta - delta/2)})
			} else {
				ep.history = append(ep.history, raylinearc.Turn{Role: "user", Text: ep.text(16)})
			}
			ep.tokens = target
			began := time.Now()
			ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
			defer cancel()
			result, err := pool.EncodeWithAffinity(ctx, ep.hash, ep.history, ep.affinity)
			o := outcome{
				Conv: t.Conv, OffsetS: began.Sub(start).Seconds(), LagS: began.Sub(due).Seconds(),
				Seconds: time.Since(began).Seconds(), TargetTokens: target, Replica: -1,
			}
			if err != nil {
				o.Error = err.Error()
				if f, ok := err.(*raylinearc.EncoderFailure); ok {
					o.ErrorClass = string(f.Class)
				}
			} else {
				o.Serialized, o.Appended = result.SerializedTokens, result.AppendedTokens
				o.Action, o.Replica, o.Failover = result.SessionAction, result.ReplicaIndex, result.ReplicaFailover
				ep.affinity = raylinearc.EncoderAffinity{Owner: result.ReplicaID, Visited: result.VisitedReplicaIDs}
			}
			resMu.Lock()
			results = append(results, o)
			resMu.Unlock()
		}(t, due)
	}
	wg.Wait()
	elapsed := time.Since(start).Seconds()
	// Close every session so the next run starts with free encoder lanes.
	for _, ep := range episodes {
		if len(ep.affinity.Visited) == 0 {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		_, _ = pool.CloseSession(ctx, ep.hash, ep.affinity.Visited)
		cancel()
	}
	writeResults(*out, results)
	summarize(results, elapsed, *speed)
}

func readTrace(path string) []traceTurn {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	var turns []traceTurn
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var t traceTurn
		if err := json.Unmarshal([]byte(line), &t); err != nil {
			panic(err)
		}
		turns = append(turns, t)
	}
	sort.SliceStable(turns, func(i, j int) bool { return turns[i].TMS < turns[j].TMS })
	return turns
}

func buildPool(spec string) *raylinearc.EncoderPool {
	var replicas []raylinearc.EncoderReplica
	for _, part := range strings.Split(spec, ",") {
		id, url, ok := strings.Cut(part, "=")
		if !ok {
			panic("replica must be id=url: " + part)
		}
		client, err := raylinearc.NewEncoderClient(raylinearc.EncoderClientConfig{
			BaseURL:               url,
			Model:                 "Qwen/Qwen3.5-0.8B",
			ModelRevision:         "2fc06364715b967f1860aea9cf38778875588b17",
			TokenizerRevision:     "2fc06364715b967f1860aea9cf38778875588b17",
			TokenizerSHA256:       raylinearc.EncoderTokenizerSHA256,
			EOSTokenID:            raylinearc.EncoderEOSTokenID,
			ExpectedBuildID:       "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca+gdn-flashinfer-eager",
			ExpectedPluginVersion: "rayline-arc-io@0.1.0",
			SerializerVersion:     "mtrouter-token-blocks-v2",
			ServingRung:           "B",
			RequiredCapabilities:  []string{"chunked_causal_mean", "resumable_causal_mean"},
			RetainedSession:       true,
			ModalKey:              os.Getenv("RAYLINE_ARC_MODAL_KEY"),
			ModalSecret:           os.Getenv("RAYLINE_ARC_MODAL_SECRET"),
			ConnectTimeout:        5 * time.Second,
			TotalTimeout:          600 * time.Second,
		})
		if err != nil {
			panic(err)
		}
		replicas = append(replicas, raylinearc.EncoderReplica{ID: id, State: raylinearc.EncoderReplicaActive, Client: client})
	}
	pool, err := raylinearc.NewEncoderPool(replicas, raylinearc.EncoderPoolConfig{
		SchemaVersion:          raylinearc.EncoderFailoverSchemaV1,
		UnavailableStatusCodes: []int{404, 502, 503, 504},
		UnavailableCooldown:    30 * time.Second,
		MaxRemaps:              1,
	})
	if err != nil {
		panic(err)
	}
	return pool
}

func writeResults(path string, results []outcome) {
	if path == "" {
		return
	}
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, o := range results {
		_ = enc.Encode(o)
	}
}

func pct(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sort.Float64s(values)
	return values[int(p*float64(len(values)-1))]
}

func summarize(results []outcome, elapsed, speed float64) {
	var lat, lag []float64
	ok, failed := 0, 0
	actions := map[string]int{}
	classes := map[string]int{}
	perReplica := map[int]int{}
	for _, o := range results {
		lag = append(lag, o.LagS+o.Seconds)
		if o.Error != "" {
			failed++
			classes[o.ErrorClass]++
			continue
		}
		ok++
		lat = append(lat, o.Seconds)
		actions[o.Action]++
		perReplica[o.Replica]++
	}
	summary := map[string]any{
		"speed": speed, "turns": len(results), "ok": ok, "failed": failed,
		"offered_per_s": float64(len(results)) / elapsed, "completed_per_s": float64(ok) / elapsed,
		"encode_s":            map[string]float64{"p50": pct(lat, .5), "p95": pct(lat, .95), "p99": pct(lat, .99), "max": pct(lat, 1)},
		"queue_plus_encode_s": map[string]float64{"p50": pct(lag, .5), "p95": pct(lag, .95), "max": pct(lag, 1)},
		"actions":             actions, "error_classes": classes, "per_replica": perReplica,
	}
	data, _ := json.Marshal(summary)
	fmt.Println(string(data))
}
