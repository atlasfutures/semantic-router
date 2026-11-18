package extproc

import (
	"maps"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Every lookup whose body and routing block are read is counted once by the
// routing.execution envelope it carries, whatever it is then answered, and
// an answered lookup's routing_decision event names the same version. A
// lookup whose routing block is refused as invalid is not counted.
func TestRaylineRoutesCountLookupsByExecutionVersion(t *testing.T) {
	var v1, v2 string
	for _, vector := range readExecutionVectors(t).Vectors {
		if vector.Expect.Refused == "" {
			v1 = string(vector.Execution)
			break
		}
	}
	for _, vector := range readExecutionV2Vectors(t).Vectors {
		if vector.Expect.Refused == "" {
			v2 = string(vector.Execution)
			break
		}
	}
	if v1 == "" || v2 == "" {
		t.Fatal("no accepted execution vector")
	}
	plain := []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"hi"}]}`)
	for _, tc := range []struct {
		name, want string
		body       []byte
		// routing serves routing controls (routes_api.routing); a cell
		// without it refuses any body that carries a routing block.
		routing bool
	}{
		{"cell without routing controls", "none", plain, false},
		{"no routing block", "none", plain, true},
		{"routing without execution", "none", []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"hi"}],"routing":{"value_usd":1}}`), true},
		{"v1", "v1", executionBody(v1), true},
		{"v2", "v2", executionV2Body(`{"execution":` + v2 + `}`), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, _ := oneCellRouter(t, "main")
			if tc.routing {
				router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Routing = &config.RaylineARCRoutesRoutingConfig{}
			}
			want := lookupCounts(t)
			want[tc.want]++
			logs := captureLogs(t)
			router.handleRaylineRoutesAPI(tc.body, routesContext(nil))
			router.background.Wait()
			if got := lookupCounts(t); !maps.Equal(got, want) {
				t.Fatalf("counted %v, want %v", got, want)
			}
			if line := findLogEvent(t, logs, "routing_decision"); line["routing_execution"] != tc.want {
				t.Fatalf("routing_decision = %v", line)
			}
		})
	}

	// Refused later, for something other than the routing block: still
	// counted, by the envelope the lookup carried.
	for name, tc := range map[string]struct {
		shape, want string
		body        []byte
		status      int
	}{
		"unknown price model": {"main", "v2", executionV2Body(`{"execution":` + v2 +
			`,"prices":{"acme/x":{"input":1,"cache_read":0,"cache_write":0,"output":1}}}`), 400},
		"tier the cell does not serve": {"sage", "none", []byte(`{"model":"arc-swift-1.0","messages":[{"role":"user","content":"hi"}]}`), 404},
		"message the codec refuses": {"main", "v1", []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"a"},` +
			`{"role":"assistant","content":"b"},{"role":"user","content":"c"},{"role":"assistant","content":"d"},` +
			`{"role":"user","content":[{"type":"no_such_block"}]}],"routing":{"execution":` + v1 + `}}`), 400},
	} {
		t.Run(name, func(t *testing.T) {
			router, _ := oneCellRouter(t, tc.shape)
			router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Routing = &config.RaylineARCRoutesRoutingConfig{}
			want := lookupCounts(t)
			want[tc.want]++
			if status := immediateStatusCode(t, router.handleRaylineRoutesAPI(tc.body, routesContext(nil))); status != tc.status {
				t.Fatalf("answered %d, want %d", status, tc.status)
			}
			if got := lookupCounts(t); !maps.Equal(got, want) {
				t.Fatalf("counted %v, want %v", got, want)
			}
		})
	}

	t.Run("invalid execution envelope", func(t *testing.T) {
		router, _ := oneCellRouter(t, "main")
		router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Routing = &config.RaylineARCRoutesRoutingConfig{}
		before := lookupCounts(t)
		if status := immediateStatusCode(t, router.handleRaylineRoutesAPI(executionBody(`{"version":"arc-execution-context.v1"}`), routesContext(nil))); status != 400 {
			t.Fatalf("answered %d, want 400", status)
		}
		if after := lookupCounts(t); !maps.Equal(after, before) {
			t.Fatalf("a refused routing block was counted: %v -> %v", before, after)
		}
	})
}

// lookupCounts reads every series of the lookup counter, whatever its label.
func lookupCounts(t *testing.T) map[string]float64 {
	t.Helper()
	series := make(chan prometheus.Metric)
	go func() {
		metrics.RaylineARCRoutesLookups.Collect(series)
		close(series)
	}()
	counts := map[string]float64{}
	for metric := range series {
		var sample dto.Metric
		if err := metric.Write(&sample); err != nil {
			t.Fatal(err)
		}
		for _, label := range sample.GetLabel() {
			if label.GetName() == "execution" {
				counts[label.GetValue()] = sample.GetCounter().GetValue()
			}
		}
	}
	return counts
}
