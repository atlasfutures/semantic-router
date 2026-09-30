package protocolcodec

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// upstreamUnknownFieldDropped counts the provider members the fork's
// drop-and-count fallback removed (US-002). Upstream's vendor allowlist runs
// first and reports its own drops as diagnostics; what reaches this counter is
// a member no vendor rule and no named field covers. The path is a
// type-directed JSON path with array indexes collapsed, so its label set is
// bounded by what providers actually send.
var upstreamUnknownFieldDropped = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "sr_protocol_upstream_unknown_field_dropped_total",
		Help: "Provider response members dropped because the wire contract does not name them",
	},
	[]string{"vendor", "field"},
)

func countDroppedProviderFields(vendor string, paths []string) {
	if vendor == "" {
		vendor = "unlisted"
	}
	for _, path := range paths {
		upstreamUnknownFieldDropped.WithLabelValues(vendor, path).Inc()
	}
}
