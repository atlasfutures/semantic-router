package config

// EmbeddingAPIConfig prepares the globally configured embedding for diagnostics
// even when no routing recipe needs it. Explicit recipe requests stay isolated.
type EmbeddingAPIConfig struct {
	Enabled bool `yaml:"enabled"`
}

type APIConfig struct {
	Embeddings          EmbeddingAPIConfig        `yaml:"embeddings,omitempty"`
	BatchClassification BatchClassificationConfig `yaml:"batch_classification"`
	RoutingPreview      RoutingPreviewConfig      `yaml:"routing_preview,omitempty"`
}

type ObservabilityConfig struct {
	Tracing      TracingConfig      `yaml:"tracing"`
	Metrics      MetricsConfig      `yaml:"metrics"`
	Profiling    ProfilingConfig    `yaml:"profiling"`
	UsageRecords UsageRecordsConfig `yaml:"usage_records,omitempty"`
}

const (
	// DefaultUsageRecordsStream is the Redis stream usage records are added
	// to when none is named.
	DefaultUsageRecordsStream = "vsr:llm_usage"
	// DefaultUsageRecordsMaxLen bounds the stream when no bound is named. The
	// trim is approximate, so the stream may briefly hold a little more.
	DefaultUsageRecordsMaxLen = 1000000
	// DefaultUsageRecordsQueueSize bounds the records waiting to be written.
	DefaultUsageRecordsQueueSize = 4096
)

// UsageRecordsConfig sends every llm_usage record to a durable Redis stream
// as well as the log. It is off unless redis.address is set.
//
// The stream is as durable as the Redis behind it: enable AOF or snapshots
// there. Records go through a bounded queue, so a slow or absent Redis never
// delays a response; a record that cannot be written is counted and remains
// on the log line.
type UsageRecordsConfig struct {
	Redis UsageRecordsRedisConfig `yaml:"redis,omitempty"`
}

// UsageRecordsRedisConfig carries non-secret connection settings. The password
// is read only from PasswordEnv so serialized config never holds it.
type UsageRecordsRedisConfig struct {
	Address     string `yaml:"address,omitempty"`
	DB          int    `yaml:"db,omitempty"`
	PasswordEnv string `yaml:"password_env,omitempty"`
	UseTLS      bool   `yaml:"use_tls,omitempty"`
	Stream      string `yaml:"stream,omitempty"`
	MaxLen      int64  `yaml:"max_len,omitempty"`
	QueueSize   int    `yaml:"queue_size,omitempty"`
}

// Enabled reports whether usage records go to Redis.
func (cfg UsageRecordsConfig) Enabled() bool {
	return cfg.Redis.Address != ""
}

type MetricsConfig struct {
	Enabled         *bool                 `yaml:"enabled,omitempty"`
	WindowedMetrics WindowedMetricsConfig `yaml:"windowed_metrics"`
}

const (
	// DefaultProfilingPort is the port the pprof listener uses when profiling is
	// enabled without an explicit port.
	DefaultProfilingPort = 6060
	// DefaultProfilingBind keeps pprof on loopback unless deliberately widened.
	DefaultProfilingBind = "127.0.0.1"
)

// ProfilingConfig controls the optional in-process pprof HTTP listener. It is
// disabled by default and binds to loopback so profiles are never exposed on a
// routable interface without an explicit operator decision.
type ProfilingConfig struct {
	Enabled bool   `yaml:"enabled"`
	Port    int    `yaml:"port,omitempty"`
	Bind    string `yaml:"bind,omitempty"`
}

type WindowedMetricsConfig struct {
	Enabled        bool     `yaml:"enabled"`
	TimeWindows    []string `yaml:"time_windows,omitempty"`
	UpdateInterval string   `yaml:"update_interval,omitempty"`
	MaxModels      int      `yaml:"max_models,omitempty"`
}

type TracingConfig struct {
	Enabled  bool                  `yaml:"enabled"`
	Provider string                `yaml:"provider,omitempty"`
	Exporter TracingExporterConfig `yaml:"exporter"`
	Sampling TracingSamplingConfig `yaml:"sampling"`
	Resource TracingResourceConfig `yaml:"resource"`
}

type TracingExporterConfig struct {
	Type     string `yaml:"type"`
	Endpoint string `yaml:"endpoint,omitempty"`
	Insecure bool   `yaml:"insecure"`
}

type TracingSamplingConfig struct {
	Type string  `yaml:"type"`
	Rate float64 `yaml:"rate"`
}

type TracingResourceConfig struct {
	ServiceName           string `yaml:"service_name"`
	ServiceVersion        string `yaml:"service_version,omitempty"`
	DeploymentEnvironment string `yaml:"deployment_environment,omitempty"`
}

type BatchClassificationMetricsConfig struct {
	SampleRate                float64                `yaml:"sample_rate,omitempty"`
	BatchSizeRanges           []BatchSizeRangeConfig `yaml:"batch_size_ranges,omitempty"`
	DurationBuckets           []float64              `yaml:"duration_buckets,omitempty"`
	SizeBuckets               []float64              `yaml:"size_buckets,omitempty"`
	Enabled                   bool                   `yaml:"enabled,omitempty"`
	DetailedGoroutineTracking bool                   `yaml:"detailed_goroutine_tracking,omitempty"`
	HighResolutionTiming      bool                   `yaml:"high_resolution_timing,omitempty"`
}

type BatchSizeRangeConfig struct {
	Min   int    `yaml:"min"`
	Max   int    `yaml:"max"`
	Label string `yaml:"label"`
}
