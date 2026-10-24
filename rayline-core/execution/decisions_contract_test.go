package execution

import (
	"math"
	"testing"
)

func echoPtr[T any](value T) *T { return &value }
func defaultWorkerEcho() workerEcho {
	return workerEcho{ActionID: "a", Worker: "local", Held: echoPtr(false), Pricing: "default_rival_cold", Read: echoPtr(true), Basis: echoPtr("unknown"), PWarm: echoPtr(0.0), PrefixTokens: echoPtr(100), PrefixBasis: echoPtr("encoder_fallback"), NotAdmitted: []notAdmittedEcho{}}
}
func unpricedWorkerEcho(reason string) workerEcho {
	return workerEcho{ActionID: "a", Worker: "local", Pricing: "not_priced", Read: echoPtr(false), Reason: echoPtr(reason), NotAdmitted: []notAdmittedEcho{}}
}
func prefixWorker() externalWorker {
	return externalWorker{ActionID: "a", Worker: "local", CurrentPrefix: &externalPrefix{Tokens: 100, Units: "provider:local", Provenance: "local_measured"}}
}
func countedWorkerEcho() workerEcho {
	e := defaultWorkerEcho()
	e.PrefixBasis = echoPtr("worker_count")
	return e
}

func TestWorkerPricingEchoPreservesDefaultsAndUnknowns(t *testing.T) {
	workers := []externalWorker{{ActionID: "a", Worker: "local"}}
	for _, test := range []struct {
		name   string
		change func(*workerEcho)
		valid  bool
	}{
		{"read can use cold default", func(*workerEcho) {}, true},
		{"held warm default", func(e *workerEcho) { e.Held = echoPtr(true); e.Pricing = "default_held_warm"; e.PWarm = echoPtr(1.0) }, true},
		{"observed", func(e *workerEcho) { e.Pricing = "observed"; e.Basis = echoPtr("observed_ttl"); e.PWarm = echoPtr(0.5) }, true},
		{"superseded copies another action's model row", func(e *workerEcho) {
			e.Read = echoPtr(false)
			e.Reason = echoPtr("superseded")
			e.Basis = echoPtr("observed_ttl")
			e.PWarm = echoPtr(0.5)
		}, true},
		{"package default", func(e *workerEcho) { e.Read = echoPtr(false); e.Reason = echoPtr("package_mode") }, true},
		{"not in offer", func(e *workerEcho) { *e = unpricedWorkerEcho("not_in_offer") }, true},
		{"unoffered lower trace read is not a public read", func(e *workerEcho) { *e = unpricedWorkerEcho("not_in_offer"); e.Read = echoPtr(true) }, false},
		{"wrong worker", func(e *workerEcho) { e.Worker = "cloud" }, false},
		{"implicit read", func(e *workerEcho) { e.Read = nil }, false},
		{"missing pricing", func(e *workerEcho) { e.Pricing = "" }, false},
		{"missing held", func(e *workerEcho) { e.Held = nil }, false},
		{"wrong held default", func(e *workerEcho) { e.Held = echoPtr(true) }, false},
		{"wrong rival default", func(e *workerEcho) { e.Pricing = "default_held_warm" }, false},
		{"read empty reason is not null", func(e *workerEcho) { e.Reason = echoPtr("") }, false},
		{"read with reason", func(e *workerEcho) { e.Reason = echoPtr("package_mode") }, false},
		{"silent unread", func(e *workerEcho) { e.Read = echoPtr(false) }, false},
		{"old reason spelling", func(e *workerEcho) { *e = unpricedWorkerEcho("not_offered") }, false},
		{"observed without observed basis", func(e *workerEcho) { e.Pricing = "observed" }, false},
		{"observed without read", func(e *workerEcho) {
			e.Pricing = "observed"
			e.Basis = echoPtr("observed_ttl")
			e.Read = echoPtr(false)
			e.Reason = echoPtr("superseded")
		}, false},
		{"unknown warmth", func(e *workerEcho) { e.PWarm = nil }, false},
		{"invalid warmth", func(e *workerEcho) { e.PWarm = echoPtr(math.NaN()) }, false},
		{"out of range warmth", func(e *workerEcho) { e.PWarm = echoPtr(1.1) }, false},
		{"missing prefix", func(e *workerEcho) { e.PrefixTokens = nil }, false},
		{"negative prefix", func(e *workerEcho) { e.PrefixTokens = echoPtr(-1) }, false},
		{"missing prefix basis", func(e *workerEcho) { e.PrefixBasis = echoPtr("") }, false},
		{"missing admissions", func(e *workerEcho) { e.NotAdmitted = nil }, false},
		{"unknown admissions", func(e *workerEcho) { e.NotAdmitted = []notAdmittedEcho{{Field: "current_prefix", Reason: "unknown"}} }, false},
		{"unpriced cannot invent zero warmth", func(e *workerEcho) { *e = unpricedWorkerEcho("not_in_offer"); e.PWarm = echoPtr(0.0) }, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			e := defaultWorkerEcho()
			test.change(&e)
			err := validateWorkerEcho(workers, []workerEcho{e}, "user_message", "repartition_multiwarm")
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v, error=%v", test.valid, err)
			}
		})
	}
}

func TestWorkerEchoRequiresEveryRowInRequestOrder(t *testing.T) {
	workers := []externalWorker{{ActionID: "a", Worker: "local"}, {ActionID: "b", Worker: "cloud"}}
	a, b := defaultWorkerEcho(), defaultWorkerEcho()
	b.ActionID, b.Worker = "b", "cloud"
	for _, rows := range [][]workerEcho{nil, {a}, {a, a}, {b, a}} {
		if err := validateWorkerEcho(workers, rows, "user_message", "repartition_multiwarm"); err == nil {
			t.Fatal("accepted missing, duplicate or reordered evidence")
		}
	}
	if err := validateWorkerEcho(workers, []workerEcho{a, b}, "user_message", "repartition_multiwarm"); err != nil {
		t.Fatal(err)
	}
}

func TestAdmittedEvidenceRequiresMultiwarmOnlyAtUserMessage(t *testing.T) {
	for _, test := range []struct {
		name           string
		worker         externalWorker
		echo           workerEcho
		boundary, mode string
		valid          bool
	}{
		{"usage-only package default", externalWorker{Usage: &externalUsage{Units: "provider:local", Provenance: "provider_reported"}}, defaultWorkerEcho(), "user_message", "frozen", true},
		{"TTL-only package default", externalWorker{TTL: &externalTTL{Seconds: 300, Basis: "wire_declared"}}, defaultWorkerEcho(), "user_message", "frozen", true},
		{"qualified prefix cannot keep package default", prefixWorker(), countedWorkerEcho(), "user_message", "frozen", false},
		{"qualified prefix in worker units", prefixWorker(), countedWorkerEcho(), "user_message", "repartition_multiwarm", true},
		{"qualified prefix cannot become encoder count", prefixWorker(), defaultWorkerEcho(), "user_message", "repartition_multiwarm", false},
		{"qualified prefix at held boundary", prefixWorker(), unpricedWorkerEcho("boundary"), "tool_result", "frozen", true},
		{"user message cannot be unpriced wholesale", prefixWorker(), unpricedWorkerEcho("boundary"), "user_message", "repartition_multiwarm", false},
		{"tool boundary cannot price workers", prefixWorker(), countedWorkerEcho(), "tool_result", "repartition_multiwarm", false},
		{"measured local residency requires multiwarm", externalWorker{Placement: "local", Residency: &Residency{State: "resident", Provenance: "local_measured"}}, defaultWorkerEcho(), "user_message", "frozen", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			test.worker.ActionID, test.worker.Worker = "a", "local"
			err := validateWorkerEcho([]externalWorker{test.worker}, []workerEcho{test.echo}, test.boundary, test.mode)
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v, error=%v", test.valid, err)
			}
		})
	}
	if err := validateWorkerEcho(nil, []workerEcho{}, "task_start", "frozen"); err != nil {
		t.Fatal(err)
	}
}

func TestEvidenceAdmissionIsExplicitAndDoesNotRelabelCounts(t *testing.T) {
	worker := prefixWorker()
	worker.CurrentPrefix.Provenance = "caller_estimated"
	echo := defaultWorkerEcho()
	if err := validateWorkerEcho([]externalWorker{worker}, []workerEcho{echo}, "user_message", "frozen"); err == nil {
		t.Fatal("estimated prefix silently disappeared")
	}
	echo.NotAdmitted = []notAdmittedEcho{{Field: "current_prefix", Reason: "estimated_not_admitted"}}
	if err := validateWorkerEcho([]externalWorker{worker}, []workerEcho{echo}, "user_message", "frozen"); err != nil {
		t.Fatal(err)
	}
	worker.CurrentPrefix.Provenance = "local_measured"
	if err := validateWorkerEcho([]externalWorker{worker}, []workerEcho{echo}, "user_message", "repartition_multiwarm"); err == nil {
		t.Fatal("qualified count rejected as estimated")
	}
	worker.Usage = &externalUsage{Units: "other-tokenizer", Provenance: "provider_reported"}
	echo = countedWorkerEcho()
	echo.NotAdmitted = []notAdmittedEcho{{Field: "usage", Reason: "units_mismatch"}}
	if err := validateWorkerEcho([]externalWorker{worker}, []workerEcho{echo}, "user_message", "repartition_multiwarm"); err != nil {
		t.Fatal(err)
	}
	echo.PrefixTokens = echoPtr(99)
	if err := validateWorkerEcho([]externalWorker{worker}, []workerEcho{echo}, "user_message", "repartition_multiwarm"); err == nil {
		t.Fatal("qualified count changed silently")
	}
}

func TestCachePolicyEchoStatesItsEffectiveModeAndSource(t *testing.T) {
	for _, source := range []string{"package", "request", "service_config"} {
		if err := validateCachePolicyEcho(source, "repartition_multiwarm", "account_scoped", echoPtr(true)); err != nil {
			t.Fatal(err)
		}
		if err := validateCachePolicyEcho(source, "frozen", "per_lookup_random", echoPtr(false)); err != nil {
			t.Fatal(err)
		}
	}
	for _, test := range []struct {
		source, mode string
		multiwarm    *bool
	}{
		{"", "repartition_multiwarm", echoPtr(true)}, {"guessed", "frozen", echoPtr(false)},
		{"package", "frozen", nil}, {"package", "frozen", echoPtr(true)}, {"service_config", "repartition_multiwarm", echoPtr(false)},
	} {
		if err := validateCachePolicyEcho(test.source, test.mode, "per_lookup_random", test.multiwarm); err == nil {
			t.Fatal("accepted missing or inconsistent policy acknowledgement")
		}
	}
}

func TestEpisodeIdentityCannotBeOmittedOrReplacedByLocalSessionIdentity(t *testing.T) {
	for _, identity := range []string{"", "session_only", "local_hash", "unsupported"} {
		if err := validateCachePolicyEcho("service_config", "repartition_multiwarm", identity, echoPtr(true)); err == nil {
			t.Fatalf("accepted invalid applied episode identity %q", identity)
		}
	}
}
