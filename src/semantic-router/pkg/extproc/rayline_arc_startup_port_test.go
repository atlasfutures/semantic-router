//go:build !windows && cgo

package extproc

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A late encoder must not hold the router's port shut.
//
// Cloud Run kills an instance that does not accept on its port inside the
// startup window, so an encoder that accepts TCP and then never answers used
// to take the whole router down with it rather than degrade one decision. The
// contract this pins is: the port opens on time, the ARC gauge reads 0, ARC
// selection fails closed rather than guessing an arm, and the router arms
// itself once the encoder answers -- with no restart in between.
//
// This drives the real construction path, extproc.NewServer plus Start, from a
// config file, because that is the path that used to block.
func TestRouterOpensItsPortWhileTheARCEncoderHangs(t *testing.T) {
	// The dispatch contract checks that each arm's credential is resolvable, so
	// the artifact's api_key_env has to exist before the router is built.
	t.Setenv(fixtureAPIKeyEnv, "arc-fixture-key")
	encoder := newHangingARCEncoder(t)
	provider := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			http.Error(writer, "no upstream call is expected", http.StatusNotImplemented)
		},
	))
	t.Cleanup(provider.Close)

	// The config watcher watches the config file's whole directory, so the
	// artifact lives in a sibling directory: writing it must not trigger a
	// reload and a second router generation.
	artifactDir := writeARCArtifact(t, provider.URL)
	// A budget far longer than the port is allowed. The dev overlay runs 600.
	configPath := writeARCRouterConfig(t, artifactDir, encoder.URL(), provider.URL,
		encoderBudget{totalTimeoutSeconds: 90, retryInitialSeconds: 1, retryMaxSeconds: 2})

	// Construction happens on the goroutine too, because construction is what
	// used to block: a router that probes the encoder before it listens never
	// reaches Start at all.
	port := freeTCPPort(t)
	var built atomic.Pointer[Server]
	serveErr := make(chan error, 1)
	go func() {
		server, err := NewServer(configPath, port, false, "", nil)
		if err != nil {
			serveErr <- fmt.Errorf("NewServer: %w", err)
			return
		}
		built.Store(server)
		serveErr <- server.Start()
	}()
	t.Cleanup(func() {
		if server := built.Load(); server != nil {
			server.Stop()
		}
	})

	// The deployment gate allows 45 s. This asks for far less on purpose: the
	// contract is that construction does not wait on the encoder at all, not
	// that it waits a tolerable amount. A router that bounded its synchronous
	// probe at 30 s passes a 45 s assertion on that default alone, and the
	// same code with a longer bound does not.
	awaitTCPAccept(t, port, raylineARCPortListenBudget, serveErr)
	server := built.Load()

	if ready := arcComponentReadyGauge(); ready != 0 {
		t.Fatalf("ARC readiness gauge = %v while the encoder hangs, want 0", ready)
	}

	assertARCSelectionNotReady(t, server)

	// The first probe is still in flight, so answering it arms the router
	// without waiting out the encoder budget.
	encoder.answer()

	awaitARCArmed(t, 30*time.Second)

	select {
	case err := <-serveErr:
		t.Fatalf("Start returned early: %v", err)
	default:
	}
}

// The router listens in well under a second when it does not wait on the
// encoder, so this leaves ample headroom on a loaded machine while staying
// far below any wait the encoder could impose.
const raylineARCPortListenBudget = 10 * time.Second

func assertARCSelectionNotReady(t *testing.T, server *Server) {
	t.Helper()
	selector, ok := server.GetRouter().ModelSelector.Get(selection.MethodRaylineARC)
	if !ok {
		t.Fatal("the ARC selector is not registered")
	}
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	_, err = selector.Select(t.Context(), validARCSelectionContext(state))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "not_ready" {
		t.Fatalf("selection error while the encoder hangs = %v, want class not_ready", err)
	}
}

// A probe that fails must not leave ARC dead until someone restarts the
// instance. This drives the recovery path the deployment gate asks for: the
// first attempt times out, the backoff runs, and a later attempt arms.
//
// The encoder budget is small here on purpose. Each attempt is bounded only by
// total_timeout_seconds and the next cannot start until this one ends, so a
// long budget would hide the re-probe behind a single very slow attempt.
func TestRouterArmsARCOnAReprobeAfterTheFirstProbeTimesOut(t *testing.T) {
	t.Setenv(fixtureAPIKeyEnv, "arc-fixture-key")
	encoder := newHangingARCEncoder(t)
	provider := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			http.Error(writer, "no upstream call is expected", http.StatusNotImplemented)
		},
	))
	t.Cleanup(provider.Close)

	artifactDir := writeARCArtifact(t, provider.URL)
	configPath := writeARCRouterConfig(t, artifactDir, encoder.URL(), provider.URL,
		encoderBudget{totalTimeoutSeconds: 2, retryInitialSeconds: 1, retryMaxSeconds: 2})

	port := freeTCPPort(t)
	var built atomic.Pointer[Server]
	serveErr := make(chan error, 1)
	go func() {
		server, err := NewServer(configPath, port, false, "", nil)
		if err != nil {
			serveErr <- fmt.Errorf("NewServer: %w", err)
			return
		}
		built.Store(server)
		serveErr <- server.Start()
	}()
	t.Cleanup(func() {
		if server := built.Load(); server != nil {
			server.Stop()
		}
	})

	awaitTCPAccept(t, port, raylineARCPortListenBudget, serveErr)

	// Pending: registered, unarmed, probing.
	if ready := arcComponentReadyGauge(); ready != 0 {
		t.Fatalf("ARC readiness gauge = %v before any probe answered, want 0", ready)
	}

	// A second attempt can only follow the first one timing out, so this is
	// the proof that the backoff ran rather than that one probe was slow.
	encoder.awaitAttempts(t, 2, 30*time.Second)
	if ready := arcComponentReadyGauge(); ready != 0 {
		t.Fatalf("ARC readiness gauge = %v after a failed probe, want 0", ready)
	}

	encoder.answer()

	awaitARCArmed(t, 30*time.Second)
}
