// Command conformance-fixture runs the protocol-conformance provider fixture as a
// standalone process.
//
// A case driver that runs in the same process should call the fixture package
// directly. This command exists for the deployed case: the fixture runs beside the
// router, the driver loads a script with POST /reset, and reads what the provider
// saw with GET /observed.
//
//	conformance-fixture -addr 127.0.0.1:8199
//	curl --data-binary @replay.yaml "http://127.0.0.1:8199/reset?dir=$PWD/seed-chat-identity"
//	curl http://127.0.0.1:8199/observed
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance/fixture"
)

// shutdownGrace bounds how long a scripted delay or stream may still be running
// when the process is asked to stop.
const shutdownGrace = 5 * time.Second

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	addr := flag.String("addr", "127.0.0.1:8199", "Address to listen on; use :0 to take any free port")
	flag.Parse()

	server, err := fixture.Start(*addr)
	if err != nil {
		return err
	}
	fmt.Println(server.URL())

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}
