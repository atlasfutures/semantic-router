// rayline-core is the rld-supervised transport for VSR's shared execution core.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/execution"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, nil)))
	if err := run(); err != nil {
		slog.Error("execution core stopped", "error", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", "", "execution config path")
	listen := flag.String("listen", "127.0.0.1:0", "loopback listening address")
	version := flag.Bool("version", false, "print process contract version")
	flag.Parse()
	if *version {
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"contract_version": execution.ContractVersion})
	}
	if *configPath == "" {
		return fmt.Errorf("--config is required")
	}
	host, _, err := net.SplitHostPort(*listen)
	if err != nil || net.ParseIP(host) == nil || !net.ParseIP(host).IsLoopback() {
		return fmt.Errorf("--listen must name an explicit loopback IP and port")
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if os.Getenv("RAYLINE_EXEC_PARENT_PIPE") == "1" {
		go func() { _, _ = io.Copy(io.Discard, os.Stdin); cancel() }()
	}
	config, registry, err := execution.LoadConfig(*configPath)
	if err != nil {
		return err
	}
	decisions, err := execution.NewHTTPDecisions(config)
	if err != nil {
		return err
	}
	handler, err := execution.NewServer(config, registry, decisions, os.Getenv("RAYLINE_EXEC_AUTH_TOKEN"))
	if err != nil {
		return err
	}
	if err := handler.RegisterConfiguredPromptCounters(ctx); err != nil {
		return err
	}
	listener, err := net.Listen("tcp", *listen)
	if err != nil {
		return err
	}
	defer listener.Close()
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 90 * time.Second, MaxHeaderBytes: 64 << 10, BaseContext: func(net.Listener) context.Context { return ctx }}
	errorsCh := make(chan error, 1)
	go func() { errorsCh <- server.Serve(listener) }()
	// Exactly one startup line is the parent handshake; all diagnostics use
	// stderr. No credentials, provider bodies or prompts belong on either.
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{"contract_version": execution.ContractVersion, "listen_url": "http://" + listener.Addr().String(), "pid": os.Getpid(), "instance_id": handler.InstanceID()}); err != nil {
		_ = server.Close()
		return err
	}
	select {
	case err := <-errorsCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdown, stop := context.WithTimeout(context.Background(), 5*time.Second)
		defer stop()
		if err := server.Shutdown(shutdown); err != nil {
			_ = server.Close()
			return err
		}
		return nil
	}
}
