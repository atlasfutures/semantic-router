/*
Copyright 2026 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type membershipMemberStatus struct {
	ID             string                         `json:"id"`
	State          raylinearc.EncoderReplicaState `json:"state"`
	DrainStartedAt *time.Time                     `json:"drain_started_at,omitempty"`
}

type membershipStatus struct {
	Command  string                   `json:"command"`
	Revision uint64                   `json:"revision"`
	Active   int                      `json:"active"`
	Draining int                      `json:"draining"`
	Members  []membershipMemberStatus `json:"members"`
}

type membershipDrainResult struct {
	Command   string `json:"command"`
	ReplicaID string `json:"replica_id"`
	Revision  uint64 `json:"revision"`
	Active    int    `json:"active"`
	Draining  int    `json:"draining"`
}

type membershipRegisterResult struct {
	Command   string `json:"command"`
	ReplicaID string `json:"replica_id"`
	Revision  uint64 `json:"revision"`
	Active    int    `json:"active"`
	Draining  int    `json:"draining"`
}

type membershipReconcileResult struct {
	Command         string `json:"command"`
	Outcome         string `json:"outcome"`
	Revision        uint64 `json:"revision,omitempty"`
	Draining        int    `json:"draining,omitempty"`
	Eligible        int    `json:"eligible,omitempty"`
	Referenced      int    `json:"referenced,omitempty"`
	Removed         int    `json:"removed,omitempty"`
	MinimumRetained int    `json:"minimum_retained,omitempty"`
	Error           string `json:"error,omitempty"`
}

func runControllerCommand(
	ctx context.Context,
	args []string,
	stdout io.Writer,
	stderr io.Writer,
	lookup environmentLookup,
) int {
	options, err := parseControllerCommandOptions(args, lookup)
	if errors.Is(err, flag.ErrHelp) {
		_, _ = io.WriteString(stdout, controllerUsage())
		return 0
	}
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "error: %s\n\n%s", err, controllerUsage())
		return 2
	}
	runtime, err := openControllerRuntime(options, lookup)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "error: %s\n", err)
		return 1
	}
	defer func() { _ = runtime.close() }()

	if err := executeControllerCommand(ctx, runtime, options, stdout); err != nil {
		_, _ = fmt.Fprintf(stderr, "error: %s\n", err)
		return 1
	}
	return 0
}

func executeControllerCommand(
	ctx context.Context,
	runtime *controllerRuntime,
	options controllerCommandOptions,
	stdout io.Writer,
) error {
	switch options.command {
	case "status":
		return executeStatus(ctx, runtime, options.operationTimeout, stdout)
	case "register":
		return executeRegister(ctx, runtime, options, stdout)
	case "drain":
		return executeDrain(ctx, runtime, options, stdout)
	case "reconcile":
		return executeReconcile(ctx, runtime, options.operationTimeout, stdout)
	case "run":
		return executeRun(ctx, runtime, options, stdout)
	default:
		return errors.New("controller command is unsupported")
	}
}

func executeRegister(
	ctx context.Context,
	runtime *controllerRuntime,
	options controllerCommandOptions,
	stdout io.Writer,
) error {
	operationContext, cancel := context.WithTimeout(ctx, options.operationTimeout)
	defer cancel()
	snapshot, err := runtime.controller.RegisterActive(
		operationContext,
		options.replicaID,
		options.replicaBaseURL,
	)
	if err != nil {
		return err
	}
	status := newMembershipStatus("register", snapshot)
	return writeJSON(stdout, membershipRegisterResult{
		Command:   "register",
		ReplicaID: options.replicaID,
		Revision:  status.Revision,
		Active:    status.Active,
		Draining:  status.Draining,
	})
}

func executeStatus(
	ctx context.Context,
	runtime *controllerRuntime,
	timeout time.Duration,
	stdout io.Writer,
) error {
	operationContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	snapshot, err := runtime.source.Load(operationContext)
	if err != nil {
		return err
	}
	return writeJSON(stdout, newMembershipStatus("status", snapshot))
}

func executeDrain(
	ctx context.Context,
	runtime *controllerRuntime,
	options controllerCommandOptions,
	stdout io.Writer,
) error {
	operationContext, cancel := context.WithTimeout(ctx, options.operationTimeout)
	defer cancel()
	snapshot, err := runtime.controller.BeginDrain(
		operationContext,
		options.replicaID,
	)
	if err != nil {
		return err
	}
	status := newMembershipStatus("drain", snapshot)
	return writeJSON(stdout, membershipDrainResult{
		Command:   "drain",
		ReplicaID: options.replicaID,
		Revision:  status.Revision,
		Active:    status.Active,
		Draining:  status.Draining,
	})
}

func executeReconcile(
	ctx context.Context,
	runtime *controllerRuntime,
	timeout time.Duration,
	stdout io.Writer,
) error {
	report, err := reconcileOnce(ctx, runtime, timeout)
	if err != nil {
		return err
	}
	return writeJSON(stdout, newReconcileResult("reconcile", report, nil))
}

func executeRun(
	ctx context.Context,
	runtime *controllerRuntime,
	options controllerCommandOptions,
	stdout io.Writer,
) error {
	ticker := time.NewTicker(options.reconcileInterval)
	defer ticker.Stop()
	lastEvent := []byte(nil)
	for {
		report, err := reconcileOnce(ctx, runtime, options.operationTimeout)
		event, marshalErr := json.Marshal(newReconcileResult("run", report, err))
		if marshalErr != nil {
			return errors.New("controller event cannot be encoded")
		}
		if !bytes.Equal(event, lastEvent) {
			if _, writeErr := stdout.Write(append(event, '\n')); writeErr != nil {
				return writeErr
			}
			lastEvent = append(lastEvent[:0], event...)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
		}
	}
}

func reconcileOnce(
	ctx context.Context,
	runtime *controllerRuntime,
	timeout time.Duration,
) (raylinearc.EncoderMembershipDrainReport, error) {
	operationContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return runtime.controller.Reconcile(operationContext)
}

func newMembershipStatus(
	command string,
	snapshot raylinearc.EncoderMembershipSnapshot,
) membershipStatus {
	status := membershipStatus{
		Command:  command,
		Revision: snapshot.Revision,
		Members:  make([]membershipMemberStatus, 0, len(snapshot.Replicas)),
	}
	for _, replica := range snapshot.Replicas {
		status.Members = append(status.Members, membershipMemberStatus{
			ID:             replica.ID,
			State:          replica.State,
			DrainStartedAt: replica.DrainStartedAt,
		})
		switch replica.State {
		case raylinearc.EncoderReplicaActive:
			status.Active++
		case raylinearc.EncoderReplicaDraining:
			status.Draining++
		}
	}
	return status
}

func newReconcileResult(
	command string,
	report raylinearc.EncoderMembershipDrainReport,
	err error,
) membershipReconcileResult {
	result := membershipReconcileResult{
		Command:         command,
		Outcome:         "success",
		Revision:        report.Revision,
		Draining:        report.Draining,
		Eligible:        report.Eligible,
		Referenced:      report.Referenced,
		Removed:         report.Removed,
		MinimumRetained: report.MinimumRetained,
	}
	if err != nil {
		result.Outcome = "error"
		result.Error = boundedControllerError(err)
	}
	return result
}

func boundedControllerError(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, raylinearc.ErrEncoderMembershipConflict):
		return "revision_conflict"
	case errors.Is(err, raylinearc.ErrEncoderMembershipNotFound):
		return "membership_missing"
	default:
		return "control_plane"
	}
}

func writeJSON(writer io.Writer, value interface{}) error {
	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return errors.New("controller output cannot be encoded")
	}
	return nil
}
