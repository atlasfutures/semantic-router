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
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"
)

const (
	controllerConfigEnv   = "RAYLINE_ARC_CONTROLLER_CONFIG"
	controllerDecisionEnv = "RAYLINE_ARC_CONTROLLER_DECISION"
	// #nosec G101 -- this is an environment variable name, never a credential value.
	controllerCredentialNameEnv = "RAYLINE_ARC_CONTROLLER_PASSWORD_ENV"
	controllerRedisUsernameEnv  = "RAYLINE_ARC_CONTROLLER_REDIS_USERNAME"
	defaultConfigPath           = "/app/config.yaml"
)

type environmentLookup func(string) (string, bool)

type controllerCommandOptions struct {
	command             string
	configPath          string
	decisionName        string
	passwordEnvironment string
	redisUsername       string
	replicaID           string
	replicaBaseURL      string
	operationTimeout    time.Duration
	reconcileInterval   time.Duration
}

func parseControllerCommandOptions(
	args []string,
	lookup environmentLookup,
) (controllerCommandOptions, error) {
	if len(args) == 0 {
		return controllerCommandOptions{}, errors.New("a command is required")
	}
	command, err := parseControllerCommand(args[0])
	if err != nil {
		return controllerCommandOptions{}, err
	}

	options := controllerCommandOptions{
		command:      command,
		configPath:   environmentDefault(lookup, controllerConfigEnv, defaultConfigPath),
		decisionName: environmentDefault(lookup, controllerDecisionEnv, ""),
		passwordEnvironment: environmentDefault(
			lookup,
			controllerCredentialNameEnv,
			"",
		),
		redisUsername: environmentDefault(
			lookup,
			controllerRedisUsernameEnv,
			"",
		),
		operationTimeout:  5 * time.Second,
		reconcileInterval: 5 * time.Second,
	}
	set := flag.NewFlagSet(command, flag.ContinueOnError)
	set.SetOutput(io.Discard)
	set.StringVar(&options.configPath, "config", options.configPath, "canonical router config path")
	set.StringVar(&options.decisionName, "decision", options.decisionName, "dynamic Rayline ARC decision name")
	set.StringVar(
		&options.passwordEnvironment,
		"password-env",
		options.passwordEnvironment,
		"controller-only Redis password environment override",
	)
	set.StringVar(
		&options.redisUsername,
		"redis-username",
		options.redisUsername,
		"controller-only Redis ACL username",
	)
	set.DurationVar(
		&options.operationTimeout,
		"timeout",
		options.operationTimeout,
		"timeout for each Redis operation",
	)
	addControllerCommandFlags(set, &options)
	if err := set.Parse(args[1:]); err != nil {
		return controllerCommandOptions{}, err
	}
	return validateControllerCommandOptions(set, options)
}

func parseControllerCommand(value string) (string, error) {
	command := strings.TrimSpace(value)
	switch command {
	case "status", "register", "drain", "reconcile", "run":
		return command, nil
	case "help", "-h", "--help":
		return "", flag.ErrHelp
	default:
		return "", fmt.Errorf("unknown command %q", command)
	}
}

func addControllerCommandFlags(
	set *flag.FlagSet,
	options *controllerCommandOptions,
) {
	if set == nil || options == nil {
		return
	}
	if options.command == "register" || options.command == "drain" {
		set.StringVar(&options.replicaID, "replica-id", "", "stable encoder replica ID")
	}
	if options.command == "register" {
		set.StringVar(&options.replicaBaseURL, "base-url", "", "reviewed encoder base URL")
	}
	if options.command == "run" {
		set.DurationVar(
			&options.reconcileInterval,
			"interval",
			options.reconcileInterval,
			"continuous reconciliation interval",
		)
	}
}

func validateControllerCommandOptions(
	set *flag.FlagSet,
	options controllerCommandOptions,
) (controllerCommandOptions, error) {
	if set == nil {
		return controllerCommandOptions{}, errors.New("controller flags are unavailable")
	}
	if set.NArg() != 0 {
		return controllerCommandOptions{}, errors.New("unexpected positional arguments")
	}
	if strings.TrimSpace(options.configPath) == "" || options.operationTimeout <= 0 {
		return controllerCommandOptions{}, errors.New("config and positive timeout are required")
	}
	if (options.command == "register" || options.command == "drain") &&
		strings.TrimSpace(options.replicaID) == "" {
		return controllerCommandOptions{}, fmt.Errorf("%s requires --replica-id", options.command)
	}
	if options.command == "register" && strings.TrimSpace(options.replicaBaseURL) == "" {
		return controllerCommandOptions{}, errors.New("register requires --base-url")
	}
	if options.command == "run" && options.reconcileInterval <= 0 {
		return controllerCommandOptions{}, errors.New("run requires a positive --interval")
	}
	return options, nil
}

func environmentDefault(
	lookup environmentLookup,
	name string,
	fallback string,
) string {
	if lookup == nil {
		return fallback
	}
	if value, exists := lookup(name); exists && strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}

func controllerUsage() string {
	return "usage: rayline-arc-controller <command> [options]\n\n" +
		"commands:\n" +
		"  status      print privacy-safe membership state\n" +
		"  register    add one reviewed active replica with a revisioned CAS\n" +
		"  drain       mark one replica draining with a revisioned CAS\n" +
		"  reconcile   remove eligible zero-reference draining replicas once\n" +
		"  run         continuously reconcile until SIGINT or SIGTERM\n\n" +
		"common options:\n" +
		"  --config PATH         canonical router config (default /app/config.yaml)\n" +
		"  --decision NAME       required only when multiple dynamic ARC decisions exist\n" +
		"  --password-env NAME   override the router Redis password environment name\n" +
		"  --redis-username NAME controller-only Redis ACL username\n" +
		"  --timeout DURATION    per-operation timeout (default 5s)\n\n" +
		"register options:\n" +
		"  --replica-id ID\n" +
		"  --base-url URL\n\n" +
		"drain options:\n" +
		"  --replica-id ID\n\n" +
		"run options:\n" +
		"  --interval DURATION   reconciliation interval (default 5s)\n"
}
