# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import openrouter_modal_native_benchmark as benchmark
import yaml
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_modal_native_fixture import (
    CHECKPOINT_REMOTE_PATH,
    DECISION_LOG_REMOTE_PATH,
    router_config_text,
)
from openrouter_modal_native_fixture import WORKERS as NATIVE_WORKERS

EXPECTED_COMPLETION_LIMIT = 96
EXPECTED_NATIVE_REQUESTS = 63
EXPECTED_DIRECT_REQUESTS = 13
EXPECTED_TOTAL_REQUESTS = 76
EXPECTED_COVERAGE_REQUESTS = 24
EXPECTED_CONCURRENCY = 4
EXPECTED_REPETITIONS = 2
FIXTURE_TOTAL_REQUESTS = 6
FIXTURE_THROUGHPUT_RATIO = 0.5
FIXTURE_EMBEDDING_SECONDS = 1.5


def test_native_config_is_the_agt013_pool_with_local_kv_encoding() -> None:
    config = yaml.safe_load(router_config_text())
    router = config["router"]
    assert router["checkpoint_path"] == f"/artifacts/{CHECKPOINT_REMOTE_PATH}"
    assert router["log_path"] == f"/artifacts/{DECISION_LOG_REMOTE_PATH}"
    assert router["mtrouter_device"] == "cuda"
    assert router["mtrouter_incremental_encode"] is True
    assert [worker["id"] for worker in config["workers"]] == list(WORKERS)
    assert {worker["id"]: worker["model"] for worker in config["workers"]} == WORKERS
    assert all(
        worker["openrouter_allow_fallbacks"] is False for worker in config["workers"]
    )
    assert all(
        worker["max_completion_tokens"] == EXPECTED_COMPLETION_LIMIT
        for worker in config["workers"]
    )


def test_native_config_temperature_mirrors_the_artifact_per_worker() -> None:
    """The native config must not invent a temperature the artifact omits.

    This generator hardcoded `temperature: 0` for every worker, so a worker
    whose model does not advertise the parameter still had it sent, and
    `require_parameters` turned that into a 404 "No endpoints found" on the
    measured path. The provider preflight cannot catch it — that path builds
    its payload from the benchmark client rather than from this config — so
    the mismatch is only observable by a paid arm aborting mid-run.
    """

    config = yaml.safe_load(router_config_text())
    configured = {worker["id"]: worker for worker in config["workers"]}

    for worker in NATIVE_WORKERS:
        declared = worker["temperature"]
        emitted = configured[worker["id"]]
        if declared is None:
            assert "temperature" not in emitted, (
                f"{worker['id']} declares no temperature but the native config "
                "sends one"
            )
        else:
            assert emitted["temperature"] == declared


def test_native_benchmark_has_a_fixed_small_request_envelope() -> None:
    assert benchmark.EXPECTED_NATIVE_REQUESTS == EXPECTED_NATIVE_REQUESTS
    assert benchmark.EXPECTED_DIRECT_REQUESTS == EXPECTED_DIRECT_REQUESTS
    assert benchmark.MAX_PROVIDER_REQUESTS == EXPECTED_TOTAL_REQUESTS
    assert benchmark.MAX_COVERAGE_REQUESTS == EXPECTED_COVERAGE_REQUESTS
    assert benchmark.CONCURRENCY == EXPECTED_CONCURRENCY
    assert benchmark.REPETITIONS == EXPECTED_REPETITIONS


def test_decision_reader_excludes_same_sink_budget_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"schema_version":"rayline-router.budget.v1"}',
                '{"schema_version":"rayline-router.decision.v3","request_id":"r1"}',
            ]
        )
        + "\n"
    )
    assert benchmark.read_decisions(path) == [
        {"schema_version": "rayline-router.decision.v3", "request_id": "r1"}
    ]


def _result(request_id: str, path: str, total: float) -> dict[str, object]:
    return {
        "path": path,
        "case_id": "agentic-00",
        "scenario": "code_patch",
        "request_id": request_id,
        "selected_worker": "worker-a",
        "response_model": WORKERS["worker-a"],
        "time_to_first_event_seconds": total - 0.01,
        "time_to_first_token_seconds": total - 0.005,
        "total_seconds": total,
        "data_events": 3,
    }


def _decision(request_id: str, *, embedding_ms: float) -> dict[str, object]:
    return {
        "request_id": request_id,
        "error": "",
        "selected_worker": "worker-a",
        "worker_model": WORKERS["worker-a"],
        "served_model": WORKERS["worker-a"],
        "served_provider": PROVIDER_NAMES["worker-a"][0],
        "input_tokens": 100,
        "output_tokens": 10,
        "decision_latency_ms": embedding_ms + 1,
        "features": {
            "embedding_latency_ms": embedding_ms,
            "q_latency_ms": 1,
            "encode_mode": "full" if embedding_ms else "not_applicable",
            "serialized_tokens": 100,
        },
        "transport": {"attempts": [{"status": "success"}]},
        "settlement_cost_basis": {"provider_charged_usd": 0.001},
    }


def test_final_report_joins_client_and_decision_planes(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "EXPECTED_NATIVE_REQUESTS", 4)
    monkeypatch.setattr(benchmark, "EXPECTED_DIRECT_REQUESTS", 2)
    monkeypatch.setattr(benchmark, "MAX_PROVIDER_REQUESTS", 6)
    static = _result("static", "modal_static", 2.0)
    arc = _result("arc", "modal_arc", 4.0)
    stratified = _result("stratified", "modal_static", 1.0)
    endpoint = _result("endpoint", "modal_static", 1.0)
    direct = {
        "path": "direct",
        "case_id": "agentic-00",
        "scenario": "code_patch",
        "selected_worker": "worker-a",
        "response_model": WORKERS["worker-a"],
        "provider": PROVIDER_NAMES["worker-a"][0],
        "external_attempts": 1,
        "time_to_first_event_seconds": 0.5,
        "time_to_first_token_seconds": 0.5,
        "total_seconds": 1.0,
        "envoy_upstream_service_seconds": None,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cost_usd": 0.001,
        "data_events": 3,
    }
    client = {
        "run_id": "test",
        "key_readiness": {"external_attempts": 1, "cost_usd": 0.001},
        "endpoint_probes": [endpoint],
        "coverage": [],
        "selected_cases": [
            {
                "case_id": "agentic-00",
                "scenario": "code_patch",
                "expected_worker": "worker-a",
            }
        ],
        "natural_phases": [
            {"path": "modal_static", "wall_seconds": 2.0, "results": [static]},
            {"path": "modal_arc", "wall_seconds": 4.0, "results": [arc]},
        ],
        "stratified_phases": [
            {"path": "direct", "wall_seconds": 1.0, "results": [direct]},
            {
                "path": "modal_static",
                "wall_seconds": 1.0,
                "results": [stratified],
            },
        ],
    }
    report = benchmark.finalize_report(
        client_report=client,
        decisions=[
            _decision("endpoint", embedding_ms=0),
            _decision("static", embedding_ms=0),
            _decision("arc", embedding_ms=1500),
            _decision("stratified", embedding_ms=0),
        ],
        actual_openrouter_cost_usd=0.006,
        deployment={"gpu": "L40S"},
        checkpoint={"checkpoint_sha256": "a" * 64},
    )
    assert report["status"] == "passed"
    assert report["actual_provider_requests"] == FIXTURE_TOTAL_REQUESTS
    assert (
        report["natural_comparison"]["arc_to_static_throughput_ratio"]
        == FIXTURE_THROUGHPUT_RATIO
    )
    assert (
        report["natural_paths"]["modal_arc"]["embedding_latency"]["p50_seconds"]
        == FIXTURE_EMBEDDING_SECONDS
    )
    assert (
        report["natural_paths"]["modal_arc"]["embedding_latency"]["mean_seconds"]
        == FIXTURE_EMBEDDING_SECONDS
    )
    assert report["natural_paths"]["modal_arc"]["providers"] == ["Baidu"]
