# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

benchmark = importlib.import_module("modal_fullstack_benchmark")

EXPECTED_WAVES_PER_LEVEL = 2
EXPECTED_SOAK_CONCURRENCY = 4
EXPECTED_SOAK_WAVES = 5
EXPECTED_LADDER_REQUESTS_PER_PATH = 30
EXPECTED_SOAK_REQUESTS = 20
EXPECTED_BENCHMARK_REQUESTS = 80
EXPECTED_MAX_GENERATION_REQUESTS = 112
EXPECTED_REQUESTS_PER_WORKER = 40
TEST_CONCURRENCY = 4
EXPECTED_TEST_COMPLETION_TOKENS = 8
EXPECTED_THROUGHPUT_RATIO = 0.5
EXPECTED_P95_OVERHEAD_SECONDS = 0.75
EXPECTED_P95_RATIO = 2.5


def test_workload_is_fixed_small_and_balanced() -> None:
    assert benchmark.CONCURRENCY_LEVELS == (1, 2, 4, 8)
    assert benchmark.WAVES_PER_LEVEL == EXPECTED_WAVES_PER_LEVEL
    assert benchmark.SOAK_CONCURRENCY == EXPECTED_SOAK_CONCURRENCY
    assert benchmark.SOAK_WAVES == EXPECTED_SOAK_WAVES
    assert benchmark.LADDER_REQUESTS_PER_PATH == EXPECTED_LADDER_REQUESTS_PER_PATH
    assert benchmark.SOAK_REQUESTS == EXPECTED_SOAK_REQUESTS
    assert benchmark.BENCHMARK_REQUESTS == EXPECTED_BENCHMARK_REQUESTS
    assert benchmark.MAX_GENERATION_REQUESTS == EXPECTED_MAX_GENERATION_REQUESTS
    assert benchmark.EXPECTED_REQUESTS_PER_WORKER == EXPECTED_REQUESTS_PER_WORKER
    for concurrency in benchmark.CONCURRENCY_LEVELS:
        targets = [
            target
            for wave in range(benchmark.WAVES_PER_LEVEL)
            for target in benchmark._balanced_targets(concurrency, wave)
        ]
        assert targets.count("worker-a") == targets.count("worker-b")


def test_prometheus_snapshot_sums_labeled_series() -> None:
    metrics = """
vllm:request_success_total{finished_reason="stop",model_name="x"} 3
vllm:request_success_total{finished_reason="length",model_name="x"} 2
vllm:prompt_tokens_total{model_name="x"} 41
vllm:generation_tokens_total{model_name="x"} 25
vllm:num_preemptions_total{model_name="x"} 0
vllm:time_to_first_token_seconds_count{model_name="x"} 5
vllm:time_to_first_token_seconds_sum{model_name="x"} 1.5
vllm:e2e_request_latency_seconds_count{model_name="x"} 5
vllm:e2e_request_latency_seconds_sum{model_name="x"} 3.5
vllm:request_queue_time_seconds_count{model_name="x"} 5
vllm:request_queue_time_seconds_sum{model_name="x"} 0.25
vllm:num_requests_running{model_name="x"} 2
vllm:num_requests_waiting{model_name="x"} 1
vllm:kv_cache_usage_perc{model_name="x"} 0.125
""".strip()

    snapshot = benchmark._worker_metric_snapshot(metrics)

    assert snapshot == {
        "request_success": 5,
        "prompt_tokens": 41,
        "generation_tokens": 25,
        "preemptions": 0,
        "time_to_first_token_count": 5,
        "time_to_first_token_sum_seconds": 1.5,
        "e2e_request_latency_count": 5,
        "e2e_request_latency_sum_seconds": 3.5,
        "request_queue_time_count": 5,
        "request_queue_time_sum_seconds": 0.25,
        "requests_running": 2,
        "requests_waiting": 1,
        "kv_cache_usage": 0.125,
    }


@pytest.mark.parametrize("path", ["direct", "arc"])
def test_wave_uses_the_same_frozen_worker_prompts(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    calls: list[dict[str, Any]] = []
    prompts = {"worker-a": "public-a", "worker-b": "public-b"}

    def fake_chat(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        target = "worker-a" if kwargs["prompt"] == prompts["worker-a"] else "worker-b"
        return {
            "latency_seconds": 0.1,
            "response_model": benchmark.WORKERS[target],
            "completion_tokens": 2,
            "selected_worker": target if path == "arc" else "",
        }

    monkeypatch.setattr(benchmark, "_nonstream_chat", fake_chat)
    report = benchmark._run_wave(
        path=path,
        concurrency=TEST_CONCURRENCY,
        wave=0,
        gateway_url="http://gateway",
        worker_urls={"worker-a": "http://a", "worker-b": "http://b"},
        selected_prompts=prompts,
        worker_authorization="Bearer public-test-key",
        run_id="public-unit",
        phase_label="unit",
        timeout_seconds=1,
    )

    assert len(calls) == TEST_CONCURRENCY
    assert {call["prompt"] for call in calls} == set(prompts.values())
    assert report["requests"] == TEST_CONCURRENCY
    assert report["selection_counts"] == {"worker-a": 2, "worker-b": 2}
    assert report["completion_tokens"] == EXPECTED_TEST_COMPLETION_TOKENS
    if path == "arc":
        assert all(call["episode_id"] for call in calls)
        assert all(call["base_url"] == "http://gateway" for call in calls)
    else:
        assert all("episode_id" not in call for call in calls)


def test_worker_metric_gate_rejects_preemption() -> None:
    deltas = {
        worker: {
            "request_success": 40,
            "prompt_tokens": 400,
            "generation_tokens": 320,
            "preemptions": int(worker == "worker-b"),
            "time_to_first_token_count": 40,
            "e2e_request_latency_count": 40,
            "request_queue_time_count": 40,
        }
        for worker in benchmark.WORKERS
    }
    with pytest.raises(RuntimeError, match="preempted"):
        benchmark._validate_worker_metric_deltas(deltas)


def test_ladder_comparison_reports_arc_overhead() -> None:
    phases = []
    for concurrency in benchmark.CONCURRENCY_LEVELS:
        for path, wall, p95 in (("direct", 1.0, 0.5), ("arc", 2.0, 1.25)):
            for wave in range(benchmark.WAVES_PER_LEVEL):
                phases.append(
                    {
                        "path": path,
                        "concurrency": concurrency,
                        "wave": wave,
                        "requests": concurrency,
                        "wall_seconds": wall,
                        "latency": {"p95_seconds": p95, "max_seconds": p95},
                        "completion_tokens": concurrency,
                        "selection_counts": {
                            "worker-a": concurrency // 2,
                            "worker-b": concurrency - (concurrency // 2),
                        },
                    }
                )

    comparison = benchmark._ladder_comparison(phases)

    assert [row["concurrency"] for row in comparison] == [1, 2, 4, 8]
    assert all(
        row["arc_to_direct_throughput_ratio"] == EXPECTED_THROUGHPUT_RATIO
        for row in comparison
    )
    assert all(
        row["arc_p95_overhead_seconds"] == EXPECTED_P95_OVERHEAD_SECONDS
        for row in comparison
    )
    assert all(
        row["arc_to_direct_p95_ratio"] == EXPECTED_P95_RATIO for row in comparison
    )
