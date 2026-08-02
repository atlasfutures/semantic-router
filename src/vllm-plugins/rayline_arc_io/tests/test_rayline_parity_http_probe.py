# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

probe = importlib.import_module("rayline_parity_http_probe")
MEASURED_CASES = 128
TOTAL_CASES = 136
MEASURED_AFTER_ONE_FAILURE = 127


class FakeClient:
    def __init__(self, arm: str, *, fail_case: str = "") -> None:
        self.arm = arm
        self.fail_case = fail_case
        self.commits = 0
        self.settles = 0
        self.last_headers: dict[str, str] = {}

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str], float]:
        self.last_headers = dict(headers or {})
        if path == "/v1/route/capabilities":
            return (
                200,
                {
                    "bundle_version": "test-bundle",
                    "workers": ["remote-a", "remote-b"],
                },
                {},
                0.001,
            )
        if path == "/v1/route/commit":
            self.commits += 1
            return 200, {"state": "committed"}, {}, 0.001
        if path == "/v1/route/settle":
            self.settles += 1
            return 200, {"state": "settled"}, {}, 0.001
        assert body is not None
        case_id = self._case_id(body, headers or {})
        if case_id == self.fail_case:
            return 503, {}, {}, 0.003
        suffix = "a" if int(case_id[-3:]) % 2 == 0 else "b"
        observed = f"{self.arm}-{suffix}"
        if path == "/v1/route":
            return 200, {"selected_worker": observed}, {}, 0.010
        if path == "/v1/route/prepare":
            return (
                200,
                {
                    "state": "prepared",
                    "selected_worker": observed,
                    "receipt": "opaque",
                },
                {},
                0.011,
            )
        if path == "/v1/chat/completions":
            return 200, {}, {"x-vsr-selected-model": observed}, 0.012
        raise AssertionError(path)

    @staticmethod
    def _case_id(body: dict[str, Any], headers: dict[str, str]) -> str:
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            return str(metadata["rayline_task_id"])
        decision_id = str(body.get("decision_id") or "")
        if decision_id:
            return decision_id.rsplit(":", 1)[-1]
        return headers["x-rayline-route-id"].rsplit(":", 1)[-1]


def _case(prefix: str, index: int) -> dict[str, Any]:
    return {
        "case_id": f"{prefix}{index:03d}",
        "episode_id": f"public-{prefix}-episode-{index // 4:03d}",
        "input_tokens": index + 1,
        "messages": [
            {
                "role": "user",
                "content": f"Public synthetic routing case {prefix}-{index}.",
            }
        ],
    }


def _write_packet(root: Path) -> dict[str, Path]:
    corpus = root / "corpus.json"
    workload = root / "workload.json"
    topology = root / "topology.json"
    identity = root / "identity.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": probe.CORPUS_SCHEMA,
                "warmup": [_case("w", index) for index in range(8)],
                "measured": [_case("m", index) for index in range(128)],
            },
            sort_keys=True,
        )
    )
    workload.write_text(
        json.dumps(
            {
                "schema_version": probe.WORKLOAD_SCHEMA,
                "concurrency": 8,
                "warmup_cases": 8,
                "measured_cases": 128,
                "seed": 20260730,
            },
            sort_keys=True,
        )
    )
    topology.write_text(
        json.dumps(
            {
                "schema_version": probe.TOPOLOGY_SCHEMA,
                "canonical_workers": ["worker-a", "worker-b"],
                "arm_worker_maps": {
                    "modal_inprocess": {
                        "modal_inprocess-a": "worker-a",
                        "modal_inprocess-b": "worker-b",
                    },
                    "rayline_remote": {
                        "rayline_remote-a": "worker-a",
                        "rayline_remote-b": "worker-b",
                    },
                    "rayline_arc": {
                        "rayline_arc-a": "worker-a",
                        "rayline_arc-b": "worker-b",
                    },
                },
            },
            sort_keys=True,
        )
    )
    identity.write_text(
        json.dumps(
            {
                "measurement_scope": "architecture_decision_boundary",
                "case_count": 128,
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
                "encoder_model": "Qwen/Qwen3.5-0.8B",
                "encoder_revision": "public-revision",
                "tokenizer_sha256": "a" * 64,
                "serializer_version": "mtrouter-token-blocks-v2",
                "policy_artifact_revision": "public-artifact-revision",
                "gpu_class": "NVIDIA H100 80GB",
                "worker_topology_sha256": hashlib.sha256(
                    topology.read_bytes()
                ).hexdigest(),
                "placement_profile": "modal-us-east-public-https",
                "warm_state": "warm",
                "seed": 20260730,
            },
            sort_keys=True,
        )
    )
    return {
        "corpus": corpus,
        "workload": workload,
        "topology": topology,
        "identity": identity,
    }


def _loaded(root: Path, arm: str) -> tuple[Any, ...]:
    paths = _write_packet(root)
    return probe.load_packet(
        arm=arm,
        corpus_path=paths["corpus"],
        workload_path=paths["workload"],
        topology_path=paths["topology"],
        identity_path=paths["identity"],
    )


@pytest.mark.parametrize(
    ("input_tokens", "expected"),
    [
        (8191, "lt_8192"),
        (8192, "8192_to_32767"),
        (32767, "8192_to_32767"),
        (32768, "32768_to_131071"),
        (131071, "32768_to_131071"),
        (131072, "gte_131072"),
    ],
)
def test_input_token_bucket_boundaries(input_tokens: int, expected: str) -> None:
    assert probe._input_token_bucket(input_tokens) == expected


def test_run_id_namespaces_remote_and_arc_episode_state() -> None:
    case = probe.Case(
        "public-case-000",
        "public-episode",
        100,
        ({"role": "user", "content": "Public synthetic request."},),
    )
    first_key = probe._episode_key(case, 20260730, "perf017-c1")
    second_key = probe._episode_key(case, 20260730, "perf017-c8")
    client = FakeClient("rayline_arc")

    probe._select_arc(client, case, "perf017-c1")

    assert first_key.startswith("hmac-sha256:")
    assert first_key != second_key
    assert client.last_headers["x-rayline-episode-id"] == ("perf017-c1:public-episode")


@pytest.mark.parametrize(
    ("arm", "protocol"),
    list(probe.PROTOCOL_BY_ARM.items()),
)
def test_each_registered_protocol_emits_the_same_canonical_trace(
    tmp_path: Path,
    arm: str,
    protocol: str,
) -> None:
    warmup, measured, identity, worker_map = _loaded(tmp_path, arm)
    client = FakeClient(arm)

    receipt = probe.run_probe(
        arm=arm,
        protocol=protocol,
        client=client,
        warmup=warmup,
        measured=measured,
        identity=identity,
        worker_map=worker_map,
        run_id=f"test-{arm}",
    )

    assert receipt["results"]["completed"] == MEASURED_CASES
    assert receipt["results"]["failed"] == 0
    assert receipt["results"]["provider_calls"] == 0
    buckets = receipt["results"]["latency_by_input_tokens"]
    assert buckets["lt_8192"]["completed"] == MEASURED_CASES
    assert buckets["8192_to_32767"]["selection_latency_seconds"] is None
    if arm == "rayline_remote":
        assert client.commits == TOTAL_CASES
        assert client.settles == TOTAL_CASES


def test_packet_rejects_digest_drift(tmp_path: Path) -> None:
    paths = _write_packet(tmp_path)
    paths["corpus"].write_text(paths["corpus"].read_text() + "\n")

    with pytest.raises(probe.ProbeError, match="corpus_sha256"):
        probe.load_packet(
            arm="rayline_arc",
            corpus_path=paths["corpus"],
            workload_path=paths["workload"],
            topology_path=paths["topology"],
            identity_path=paths["identity"],
        )


def test_failed_request_is_retained_as_aggregate_evidence(tmp_path: Path) -> None:
    arm = "rayline_arc"
    warmup, measured, identity, worker_map = _loaded(tmp_path, arm)

    receipt = probe.run_probe(
        arm=arm,
        protocol=probe.PROTOCOL_BY_ARM[arm],
        client=FakeClient(arm, fail_case="m007"),
        warmup=warmup,
        measured=measured,
        identity=identity,
        worker_map=worker_map,
        run_id="test-failure",
    )

    assert receipt["results"]["completed"] == MEASURED_AFTER_ONE_FAILURE
    assert receipt["results"]["failed"] == 1


def test_arm_protocol_mismatch_fails_before_requests(tmp_path: Path) -> None:
    arm = "rayline_arc"
    warmup, measured, identity, worker_map = _loaded(tmp_path, arm)

    with pytest.raises(probe.ProbeError, match="not registered"):
        probe.run_probe(
            arm=arm,
            protocol="modal_route",
            client=FakeClient(arm),
            warmup=warmup,
            measured=measured,
            identity=identity,
            worker_map=worker_map,
            run_id="test-mismatch",
        )


def test_measured_turns_never_overlap_within_one_episode(tmp_path: Path) -> None:
    class EpisodeFenceClient(FakeClient):
        def __init__(self) -> None:
            super().__init__("modal_inprocess")
            self._lock = threading.Lock()
            self._active: set[str] = set()

        def request(
            self,
            method: str,
            path: str,
            *,
            body: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, Any], dict[str, str], float]:
            assert body is not None
            metadata = body["metadata"]
            episode = str(metadata["rayline_session_id"])
            with self._lock:
                if episode in self._active:
                    return 409, {}, {}, 0.001
                self._active.add(episode)
            try:
                time.sleep(0.001)
                return super().request(
                    method,
                    path,
                    body=body,
                    headers=headers,
                )
            finally:
                with self._lock:
                    self._active.remove(episode)

    arm = "modal_inprocess"
    warmup, measured, identity, worker_map = _loaded(tmp_path, arm)
    receipt = probe.run_probe(
        arm=arm,
        protocol=probe.PROTOCOL_BY_ARM[arm],
        client=EpisodeFenceClient(),
        warmup=warmup,
        measured=measured,
        identity=identity,
        worker_map=worker_map,
        run_id="test-episode-order",
    )

    assert receipt["results"]["completed"] == MEASURED_CASES
    assert receipt["results"]["failed"] == 0
