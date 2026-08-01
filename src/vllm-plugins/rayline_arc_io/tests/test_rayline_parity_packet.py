# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

artifact_fixture = importlib.import_module("artifact_fixture")
packet = importlib.import_module("rayline_parity_packet")
probe = importlib.import_module("rayline_parity_http_probe")


def _source_packet(root: Path) -> tuple[Path, Path, Path]:
    runtime = root / "runtime"
    artifact_fixture.generate(runtime)
    runtime_manifest = json.loads((runtime / "manifest.json").read_text())
    workers = [worker["id"] for worker in runtime_manifest["workers"]]

    corpus = root / "source.jsonl"
    rows = []
    for index in range(packet.MEASURED_CASES + packet.WARMUP_CASES):
        episode = index // packet.DECISIONS_PER_EPISODE
        sequence = index % packet.DECISIONS_PER_EPISODE
        rows.append(
            {
                "schema_version": packet.SOURCE_CASE_SCHEMA,
                "decision_id": f"parity-{index:06d}",
                "episode_affinity_hash": f"public-episode-{episode:03d}",
                "sequence_index": sequence,
                "shape": "short_chat",
                "previous_worker": "",
                "turns": [
                    {
                        "role": "user",
                        "text": f"Public synthetic case {index}.",
                    }
                ],
                "token_counts": {
                    "prefix": index + 1,
                    "new_turn": 1,
                    "full": index + 2,
                    "truncated": 0,
                },
            }
        )
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    encoder = runtime_manifest["encoder"]
    manifest = root / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": packet.SOURCE_CORPUS_SCHEMA,
                "seed": 20260730,
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "model": encoder["model"],
                "model_revision": encoder["revision"],
                "tokenizer_sha256": "a" * 64,
                "serializer_version": encoder["serialization"],
                "worker_order": workers,
                "contains_customer_data": False,
            }
        )
    )
    return corpus, manifest, runtime


def test_packet_is_probe_ready_and_preserves_complete_episodes(tmp_path: Path) -> None:
    corpus, manifest, runtime = _source_packet(tmp_path)
    output = tmp_path / "packet"

    receipt = packet.build_packet(
        source_corpus_path=corpus,
        source_manifest_path=manifest,
        runtime_dir=runtime,
        output_dir=output,
        placement_profile="modal-us-east-public-https",
        gpu_class="NVIDIA H100 80GB",
    )

    assert receipt["case_count"] == 128
    warmup, measured, identity, worker_map = probe.load_packet(
        arm="rayline_arc",
        corpus_path=output / "corpus.json",
        workload_path=output / "workload.json",
        topology_path=output / "topology.json",
        identity_path=output / "identity.json",
    )
    assert len(warmup) == 8
    assert len(measured) == 128
    assert len({case.episode_id for case in measured}) == 32
    assert not (
        {case.episode_id for case in measured} & {case.episode_id for case in warmup}
    )
    assert identity["policy_artifact_revision"].startswith("sha256:")
    assert set(worker_map) == set(worker_map.values())


def test_packet_rejects_source_digest_drift(tmp_path: Path) -> None:
    corpus, manifest, runtime = _source_packet(tmp_path)
    corpus.write_text(corpus.read_text() + "\n")

    with pytest.raises(packet.PacketError, match="source corpus digest"):
        packet.build_packet(
            source_corpus_path=corpus,
            source_manifest_path=manifest,
            runtime_dir=runtime,
            output_dir=tmp_path / "packet",
            placement_profile="modal-us-east-public-https",
            gpu_class="NVIDIA H100 80GB",
        )


def test_packet_refuses_overwrite(tmp_path: Path) -> None:
    corpus, manifest, runtime = _source_packet(tmp_path)
    output = tmp_path / "packet"
    output.mkdir()

    with pytest.raises(packet.PacketError, match="already exists"):
        packet.build_packet(
            source_corpus_path=corpus,
            source_manifest_path=manifest,
            runtime_dir=runtime,
            output_dir=output,
            placement_profile="modal-us-east-public-https",
            gpu_class="NVIDIA H100 80GB",
        )
