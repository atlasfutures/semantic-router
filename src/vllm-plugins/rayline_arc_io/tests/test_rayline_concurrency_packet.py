# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

directional = importlib.import_module("test_rayline_parity_packet")
packet = importlib.import_module("rayline_parity_packet")
probe = importlib.import_module("rayline_parity_http_probe")
sweep = importlib.import_module("rayline_concurrency_packet")


def _directional_packet(root: Path) -> Path:
    corpus, manifest, runtime = directional._source_packet(root)
    output = root / "directional"
    packet.build_packet(
        source_corpus_path=corpus,
        source_manifest_path=manifest,
        runtime_dir=runtime,
        output_dir=output,
        placement_profile="modal-us-east-public-https",
        gpu_class="NVIDIA H100 80GB",
    )
    return output


def test_sweep_packet_preserves_episodes_buckets_and_frozen_cells(
    tmp_path: Path,
) -> None:
    source = _directional_packet(tmp_path)
    output = tmp_path / "sweep"

    receipt = sweep.build_sweep_packet(
        source_packet_dir=source,
        output_dir=output,
    )

    assert receipt["measured_cases"] == sweep.MEASURED_CASES
    assert receipt["measured_episodes"] == (
        sweep.MEASURED_CASES // sweep.DECISIONS_PER_EPISODE
    )
    assert set(receipt["input_token_buckets"]) == set(probe.INPUT_TOKEN_BUCKET_NAMES)
    assert sum(receipt["input_token_buckets"].values()) == sweep.MEASURED_CASES
    # The registry also carries cells for other case counts (the PERF034 c32
    # cell); the default build must produce exactly the frozen PERF017 trio.
    assert set(receipt["cells"]) == {"c1", "c4", "c8"}
    for label in receipt["cells"]:
        contract = probe.SWEEP_WORKLOADS[int(label.removeprefix("c"))]
        cell = output / "cells" / label
        warmup, measured, identity, _worker_map = probe.load_packet(
            arm="rayline_arc",
            corpus_path=output / "corpus.json",
            workload_path=cell / "workload.json",
            topology_path=output / "topology.json",
            identity_path=cell / "identity.json",
            workload_contract=contract,
        )
        assert len(warmup) == sweep.WARMUP_CASES
        assert len(measured) == sweep.MEASURED_CASES
        assert identity["measurement_scope"].endswith("concurrency_sweep")


def test_cap_raise_counts_select_only_the_c32_cell(tmp_path: Path) -> None:
    """The 128-case build carries every lane of the directional corpus.

    The full-corpus counts must select exactly the PERF034 c32 cell: the
    32-case cells cannot coexist with it in one packet because a packet has
    one corpus, and the corpus either is or is not the full 128 cases.
    """

    source = _directional_packet(tmp_path)
    output = tmp_path / "sweep-c32"

    receipt = sweep.build_sweep_packet(
        source_packet_dir=source,
        output_dir=output,
        measured_cases=128,
        warmup_cases=8,
    )

    assert set(receipt["cells"]) == {"c32"}
    assert receipt["measured_cases"] == 128
    assert receipt["warmup_cases"] == 8
    assert receipt["measured_episodes"] == 128 // sweep.DECISIONS_PER_EPISODE
    assert receipt["warmup_episodes"] == 8 // sweep.DECISIONS_PER_EPISODE
    assert sum(receipt["input_token_buckets"].values()) == 128
    cell = output / "cells/c32"
    warmup, measured, identity, _worker_map = probe.load_packet(
        arm="rayline_arc",
        corpus_path=output / "corpus.json",
        workload_path=cell / "workload.json",
        topology_path=output / "topology.json",
        identity_path=cell / "identity.json",
        workload_contract=probe.SWEEP_WORKLOADS[32],
    )
    assert len(warmup) == 8
    assert len(measured) == 128
    assert identity["case_count"] == 128


def test_unregistered_case_counts_are_refused_before_any_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sweep"

    with pytest.raises(sweep.SweepPacketError, match="no registered sweep workload"):
        sweep.build_sweep_packet(
            source_packet_dir=tmp_path,
            output_dir=output,
            measured_cases=64,
            warmup_cases=4,
        )
    assert not output.exists()


def test_directional_profile_rejects_a_sweep_cell(tmp_path: Path) -> None:
    source = _directional_packet(tmp_path)
    output = tmp_path / "sweep"
    sweep.build_sweep_packet(source_packet_dir=source, output_dir=output)
    cell = output / "cells/c1"

    with pytest.raises(probe.ProbeError, match="requires concurrency 8"):
        probe.load_packet(
            arm="rayline_remote",
            corpus_path=output / "corpus.json",
            workload_path=cell / "workload.json",
            topology_path=output / "topology.json",
            identity_path=cell / "identity.json",
        )


def test_sweep_packet_refuses_overwrite(tmp_path: Path) -> None:
    source = _directional_packet(tmp_path)
    output = tmp_path / "sweep"
    output.mkdir()

    with pytest.raises(sweep.SweepPacketError, match="already exists"):
        sweep.build_sweep_packet(source_packet_dir=source, output_dir=output)


def test_cell_identity_digest_tracks_workload(tmp_path: Path) -> None:
    source = _directional_packet(tmp_path)
    output = tmp_path / "sweep"
    sweep.build_sweep_packet(source_packet_dir=source, output_dir=output)
    c1 = json.loads((output / "cells/c1/identity.json").read_text())
    c8 = json.loads((output / "cells/c8/identity.json").read_text())

    assert c1["corpus_sha256"] == c8["corpus_sha256"]
    assert c1["workload_sha256"] != c8["workload_sha256"]
