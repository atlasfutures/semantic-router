# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

concurrency_test = importlib.import_module("test_rayline_concurrency_packet")
concurrency_packet = importlib.import_module("rayline_concurrency_packet")
open_packet = importlib.import_module("rayline_open_loop_packet")
open_probe = importlib.import_module("rayline_open_loop_probe")
open_loop_contract = importlib.import_module("rayline_open_loop_contract")

# The private source packet PERF020 was actually derived from. It lives under
# the gitignored `.agent-harness/`, so the byte-for-byte reproduction test that
# needs it skips wherever it is absent.
PERF017_PACKET = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf017"
PERF020_PACKET = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf020"


def _source_packet(root: Path) -> Path:
    directional = concurrency_test._directional_packet(root)
    source = root / "concurrency"
    concurrency_packet.build_sweep_packet(
        source_packet_dir=directional, output_dir=source
    )
    return source


def _source_packet_c32(root: Path) -> Path:
    directional = concurrency_test._directional_packet(root)
    source = root / "concurrency-c32"
    concurrency_packet.build_sweep_packet(
        source_packet_dir=directional,
        output_dir=source,
        measured_cases=128,
        warmup_cases=8,
    )
    return source


def test_open_loop_packet_preserves_corpus_and_freezes_rate_identity(
    tmp_path: Path,
) -> None:
    source = _source_packet(tmp_path)
    output = tmp_path / "open-loop"

    manifest = open_packet.build_open_loop_packet(
        source_packet_dir=source, output_dir=output
    )

    assert manifest["measured_cases"] == open_packet.MEASURED_CASES
    assert manifest["corpus_sha256"] == open_packet._sha256(source / "corpus.json")
    assert list(manifest["cells"]) == ["r015", "r030", "r045"]
    assert len({cell["workload_sha256"] for cell in manifest["cells"].values()}) == len(
        open_packet.OFFERED_RATES
    )
    for rate in open_packet.OFFERED_RATES:
        label = open_packet.rate_label(rate)
        warmup, measured, identity, _worker_map, workload = (
            open_probe.load_open_loop_packet(
                arm="rayline_arc",
                corpus_path=output / "corpus.json",
                workload_path=output / f"cells/{label}/workload.json",
                topology_path=output / "topology.json",
                identity_path=output / f"cells/{label}/identity.json",
            )
        )
        assert len(warmup) == open_packet.WARMUP_CASES
        assert len(measured) == open_packet.MEASURED_CASES
        assert identity["measurement_scope"] == open_packet.MEASUREMENT_SCOPE
        assert workload["offered_rate_rps"] == rate


def test_default_rates_still_reproduce_the_frozen_perf020_rungs(
    tmp_path: Path,
) -> None:
    """The rung set became a parameter; the default must not have moved.

    PERF020/PERF021 are closed runs whose recorded digests are the only durable
    statement of what they measured, so the no-flag invocation has to keep
    producing them. A cell's `workload.json` depends only on the rate, the seed
    and the frozen constants -- never on the corpus -- so these digests are the
    part of the packet a hermetic fixture can pin exactly.
    """

    source = _source_packet(tmp_path)

    manifest = open_packet.build_open_loop_packet(
        source_packet_dir=source, output_dir=tmp_path / "default"
    )

    frozen = {
        cell.label: cell.workload_sha256 for cell in open_loop_contract.PERF020.cells
    }
    assert list(manifest["cells"]) == list(frozen)
    assert {
        label: cell["workload_sha256"] for label, cell in manifest["cells"].items()
    } == frozen
    assert open_packet.resolve_offered_rates(None) == open_packet.OFFERED_RATES
    assert open_packet.OFFERED_RATES == (0.15, 0.30, 0.45)


@pytest.mark.skipif(
    not (PERF017_PACKET.is_dir() and PERF020_PACKET.is_dir()),
    reason="private PERF017/PERF020 packets are not present",
)
def test_default_rates_reproduce_perf020_byte_for_byte(tmp_path: Path) -> None:
    output = tmp_path / "perf020"

    manifest = open_packet.build_open_loop_packet(
        source_packet_dir=PERF017_PACKET, output_dir=output
    )

    assert open_packet._sha256(output / "manifest.json") == (
        open_loop_contract.PERF020.packet_manifest_sha256
    )
    assert manifest["corpus_sha256"] == open_loop_contract.PERF020.corpus_sha256
    assert (
        manifest["worker_topology_sha256"] == open_loop_contract.PERF020.topology_sha256
    )
    for cell in open_loop_contract.PERF020.cells:
        assert manifest["cells"][cell.label] == {
            "offered_rate_rps": cell.offered_rate_rps,
            "workload_sha256": cell.workload_sha256,
            "identity_sha256": cell.identity_sha256,
        }


def test_explicit_rungs_reuse_the_shared_workload_derivation(tmp_path: Path) -> None:
    """PERF032's overlap rung is the same cell PERF031B ran, not a lookalike."""

    source = _source_packet(tmp_path)

    manifest = open_packet.build_open_loop_packet(
        source_packet_dir=source,
        output_dir=tmp_path / "knee",
        offered_rates=(0.45, 0.60, 0.90, 1.20),
    )

    assert list(manifest["cells"]) == ["r045", "r060", "r090", "r120"]
    frozen_r045 = next(
        cell for cell in open_loop_contract.PERF020.cells if cell.label == "r045"
    )
    assert manifest["cells"]["r045"]["workload_sha256"] == frozen_r045.workload_sha256
    assert len({cell["workload_sha256"] for cell in manifest["cells"].values()}) == 4


def test_episode_lanes_selects_the_c32_cell_and_its_counts(tmp_path: Path) -> None:
    """A 32-lane packet derives from the c32 cell and carries the full corpus."""

    source = _source_packet_c32(tmp_path)
    output = tmp_path / "open-loop-c32"

    manifest = open_packet.build_open_loop_packet(
        source_packet_dir=source,
        output_dir=output,
        offered_rates=(1.20, 2.40, 4.00, 6.45),
        episode_lanes=32,
    )

    assert manifest["measured_cases"] == 128
    assert manifest["warmup_cases"] == 8
    assert list(manifest["cells"]) == ["r120", "r240", "r400", "r645"]
    for label in manifest["cells"]:
        warmup, measured, identity, _worker_map, workload = (
            open_probe.load_open_loop_packet(
                arm="rayline_arc",
                corpus_path=output / "corpus.json",
                workload_path=output / f"cells/{label}/workload.json",
                topology_path=output / "topology.json",
                identity_path=output / f"cells/{label}/identity.json",
            )
        )
        assert workload["max_episode_lanes"] == 32
        assert len(warmup) == 8
        assert len(measured) == 128
        assert identity["case_count"] == 128


def test_unregistered_episode_lanes_are_refused_before_any_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "open-loop"

    with pytest.raises(
        open_packet.OpenLoopPacketError, match="no registered sweep workload"
    ):
        open_packet.build_open_loop_packet(
            source_packet_dir=tmp_path, output_dir=output, episode_lanes=16
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("rates", "message"),
    [
        ((), "at least one"),
        ((0.45, 0.45), "distinct"),
        ((0.90, 0.45), "ascend"),
        ((0.455,), "two-decimal"),
        ((0.0,), "two-decimal"),
    ],
)
def test_rung_sets_are_validated_before_any_file_is_written(
    rates: tuple[float, ...], message: str
) -> None:
    with pytest.raises(open_packet.OpenLoopPacketError, match=message):
        open_packet.resolve_offered_rates(rates)


def test_open_loop_packet_refuses_overwrite(tmp_path: Path) -> None:
    source = _source_packet(tmp_path)
    output = tmp_path / "open-loop"
    output.mkdir()

    with pytest.raises(open_packet.OpenLoopPacketError, match="already exists"):
        open_packet.build_open_loop_packet(source_packet_dir=source, output_dir=output)
