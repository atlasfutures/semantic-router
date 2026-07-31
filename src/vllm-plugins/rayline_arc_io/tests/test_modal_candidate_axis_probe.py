# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

probe = importlib.import_module("modal_candidate_axis_probe")
CANDIDATE_PROMPTS = importlib.import_module("modal_fullstack_inputs").CANDIDATE_PROMPTS

BALANCED_AXIS = 7
UNBALANCED_AXIS = 9
UNBALANCED_SPLIT = 8
MIN_EXPECTED_MARGIN = 0.2


def _vectors() -> list[tuple[float, ...]]:
    vectors: list[tuple[float, ...]] = []
    for index in range(len(CANDIDATE_PROMPTS)):
        values = [0.0] * probe.EMBEDDING_DIMENSION
        values[0] = 0.001 if index % 2 else -0.001
        values[BALANCED_AXIS] = 0.25 if index % 2 else -0.25
        values[UNBALANCED_AXIS] = 0.5 if index < UNBALANCED_SPLIT else -0.5
        values[11] = 1.0
        vectors.append(tuple(values))
    return vectors


class _FakeClient:
    def __init__(self, vectors: list[tuple[float, ...]]) -> None:
        self.vectors = iter(vectors)
        self.opened: set[str] = set()
        self.maximum_opened = 0

    def encode_with_embedding(self, episode_id, turns):
        assert turns[0]["text"] in CANDIDATE_PROMPTS
        self.opened.add(episode_id)
        self.maximum_opened = max(self.maximum_opened, len(self.opened))
        return {"latency_seconds": 0.01}, next(self.vectors)

    def close(self, episode_id):
        self.opened.remove(episode_id)

    def request(self, method, path):
        assert (method, path) == ("GET", "/health")
        return {"resident_sessions": len(self.opened)}, 0.01


def test_axis_selection_prioritizes_balance_then_robust_margin() -> None:
    selected = probe.select_axis(_vectors())

    assert selected["axis_index"] == BALANCED_AXIS
    assert selected["sign_counts"] == {"negative": 12, "positive": 12, "zero": 0}
    assert selected["absolute_margin"]["minimum"] > MIN_EXPECTED_MARGIN


def test_probe_report_is_aggregate_and_closes_sessions() -> None:
    client = _FakeClient(_vectors())
    report = probe.run_probe(client, "public-axis-unit")
    encoded = json.dumps(report)

    assert report["status"] == "passed"
    assert report["candidate_count"] == len(CANDIDATE_PROMPTS)
    assert report["resident_sessions_after_close"] == 0
    assert report["raw_embeddings_emitted"] is False
    assert report["prompt_text_emitted"] is False
    assert not client.opened
    assert client.maximum_opened == 1
    assert all(prompt not in encoded for prompt in CANDIDATE_PROMPTS)


def test_launcher_is_bounded_and_stops_only_the_encoder_app_containers() -> None:
    launcher = (SCRIPT_DIR / "run_modal_candidate_axis_probe.py").read_text()

    assert 'REQUIRED_MODAL_VERSION = "1.5.1"' in launcher
    assert "MAX_PROBE_SECONDS = 12 * 60" in launcher
    assert 'ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"' in launcher
    assert '"container", "stop", container_id, "--yes"' in launcher
    assert "manager.delete(proxy_token.token_id)" in launcher
    assert "1000" not in launcher
