"""CI configuration tests for the E2E cadence split (plan PL-0042, DPC-105).

The PR gate runs a compact smoke tier and the nightly run covers the full
matrix. Which test cases each cadence selects lives in ``e2e/pkg/testmatrix``,
not in workflow YAML: these tests assert the workflows keep deriving it from
there, and that the nightly profile list stays equal to the registry's full-CI
set instead of drifting as a second hand-maintained copy.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from test_domain_registry import profile_records  # noqa: E402

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
E2E_WORKFLOW = WORKFLOWS / "integration-test-k8s.yml"
TESTMATRIX = REPO_ROOT / "e2e" / "pkg" / "testmatrix" / "citiers.go"


def load_workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML parses the bare ``on:`` key as the boolean True.
    if True in data:
        data["on"] = data.pop(True)
    return data


def e2e_job(path: Path) -> dict:
    return load_workflow(path)["jobs"]["e2e"]


class E2ECadenceGateTests(unittest.TestCase):
    def test_pull_requests_run_the_smoke_cadence(self) -> None:
        job = e2e_job(WORKFLOWS / "pr.yml")

        self.assertEqual(job["uses"], "./.github/workflows/integration-test-k8s.yml")
        self.assertEqual(job["with"]["cadence"], "pr")

    def test_nightly_runs_the_full_cadence(self) -> None:
        job = e2e_job(WORKFLOWS / "nightly-build.yml")

        self.assertEqual(job["with"]["cadence"], "nightly")

    def test_nightly_profiles_match_the_registry_full_ci_set(self) -> None:
        job = e2e_job(WORKFLOWS / "nightly-build.yml")
        nightly = set(json.loads(job["with"]["profiles"]))
        full_ci = {
            name for name, data in profile_records().items() if data.get("full_ci")
        }

        self.assertEqual(nightly, full_ci)

    def test_protocol_conformance_reaches_the_pr_gate(self) -> None:
        record = profile_records()["protocol-conformance"]

        self.assertEqual(record["selection"], "pr")
        self.assertTrue(record["full_ci"])

    def test_workflow_derives_test_cases_instead_of_listing_them(self) -> None:
        text = E2E_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("-list-tests", text)
        # The baseline subset used to be pasted into this workflow. It now lives
        # in e2e/pkg/testmatrix and must not come back.
        self.assertNotIn("ENVOY_AI_GATEWAY_CI_TESTS", text)
        self.assertNotIn("decision-priority-selection", text)

    def test_testmatrix_owns_the_cadence_subsets(self) -> None:
        text = TESTMATRIX.read_text(encoding="utf-8")

        self.assertIn("protocol-conformance-smoke", text)
        self.assertIn("decision-priority-selection", text)


if __name__ == "__main__":
    unittest.main()
