#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build the aggregate-only AGT019 comparison receipt from private inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openrouter_kv_cache_agt019_reporting import build_report
from openrouter_modal_native_benchmark import read_decisions


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "native-client",
        "native-decisions",
        "native-deployment",
        "native-provider-preflight",
        "native-key-usage",
        "remote-client",
        "remote-deployment",
        "remote-provider-preflight",
        "remote-key-usage",
        "offline-coverage",
        "cleanup-receipt",
        "budget-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("AGT019 report input must be a JSON object")
    return decoded


def main() -> None:
    args = _args()
    report = build_report(
        native=_load(args.native_client),
        decisions=read_decisions(Path(args.native_decisions)),
        remote=_load(args.remote_client),
        native_deployment=_load(args.native_deployment),
        remote_deployment=_load(args.remote_deployment),
        native_preflight=_load(args.native_provider_preflight),
        remote_preflight=_load(args.remote_provider_preflight),
        offline_coverage=_load(args.offline_coverage),
        cleanup_receipt=_load(args.cleanup_receipt),
        native_key_usage=float(_load(args.native_key_usage)["usage_usd"]),
        remote_key_usage=float(_load(args.remote_key_usage)["usage_usd"]),
        budget_receipt=_load(args.budget_receipt),
    )
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
