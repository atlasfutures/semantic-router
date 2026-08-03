#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Remote-vLLM entrypoint for the AGT016 retained-versus-replay workload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openrouter_kv_cache_benchmark import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    args.deployment = "remote_vllm"
    args.base_url = args.gateway_url
    output_dir = (
        Path(__file__).resolve().parents[3]
        / ".agent-harness/rayline-kv-cache"
        / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    for name in (
        "OPENROUTER_EPHEMERAL_API_KEY",
        "RAYLINE_ARC_E2E_MODAL_KEY",
        "RAYLINE_ARC_E2E_MODAL_SECRET",
    ):
        value = os.environ.get(name, "")
        if value and value in encoded:
            raise RuntimeError("credential entered remote KV comparison report")
    (output_dir / "remote-client.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
