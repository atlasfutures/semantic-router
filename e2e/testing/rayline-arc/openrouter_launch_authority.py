# SPDX-License-Identifier: Apache-2.0

"""Source-closed authority pins for bounded paid OpenRouter launchers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from openrouter_kv_cache_agt019_contract import (
    AUTHORIZATION_COMMIT as AGT019_AUTHORIZATION_COMMIT,
)
from openrouter_kv_cache_agt019_contract import (
    PREREGISTRATION_COMMIT as AGT019_PREREGISTRATION_COMMIT,
)
from openrouter_kv_cache_matched_contract import (
    AUTHORIZATION_COMMIT as AGT017_AUTHORIZATION_COMMIT,
)
from openrouter_kv_cache_matched_contract import (
    PREREGISTRATION_COMMIT as AGT017_PREREGISTRATION_COMMIT,
)
from openrouter_kv_cache_successor_contract import (
    AUTHORIZATION_COMMIT as AGT018_AUTHORIZATION_COMMIT,
)
from openrouter_kv_cache_successor_contract import (
    PREREGISTRATION_COMMIT as AGT018_PREREGISTRATION_COMMIT,
)

GIT_SHA1_HEX_LENGTH = 40
SOURCE_REMOTE_REF = "atlasfutures/codex/rayline-remote-mvp"
AUTHORITY_PINS = {
    "agentic": ("", ""),
    "agentic-stage": ("", ""),
    "kv-cache": ("", ""),
    "kv-cache-flashinfer": (
        AGT017_PREREGISTRATION_COMMIT,
        AGT017_AUTHORIZATION_COMMIT,
    ),
    "kv-cache-flashinfer-agt018": (
        AGT018_PREREGISTRATION_COMMIT,
        AGT018_AUTHORIZATION_COMMIT,
    ),
    "kv-cache-flashinfer-agt019": (
        AGT019_PREREGISTRATION_COMMIT,
        AGT019_AUTHORIZATION_COMMIT,
    ),
    "gateway-shape": ("", ""),
    "gateway-prime": ("", ""),
}


def _git(
    repo_root: Path,
    environment: dict[str, str],
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def verify_source_authority(
    mode: str,
    environment: dict[str, str],
    *,
    repo_root: Path,
) -> None:
    pins = AUTHORITY_PINS.get(mode)
    if pins is None:
        return
    if any(len(pin) != GIT_SHA1_HEX_LENGTH for pin in pins):
        raise SystemExit(f"{mode} launch authority is source-closed")
    if _git(repo_root, environment, "status", "--porcelain").stdout:
        raise SystemExit(f"{mode} launch requires a clean source checkpoint")
    head = _git(repo_root, environment, "rev-parse", "HEAD").stdout.strip()
    for pin in pins:
        ancestor = _git(
            repo_root,
            environment,
            "merge-base",
            "--is-ancestor",
            pin,
            head,
            check=False,
        )
        if ancestor.returncode != 0:
            raise SystemExit(f"{mode} authority pin is not in launched source")
    remote = _git(
        repo_root,
        environment,
        "rev-parse",
        SOURCE_REMOTE_REF,
    ).stdout.strip()
    pushed = _git(
        repo_root,
        environment,
        "merge-base",
        "--is-ancestor",
        head,
        remote,
        check=False,
    )
    if pushed.returncode != 0:
        raise SystemExit(f"{mode} launch source is not remote-visible")
