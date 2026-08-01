#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Stage an immutable ARC runtime for zero-cost worker-double dispatch.

The numerical head, policy, costs, source lineage, and goldens are copied
byte-for-byte. Only the worker transport contract and artifact envelope are
rewritten. This tool is schema-generic: private runtime pins and worker data
remain outside the repository and outside its stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

MANIFEST_SCHEMA = "rayline.mtrouter-runtime.v3"
TRANSFORM_SCHEMA = "rayline.arc.development-worker-double.v1"


class ArtifactError(ValueError):
    """The source runtime or requested development transform is invalid."""


def _read_manifest(runtime_dir: Path) -> dict[str, Any]:
    path = runtime_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot read source manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ArtifactError("source manifest must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ArtifactError("source manifest schema is unsupported")
    workers = manifest.get("workers")
    if not isinstance(workers, list) or len(workers) < 2:
        raise ArtifactError("source manifest must contain at least two workers")
    return manifest


def _safe_registered_file(runtime_dir: Path, raw_path: object) -> Path:
    relative = PurePosixPath(str(raw_path))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ArtifactError("registered artifact path must be relative and contained")
    path = runtime_dir.joinpath(*relative.parts)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ArtifactError(
            f"registered artifact file is unavailable: {relative}"
        ) from error
    if not stat.S_ISREG(mode):
        raise ArtifactError(
            f"registered artifact path is not a regular file: {relative}"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registered_files(
    runtime_dir: Path,
    manifest: Mapping[str, Any],
) -> list[tuple[Path, PurePosixPath]]:
    weights = manifest.get("weights")
    golden = manifest.get("golden")
    encoder = manifest.get("encoder")
    if not all(isinstance(value, Mapping) for value in (weights, golden, encoder)):
        raise ArtifactError("source manifest omits weights/golden/encoder sections")
    head = golden.get("head")
    if not isinstance(head, Mapping):
        raise ArtifactError("source manifest omits the head golden")
    registrations = [weights, head]
    encoder_golden = encoder.get("golden")
    if encoder_golden is not None:
        if not isinstance(encoder_golden, Mapping):
            raise ArtifactError("encoder golden registration is invalid")
        registrations.append(encoder_golden)

    result: list[tuple[Path, PurePosixPath]] = []
    seen: set[PurePosixPath] = set()
    for registration in registrations:
        relative = PurePosixPath(str(registration.get("file") or ""))
        path = _safe_registered_file(runtime_dir, relative)
        expected = str(registration.get("sha256") or "")
        if len(expected) != 64 or _sha256(path) != expected:
            raise ArtifactError(f"registered artifact digest differs: {relative}")
        if relative in seen:
            raise ArtifactError(f"duplicate registered artifact path: {relative}")
        seen.add(relative)
        result.append((path, relative))
    return result


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ArtifactError("worker base URL must be absolute, credential-free HTTP(S)")
    return value.rstrip("/")


def _development_worker(
    raw_worker: object,
    *,
    base_url: str,
    api_key_env: str,
) -> dict[str, Any]:
    if not isinstance(raw_worker, Mapping):
        raise ArtifactError("source worker must be an object")
    worker = dict(raw_worker)
    source_mode = str(worker.get("thinking_mode") or "")
    mode_by_source = {
        "on": "on",
        "high": "on",
        "off": "off",
        "disabled": "off",
    }
    mode = mode_by_source.get(source_mode)
    if mode is None:
        raise ArtifactError("source worker thinking mode is invalid")
    worker.update(
        {
            "api_key_env": api_key_env,
            "dispatch_backend": "openai_compatible",
            "provider_base_url": base_url,
            "thinking_mode": mode,
            "openrouter_provider_slug": "",
            "openrouter_provider_name": "",
            "openrouter_provider_order": [],
            "openrouter_allow_fallbacks": False,
            "openrouter_require_parameters": False,
            "openrouter_pricing_source": "",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": mode == "on"}},
        }
    )
    return worker


def stage_artifact(
    source_dir: Path,
    output_dir: Path,
    *,
    artifact_id: str,
    created_at: str,
    exporter_revision: str,
    worker_base_url: str,
    api_key_env: str,
) -> dict[str, Any]:
    if not artifact_id or not created_at or not exporter_revision:
        raise ArtifactError("artifact envelope values must be non-empty")
    if not api_key_env or not api_key_env.replace("_", "A").isalnum():
        raise ArtifactError("api_key_env must be an environment-variable name")
    if output_dir.exists():
        raise ArtifactError("output directory already exists")
    manifest = _read_manifest(source_dir)
    files = _registered_files(source_dir, manifest)
    base_url = _validate_base_url(worker_base_url)

    staged = json.loads(json.dumps(manifest))
    staged["artifact_id"] = artifact_id
    staged["created_at"] = created_at
    staged["exporter_commit"] = exporter_revision
    staged["workers"] = [
        _development_worker(
            worker,
            base_url=base_url,
            api_key_env=api_key_env,
        )
        for worker in manifest["workers"]
    ]

    output_dir.mkdir(parents=True)
    for source, relative in files:
        destination = output_dir.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    rendered = json.dumps(staged, indent=2, sort_keys=True) + "\n"
    (output_dir / "manifest.json").write_text(rendered)
    return {
        "schema_version": TRANSFORM_SCHEMA,
        "artifact_id": artifact_id,
        "worker_count": len(staged["workers"]),
        "manifest_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "registered_file_count": len(files),
        "source_checkpoint_sha256": staged["source"]["checkpoint"]["sha256"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--exporter-revision", required=True)
    parser.add_argument("--worker-base-url", required=True)
    parser.add_argument("--api-key-env", default="RAYLINE_PARITY_WORKER_KEY")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        receipt = stage_artifact(
            args.source_dir,
            args.output_dir,
            artifact_id=args.artifact_id,
            created_at=args.created_at,
            exporter_revision=args.exporter_revision,
            worker_base_url=args.worker_base_url,
            api_key_env=args.api_key_env,
        )
    except (OSError, ArtifactError) as error:
        raise SystemExit(f"cannot stage development artifact: {error}") from error
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
