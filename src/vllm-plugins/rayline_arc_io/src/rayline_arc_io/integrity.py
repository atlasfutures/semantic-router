# SPDX-License-Identifier: Apache-2.0

"""Content digest binding the deployed plugin to its reviewed source."""

import hashlib
from pathlib import Path


def compute_source_digest(package_root: Path) -> str:
    """Digest every package source file by relative path and content."""
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def installed_source_digest() -> str:
    return compute_source_digest(Path(__file__).resolve().parent)
