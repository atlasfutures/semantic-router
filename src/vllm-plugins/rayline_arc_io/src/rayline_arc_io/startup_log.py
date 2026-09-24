# SPDX-License-Identifier: Apache-2.0

"""Bounded capture of vLLM's own engine-sizing lines during engine build.

vLLM reports its attention block size, mamba page padding, KV cache size and
maximum concurrency through the `logging` module while the engine is being
built, never on stdout. The deployment pins `VLLM_LOGGING_LEVEL=WARNING`, so
capturing those lines needs both a handler and a temporary level change around
`AsyncLLM.from_engine_args`. Retaining them is what turns those figures from
source-derived into deployment-observed.

Known limit, deliberately not worked around here: vLLM v1 may build the engine
core in a child process whose log records never reach this process's logging
tree. The capture then yields zero lines, which the session API reports as
`captured: false` rather than as an observation of nothing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

STARTUP_LOG_PATTERNS = (
    "Setting attention block size to",
    "Padding mamba page size by",
    "GPU KV cache size",
    "Maximum concurrency for",
)
# Both the vLLM logger and the root logger are attached because vLLM's logging
# configuration may or may not propagate. Duplicates are dropped in the handler
# rather than guessing which of the two will deliver a given record.
STARTUP_LOG_LOGGER_NAMES = ("vllm", "")
MAX_STARTUP_LOG_LINES = 32
MAX_STARTUP_LOG_LINE_CHARS = 512


class StartupLogCapture(logging.Handler):
    """Retain only matching engine-sizing lines, bounded and de-duplicated."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.lines) >= MAX_STARTUP_LOG_LINES:
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - capture must never break startup
            return
        if not any(pattern in message for pattern in STARTUP_LOG_PATTERNS):
            return
        line = message[:MAX_STARTUP_LOG_LINE_CHARS]
        if line not in self.lines:
            self.lines.append(line)


@contextlib.contextmanager
def capture_startup_log() -> Iterator[StartupLogCapture]:
    """Retain engine-sizing lines emitted inside the block, then restore levels."""

    handler = StartupLogCapture()
    restored: list[tuple[logging.Logger, int]] = []
    try:
        for name in STARTUP_LOG_LOGGER_NAMES:
            target = logging.getLogger(name)
            restored.append((target, target.level))
            target.addHandler(handler)
            if not 0 < target.level <= logging.INFO:
                target.setLevel(logging.INFO)
        yield handler
    finally:
        for target, level in restored:
            target.removeHandler(handler)
            target.setLevel(level)
