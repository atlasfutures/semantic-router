# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from rayline_arc_io.startup_log import (
    MAX_STARTUP_LOG_LINE_CHARS,
    MAX_STARTUP_LOG_LINES,
    STARTUP_LOG_PATTERNS,
    capture_startup_log,
)

OBSERVED_ENGINE_LINES = (
    "Setting attention block size to 544 tokens",
    "Padding mamba page size by 1.02x to 8704 bytes",
    "GPU KV cache size: 4,455,808 tokens",
    "Maximum concurrency for 262,144 tokens per request: 17.00x",
)


def test_capture_retains_only_the_engine_sizing_lines() -> None:
    logger = logging.getLogger("vllm.test.capture")

    with capture_startup_log() as capture:
        for line in OBSERVED_ENGINE_LINES:
            logger.info(line)
        logger.info("Adding requests: 100%%|##########| 1/1")
        logger.warning("some unrelated warning")

    assert capture.lines == list(OBSERVED_ENGINE_LINES)


def test_capture_survives_a_vllm_logger_pinned_above_info() -> None:
    # The deployment image pins VLLM_LOGGING_LEVEL=WARNING, so the capture has
    # to lower the level itself or it would observe nothing at all.
    logger = logging.getLogger("vllm")
    logger.setLevel(logging.WARNING)

    try:
        with capture_startup_log() as capture:
            logger.info(OBSERVED_ENGINE_LINES[2])
        assert capture.lines == [OBSERVED_ENGINE_LINES[2]]
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(logging.NOTSET)


def test_capture_detaches_and_restores_levels_after_the_block() -> None:
    logger = logging.getLogger("vllm")
    before_handlers = list(logger.handlers)
    before_level = logger.level

    with capture_startup_log() as capture:
        pass
    logger.info(OBSERVED_ENGINE_LINES[0])

    assert capture.lines == []
    assert logger.handlers == before_handlers
    assert logger.level == before_level


def test_capture_deduplicates_and_truncates_long_lines() -> None:
    logger = logging.getLogger("vllm.test.dedup")

    with capture_startup_log() as capture:
        for _ in range(5):
            logger.info("GPU KV cache size: %s tokens", "4,455,808")
        logger.info("Maximum concurrency for " + "x" * 4096)

    assert capture.lines == [
        "GPU KV cache size: 4,455,808 tokens",
        ("Maximum concurrency for " + "x" * 4096)[:MAX_STARTUP_LOG_LINE_CHARS],
    ]


def test_capture_is_bounded_and_drops_an_unrenderable_record() -> None:
    logger = logging.getLogger("vllm.test.bounded")
    unrenderable = logging.LogRecord(
        name="vllm",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="GPU KV cache size: %d tokens",
        args=("not-a-number",),
        exc_info=None,
    )

    with capture_startup_log() as capture:
        capture.emit(unrenderable)
        for index in range(MAX_STARTUP_LOG_LINES + 10):
            logger.info("GPU KV cache size: %s tokens", index)

    # A record that cannot render is dropped rather than failing startup, and
    # the retained set stays bounded however much the engine logs.
    assert len(capture.lines) == MAX_STARTUP_LOG_LINES
    assert capture.lines[0] == "GPU KV cache size: 0 tokens"


def test_patterns_cover_every_derived_engine_sizing_figure() -> None:
    assert STARTUP_LOG_PATTERNS == (
        "Setting attention block size to",
        "Padding mamba page size by",
        "GPU KV cache size",
        "Maximum concurrency for",
    )
