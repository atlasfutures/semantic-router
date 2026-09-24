# SPDX-License-Identifier: Apache-2.0

"""Rayline ARC vLLM IO Processor registration."""


def register_io_processor() -> str:
    """Return the import path vLLM loads after plugin discovery."""

    return "rayline_arc_io.processor.RaylineArcIOProcessor"


__all__ = ["register_io_processor"]
