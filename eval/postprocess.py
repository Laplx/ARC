"""Minimal postprocess utilities (no strategy)."""

from __future__ import annotations

from typing import Any


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _has_score(candidate: Any) -> bool:
    return isinstance(candidate, dict) and "score" in candidate


def collect(task, raw_outputs: list, *, context=None) -> list:
    """Collect raw outputs without applying any selection logic."""
    return _as_list(raw_outputs)


def select(task, candidates: list, *, context=None):
    """Select without heuristics; prefer argmax(score) when present, else top-1."""
    if not candidates:
        return None

    if all(_has_score(candidate) for candidate in candidates):
        return max(candidates, key=lambda item: item.get("score", float("-inf")))

    return candidates[0]
