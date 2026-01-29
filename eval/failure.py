"""Minimal failure analysis utilities."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def _get_compare(context: Mapping[str, Any] | None) -> Callable[[Any, Any, Any], bool] | None:
    if not context:
        return None
    cfg = context.get("config")
    if isinstance(cfg, Mapping):
        compare = cfg.get("compare")
        if callable(compare):
            return compare
    return None


def record(task, prediction, truth, *, context=None) -> dict | None:
    """Record a failure if a comparator is provided and returns False."""
    compare = _get_compare(context)
    if truth is None or compare is None:
        return None

    try:
        matched = bool(compare(prediction, truth, context))
    except Exception as exc:  # pragma: no cover - leave failures to caller
        return {
            "task_id": context.get("index") if isinstance(context, Mapping) else None,
            "reason": "compare_error",
            "error": repr(exc),
        }

    if matched:
        return None

    return {
        "task_id": context.get("index") if isinstance(context, Mapping) else None,
        "reason": "mismatch",
        "prediction": prediction,
        "truth": truth,
    }


def summarize(failures: list[dict], *, config=None) -> dict:
    """Summarize failures with counts only."""
    return {"failure_count": len(failures)}
