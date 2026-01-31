"""Minimal failure analysis utilities."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def _extract_grid(value: Any) -> Any:
    if isinstance(value, dict) and "grid" in value:
        return value["grid"]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _default_compare(prediction: Any, truth: Any, context: Mapping[str, Any] | None) -> bool:
    return _extract_grid(prediction) == _extract_grid(truth)


def _get_compare(context: Mapping[str, Any] | None) -> Callable[[Any, Any, Any], bool]:
    if context:
        cfg = context.get("config")
        if isinstance(cfg, Mapping):
            compare = cfg.get("compare")
            if callable(compare):
                return compare
    return _default_compare


def record(task, prediction, truth, *, context=None) -> dict | None:
    """Record a failure if comparison returns False."""
    if truth is None:
        return None

    compare = _get_compare(context)
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
