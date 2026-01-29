"""Minimal, ARC-agnostic metrics utilities."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def _extract_grid(value: Any) -> Any:
    if isinstance(value, dict) and "grid" in value:
        return value["grid"]
    return value


def _normalize_prediction(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _default_compare(prediction: Any, truth: Any, context: Mapping[str, Any] | None) -> bool:
    pred_norm = _extract_grid(_normalize_prediction(prediction))
    truth_norm = _extract_grid(truth)
    return pred_norm == truth_norm


def _get_compare(context: Mapping[str, Any] | None) -> Callable[[Any, Any, Any], bool]:
    if context:
        cfg = context.get("config")
        if isinstance(cfg, Mapping):
            compare = cfg.get("compare")
            if callable(compare):
                return compare
    return _default_compare


def compute(task, prediction, truth, *, context=None) -> dict:
    """Compute per-task metrics with a pure grid comparator."""
    result: dict[str, Any] = {
        "has_truth": truth is not None,
        "scored": False,
    }

    if truth is None:
        return result

    compare = _get_compare(context)
    try:
        matched = bool(compare(prediction, truth, context))
    except Exception as exc:  # pragma: no cover - leave failures to caller
        result["error"] = repr(exc)
        return result

    result["scored"] = True
    result["score"] = 1.0 if matched else 0.0
    return result


def aggregate(records: list[dict], *, config=None) -> dict:
    """Aggregate per-task metrics without ARC-specific assumptions."""
    total = len(records)
    scored = 0
    score_sum = 0.0
    for record in records:
        if record.get("scored") and isinstance(record.get("score"), (int, float)):
            scored += 1
            score_sum += float(record["score"])

    summary = {
        "total": total,
        "scored": scored,
    }
    if scored:
        summary["mean_score"] = score_sum / scored
    return summary
