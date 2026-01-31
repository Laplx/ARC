"""Minimal visualization utilities."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def build(reports: dict, *, output_dir: str) -> list[str]:
    """Write report artifacts and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    meta = reports.get("meta") if isinstance(reports, dict) else None
    model_key = meta.get("model_key") if isinstance(meta, dict) else None
    model_id = meta.get("model_id") if isinstance(meta, dict) else None
    split = meta.get("split") if isinstance(meta, dict) else None
    viz_failures = bool(meta.get("viz_failures")) if isinstance(meta, dict) else False

    name_parts = ["report"]
    if model_key:
        name_parts.append(str(model_key))
    elif model_id:
        name_parts.append(str(model_id))
    if split:
        name_parts.append(str(split))

    raw_name = "_".join(name_parts)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
    output_path = os.path.join(output_dir, f"{safe_name}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=True, indent=2, default=str)

    artifacts = [output_path]
    if viz_failures:
        failure_path = os.path.join(output_dir, f"{safe_name}_failures.txt")
        with open(failure_path, "w", encoding="utf-8") as handle:
            handle.write(_build_failure_report(reports))
        artifacts.append(failure_path)

    return artifacts


def _build_failure_report(reports: dict) -> str:
    failures = reports.get("failure_records", []) if isinstance(reports, dict) else []
    metrics = reports.get("metrics", {}) if isinstance(reports, dict) else {}
    meta = reports.get("meta", {}) if isinstance(reports, dict) else {}

    lines = []
    lines.append("FAILURE REPORT")
    lines.append("==============")
    lines.append(f"model_key: {meta.get('model_key')}")
    lines.append(f"model_id: {meta.get('model_id')}")
    lines.append(f"split: {meta.get('split')}")
    lines.append(f"total: {metrics.get('total')} scored: {metrics.get('scored')} mean_score: {metrics.get('mean_score')}")
    lines.append("")

    if not failures:
        lines.append("No failure records available.")
        return "\n".join(lines)

    lines.append("Failure Distribution (by truth grid size):")
    size_counts: dict[str, int] = {}
    color_counts: dict[int, int] = {}
    for failure in failures:
        truth_grid = _extract_grid(failure.get("truth"))
        pred_grid = _extract_grid(failure.get("prediction"))
        grid = truth_grid or pred_grid
        if grid:
            rows = len(grid)
            cols = len(grid[0]) if rows else 0
            key = f"{rows}x{cols}"
            size_counts[key] = size_counts.get(key, 0) + 1
            colors = set()
            for row in grid:
                colors.update(row)
            color_counts[len(colors)] = color_counts.get(len(colors), 0) + 1

    for key in sorted(size_counts.keys()):
        lines.append(f"  size {key}: {size_counts[key]}")

    lines.append("")
    lines.append("Failure Distribution (by color count):")
    for key in sorted(color_counts.keys()):
        lines.append(f"  colors {key}: {color_counts[key]}")

    lines.append("")
    lines.append("Top-K Failures (first 10):")
    for idx, failure in enumerate(failures[:10], start=1):
        lines.append(f"  {idx}. task_id={failure.get('task_id')} reason={failure.get('reason')}")

    lines.append("")
    lines.append("Failure Examples (first 5):")
    for idx, failure in enumerate(failures[:5], start=1):
        truth_grid = _extract_grid(failure.get("truth"))
        pred_grid = _extract_grid(failure.get("prediction"))
        lines.append(f"Example {idx} task_id={failure.get('task_id')}")
        lines.append("truth:")
        lines.append(_grid_to_text(truth_grid) or "<empty>")
        lines.append("prediction:")
        lines.append(_grid_to_text(pred_grid) or "<empty>")
        lines.append("")

    return "\n".join(lines)


def _extract_grid(value: Any) -> Any:
    if isinstance(value, dict) and "grid" in value:
        return value["grid"]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _grid_to_text(grid: Any) -> str:
    if not grid:
        return ""
    return "\n".join("".join(str(cell) for cell in row) for row in grid)
