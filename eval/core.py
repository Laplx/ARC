"""Lightweight evaluation core to keep ARC runs simple."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Mapping
import json
import os
import re

_ALLOWED_CHARS = set("0123456789\n")


class GridCodec:
    def grid_to_text(self, grid: list[list[int]]) -> str:
        return "\n".join("".join(str(cell) for cell in row) for row in grid)

    def serialize_task(self, task: dict, *, test_index: int = 0) -> str:
        segments: list[str] = []
        for pair in task.get("train", []):
            grid_in = self.grid_to_text(pair["input"])
            grid_out = self.grid_to_text(pair["output"])
            segments.append(
                f"<|im_start|>user\n{grid_in}<|im_end|><|im_start|>assistant\n{grid_out}<|im_end|>"
            )

        test_items = task.get("test", [])
        if not test_items:
            return "\n".join(segments)

        grid_in = self.grid_to_text(test_items[test_index]["input"])
        segments.append(f"<|im_start|>user\n{grid_in}<|im_end|><|im_start|>assistant\n")
        return "\n".join(segments)

    def deserialize_grid(self, text: str) -> list[list[int]] | None:
        filtered = "".join(ch for ch in text if ch in _ALLOWED_CHARS).strip("\n")
        if not filtered:
            return None
        lines = filtered.splitlines()
        rows: list[list[int]] = []
        for line in lines:
            row = [int(ch) for ch in line if ch.isdigit()]
            if row:
                rows.append(row)
        if not rows:
            return None
        if len(rows) > 30:
            rows = rows[:30]
        width = len(rows[0])
        if width < 1 or width > 30:
            return None
        if any(len(row) != width for row in rows):
            return None
        if len(rows) < 1 or len(rows) > 30:
            return None
        return rows


class ARCDataset:
    def __init__(self, *, root: str, split: str, max_tasks: int = 0) -> None:
        self._root = root
        self._split = split
        self._max_tasks = max_tasks
        self._files = self._collect_files()

    def _collect_files(self) -> list[str]:
        split_dir = os.path.join(self._root, self._split)
        if not os.path.isdir(split_dir):
            return []
        files = [
            os.path.join(split_dir, name)
            for name in os.listdir(split_dir)
            if name.endswith(".json")
        ]
        files.sort()
        if self._max_tasks and self._max_tasks > 0:
            return files[: self._max_tasks]
        return files

    def __iter__(self) -> Iterable[dict]:
        for path in self._files:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            task_id = os.path.splitext(os.path.basename(path))[0]
            payload["task_id"] = task_id
            yield payload

    def info(self) -> dict:
        return {"root": self._root, "split": self._split, "count": len(self._files)}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _select_candidate(candidates: list) -> Any:
    if not candidates:
        return None
    if all(isinstance(candidate, Mapping) and "score" in candidate for candidate in candidates):
        return max(candidates, key=lambda item: item.get("score", float("-inf")))
    return candidates[0]


def _normalize_prediction(prediction: Any) -> Any:
    if isinstance(prediction, Mapping) and "grid" in prediction:
        return prediction.get("grid")
    if isinstance(prediction, list) and len(prediction) == 1:
        return prediction[0]
    return prediction


def _compare_grid(prediction: Any, truth: Any) -> bool:
    prediction = _normalize_prediction(prediction)
    if prediction is None or truth is None:
        return False
    return prediction == truth


def _grid_to_text(grid: list[list[int]] | None) -> str:
    if grid is None:
        return "<none>"
    rows = ["".join(str(cell) for cell in row) for row in grid]
    return " | ".join(rows)


def _is_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) for row in value):
        return False
    return all(all(isinstance(cell, int) for cell in row) for row in value)


def _compact_grids(value: Any) -> Any:
    if _is_grid(value):
        return _grid_to_text(value)
    if isinstance(value, list):
        return [_compact_grids(item) for item in value]
    if isinstance(value, dict):
        return {key: _compact_grids(item) for key, item in value.items()}
    return value


def _build_failure_report(failures: list[dict]) -> str:
    lines = ["ARC failure report", ""]
    for record in failures:
        lines.append(f"task_id: {record.get('task_id')}")
        lines.append(f"reason: {record.get('reason')}")
        pred = record.get("prediction")
        truth = record.get("truth")
        lines.append("prediction:")
        lines.append(_grid_to_text(_normalize_prediction(pred)))
        lines.append("truth:")
        lines.append(_grid_to_text(truth))
        lines.append("")
    return "\n".join(lines)


def run_evaluation(
    *,
    dataset: Iterable[dict],
    solver,
    get_truth: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    get_task_id: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    output_dir: str = "outputs",
    model_id: str | None = None,
    model_key: str | None = None,
    viz_failures: bool = False,
) -> dict:
    dataset_info = dataset.info() if hasattr(dataset, "info") else {}
    task_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, task in enumerate(dataset):
        context = {"index": index}
        raw_outputs = solver.solve(task, context=context)
        candidates = _as_list(raw_outputs)
        prediction = _select_candidate(candidates)

        truth = get_truth(task, context) if callable(get_truth) else None
        metrics_record: dict[str, Any] = {"has_truth": truth is not None, "scored": False}
        if truth is not None:
            matched = _compare_grid(prediction, truth)
            metrics_record["scored"] = True
            metrics_record["score"] = 1.0 if matched else 0.0
            if not matched:
                failures.append(
                    {
                        "task_id": context["index"],
                        "reason": "mismatch",
                        "prediction": prediction,
                        "truth": truth,
                    }
                )

        task_id = (
            get_task_id(task, context) if callable(get_task_id) else task.get("task_id", index)
        )
        
        if metrics_record.get("scored"):
            status = "correct" if metrics_record.get("score") == 1.0 else "wrong"
        else:
            status = "unscored"
        print(f"[eval] task={task_id} status={status}")
        
        task_records.append(
            {
                "task_id": task_id,
                "metrics": metrics_record,
                "prediction": prediction,
                "candidates": candidates,
            }
        )
        metric_records.append(metrics_record)

    scored = [record.get("score", 0.0) for record in metric_records if record.get("scored")]
    summary = {
        "accuracy": (sum(scored) / len(scored)) if scored else 0.0,
        "scored": len(scored),
        "total": len(metric_records),
    }

    failure_summary = dict(Counter(record.get("reason", "unknown") for record in failures))

    reports = {
        "metrics": summary,
        "failures": failure_summary,
        "failure_records": failures,
        "records": task_records,
        "meta": {
            "model_id": model_id,
            "model_key": model_key,
            "dataset_root": dataset_info.get("root"),
            "split": dataset_info.get("split"),
            "viz_failures": bool(viz_failures),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    name_parts = ["report"]
    if model_key:
        name_parts.append(str(model_key))
    elif model_id:
        name_parts.append(str(model_id))
    if dataset_info.get("split"):
        name_parts.append(str(dataset_info.get("split")))

    raw_name = "_".join(name_parts)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
    output_path = os.path.join(output_dir, f"{safe_name}.json")
    compact_reports = _compact_grids(reports)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(compact_reports, handle, ensure_ascii=True, indent=2, default=str)

    artifacts = [output_path]
    if viz_failures:
        failure_path = os.path.join(output_dir, f"{safe_name}_failures.txt")
        with open(failure_path, "w", encoding="utf-8") as handle:
            handle.write(_build_failure_report(failures))
        artifacts.append(failure_path)

    return {
        "dataset_info": dataset_info,
        "records": task_records,
        "summary": summary,
        "failures": failures,
        "failure_summary": failure_summary,
        "artifacts": artifacts,
    }
