"""Lightweight evaluation core to keep ARC runs simple."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Mapping
import json
import os
import re


from pathlib import Path
LOG_PATH = Path("outputs/architect_beam.log")

def _log_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _ORIG_PRINT(*args, **kwargs)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

import builtins

# 防止重复包裹：仅首次保存原 print
if not hasattr(builtins, "_orig_print_archbeam"):
    builtins._orig_print_archbeam = builtins.__dict__["print"]

_ORIG_PRINT = builtins._orig_print_archbeam
builtins.print = _log_print


_ALLOWED_CHARS = set("0123456789Ċ")


class GridCodec:
    def grid_to_text(self, grid: list[list[int]]) -> str:
        return "Ċ".join("".join(str(cell) for cell in row) for row in grid)

    def serialize_task(self, task: dict, *, test_index: int = 0) -> str:
        segments: list[str] = []
        for pair in task.get("train", []):
            grid_in = self.grid_to_text(pair["input"])
            grid_out = self.grid_to_text(pair["output"])
            segments.append(
                f"<|im_start|>userĊ{grid_in}<|im_end|><|im_start|>assistantĊ{grid_out}<|im_end|>"
            )

        test_items = task.get("test", [])
        if not test_items:
            base = "Ċ".join(segments)
            return base + base  # repeat prompt
        base = None

        grid_in = self.grid_to_text(test_items[test_index]["input"])
        segments.append(f"<|im_start|>userĊ{grid_in}<|im_end|><|im_start|>assistantĊ")
        base = "Ċ".join(segments)
        return base + base  # repeat prompt

    def deserialize_grid(self, text: str) -> list[list[int]] | None:
        # keep only allowed chars (digits + our newline marker) and strip leading/trailing separators
        # unify newline markers and keep only allowed chars
        text = text.replace("\n", "Ċ")
        text = text.replace("<|im_end|>", "Ċ")
        filtered = "".join(ch for ch in text if ch in _ALLOWED_CHARS).strip("Ċ")
        print(f"Deserializing grid from text:\n{filtered}\n")
        if not filtered:
            return None

        lines = filtered.split("Ċ")
        rows: list[list[int]] = []

        # Extract only the first contiguous block of consistent-width rows.
        expected_width: int | None = None
        for line in lines:
            # stop at blank separators that may appear inside model output
            if not line.strip():
                break
            row = [int(ch) for ch in line if ch.isdigit()]
            if not row:
                continue
            if expected_width is None:
                expected_width = len(row)
                if expected_width < 1 or expected_width > 30:
                    return None
            # if width changes, we treat it as the start of garbage and stop collecting
            if len(row) != expected_width:
                break
            rows.append(row)
            # limit rows to 30 to match ARC constraints
            if len(rows) >= 30:
                break

        if not rows:
            return None
        if len(rows) > 30:
            rows = rows[:30]
        return rows
    

class ARCDataset:
    def __init__(self, *, root: str, split: str, max_tasks: int = 0, start_from: int = 0) -> None:
        self._root = root
        self._split = split
        self._max_tasks = max_tasks
        self._start_from = start_from
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
        if self._start_from and self._start_from > 0:
            return files[self._start_from :]
        return files

    def __iter__(self) -> Iterable[dict]:
        for path in self._files:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            task_id = os.path.splitext(os.path.basename(path))[0]
            payload["task_id"] = task_id
            tests = payload.get("test") or []
            if len(tests) > 1:
                train = payload.get("train") or []
                payload["train"] = train + tests[1:]
                payload["test"] = tests[:1]
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
    return "Ċ".join(lines)


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
    max_tasks: int = 0,
    cleanup_every: int = 1,
) -> dict:
    dataset_info = dataset.info() if hasattr(dataset, "info") else {}
    task_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, task in enumerate(dataset):
        if max_tasks and max_tasks > 0 and index >= max_tasks:
            break
        context = {"index": index}
        raw_outputs = solver.solve(task, context=context)
        print(f"raw_outputs: {raw_outputs}")
        candidates = _as_list(raw_outputs)
        # 排序（若有score字段则按score降序），以便 top-k 评估
        if all(isinstance(c, Mapping) and "score" in c for c in candidates):
            candidates_sorted = sorted(
                candidates, key=lambda c: c.get("score", float("-inf")), reverse=True
            )
        else:
            candidates_sorted = candidates
        prediction = _select_candidate(candidates_sorted)

        truth = get_truth(task, context) if callable(get_truth) else None
        metrics_record: dict[str, Any] = {"has_truth": truth is not None, "scored": False}
        if truth is not None:
            # top1 命中
            hit_top1 = _compare_grid(_normalize_prediction(prediction), truth)
            # any 命中
            hit_any = any(
                _compare_grid(_normalize_prediction(cand), truth) for cand in candidates_sorted
            )
            # pass@k for k in {2,4,8}
            ks = [2, 4, 8]
            hit_topk = {}
            for k in ks:
                topk = candidates_sorted[:k] if k > 0 else candidates_sorted
                hit_topk[k] = any(
                    _compare_grid(_normalize_prediction(cand), truth) for cand in topk
                )

            metrics_record["scored"] = True
            metrics_record["hit_top1"] = bool(hit_top1)
            metrics_record["hit_any"] = bool(hit_any)
            for k, v in hit_topk.items():
                metrics_record[f"hit_top{k}"] = bool(v)
            if not hit_any:
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
            status = "correct" if metrics_record.get("hit_any") else "wrong"
        else:
            status = "unscored"
        print(f"[eval] task={task_id} status={status}")
        if metrics_record.get("scored"):
            print(
                f"[eval] task={task_id} hit_top1={metrics_record.get('hit_top1')} "
                f"hit_any={metrics_record.get('hit_any')} "
                f"hit_top2={metrics_record.get('hit_top2')} "
                f"hit_top4={metrics_record.get('hit_top4')} "
                f"hit_top8={metrics_record.get('hit_top8')}"
            )
        if status == "wrong":
            print(f"[eval] task={task_id} prediction={_grid_to_text(_normalize_prediction(prediction))}")
            print(f"[eval] task={task_id} truth={_grid_to_text(truth)}")

        # if cleanup_every and (index + 1) % cleanup_every == 0:
        #     try:
        #         import gc
        #         import torch

        #         gc.collect()
        #         if torch.cuda.is_available():
        #             torch.cuda.empty_cache()
        #     except Exception:
        #         pass
        
        task_records.append(
            {
                "task_id": task_id,
                "metrics": metrics_record,
                "prediction": prediction,
                "candidates": candidates,
            }
        )
        metric_records.append(metrics_record)

    scored_any = [1.0 if record.get("hit_any") else 0.0 for record in metric_records if record.get("scored")]
    scored_top1 = [1.0 if record.get("hit_top1") else 0.0 for record in metric_records if record.get("scored")]
    scored_top2 = [1.0 if record.get("hit_top2") else 0.0 for record in metric_records if record.get("scored")]
    scored_top4 = [1.0 if record.get("hit_top4") else 0.0 for record in metric_records if record.get("scored")]
    scored_top8 = [1.0 if record.get("hit_top8") else 0.0 for record in metric_records if record.get("scored")]
    summary = {
        "accuracy_top1": (sum(scored_top1) / len(scored_top1)) if scored_top1 else 0.0,
        "accuracy_any": (sum(scored_any) / len(scored_any)) if scored_any else 0.0,
        "accuracy_top2": (sum(scored_top2) / len(scored_top2)) if scored_top2 else 0.0,
        "accuracy_top4": (sum(scored_top4) / len(scored_top4)) if scored_top4 else 0.0,
        "accuracy_top8": (sum(scored_top8) / len(scored_top8)) if scored_top8 else 0.0,
        "scored": len(scored_any),
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
