"""Evaluation orchestration.

Runner is responsible for flow control only. It must not apply any ARC-specific
logic or decision rules. All strategy belongs to the injected components.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def run(
    dataset,
    solver,
    metrics,
    postprocess,
    failure,
    visualize,
    *,
    config,
) -> dict:
    """Run evaluation end-to-end.

    The runner only orchestrates calls between components. It does not perform
    any ARC-specific reasoning or selection beyond wiring outputs to inputs.
    """
    cfg = config or {}
    dataset_info = dataset.info() if hasattr(dataset, "info") else {}

    get_truth: Callable[[Any, Mapping[str, Any]], Any] | None = None
    get_task_id: Callable[[Any, Mapping[str, Any]], Any] | None = None

    if isinstance(cfg, Mapping):
        get_truth = cfg.get("get_truth")
        get_task_id = cfg.get("get_task_id")
        output_dir = cfg.get("output_dir", ".")
    else:
        output_dir = "."

    task_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, task in enumerate(dataset):
        context = {
            "config": cfg,
            "dataset_info": dataset_info,
            "index": index,
        }

        raw_outputs = solver.solve(task, context=context)
        raw_list = _as_list(raw_outputs)

        if postprocess is not None:
            candidates = postprocess.collect(task, raw_list, context=context)
            prediction = postprocess.select(task, candidates, context=context)
        else:
            candidates = raw_list
            prediction = raw_list

        truth = None
        if callable(get_truth):
            truth = get_truth(task, context)

        metrics_record = {}
        if metrics is not None:
            metrics_record = metrics.compute(task, prediction, truth, context=context)

        task_id = index
        if callable(get_task_id):
            task_id = get_task_id(task, context)

        task_records.append(
            {
                "task_id": task_id,
                "metrics": metrics_record,
                "prediction": prediction,
                "candidates": candidates,
            }
        )
        metric_records.append(metrics_record)

        if failure is not None:
            failure_record = failure.record(task, prediction, truth, context=context)
            if failure_record is not None:
                failures.append(failure_record)

    summary = metrics.aggregate(metric_records, config=cfg) if metrics is not None else {}
    failure_summary = (
        failure.summarize(failures, config=cfg) if failure is not None else {}
    )

    reports = {
        "metrics": summary,
        "failures": failure_summary,
        "records": task_records,
    }

    artifacts = []
    if visualize is not None:
        artifacts = visualize.build(reports, output_dir=output_dir)

    return {
        "dataset_info": dataset_info,
        "records": task_records,
        "summary": summary,
        "failures": failures,
        "failure_summary": failure_summary,
        "artifacts": artifacts,
    }
