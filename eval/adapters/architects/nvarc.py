"""NVARC-style architect with structural transforms (ARC-agnostic)."""

from __future__ import annotations

from eval.interfaces import Architect, Candidate, Prediction, Task


def _rotate90(grid):
    return [list(row) for row in zip(*grid[::-1])]


def _rotate180(grid):
    return [list(row[::-1]) for row in grid[::-1]]


def _rotate270(grid):
    return [list(row) for row in zip(*grid)][::-1]


def _flip_h(grid):
    return [list(row[::-1]) for row in grid]


def _flip_v(grid):
    return [list(row) for row in grid[::-1]]


def _transpose(grid):
    return [list(row) for row in zip(*grid)]


def _anti_transpose(grid):
    return _rotate180(_transpose(grid))


_TRANSFORMS = {
    "identity": lambda g: g,
    "rot90": _rotate90,
    "rot180": _rotate180,
    "rot270": _rotate270,
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "transpose": _transpose,
    "anti_transpose": _anti_transpose,
}


def _apply_to_pair(pair, transform):
    return {
        "input": transform(pair["input"]),
        "output": transform(pair["output"]),
    }


def _apply_to_test(item, transform):
    transformed = {"input": transform(item["input"])}
    if "output" in item:
        transformed["output"] = transform(item["output"])
    return transformed


class NVARCArchitect(Architect):
    def transform(self, task: Task) -> list[Task]:
        tasks: list[Task] = []
        for name, fn in _TRANSFORMS.items():
            transformed = {
                "train": [_apply_to_pair(pair, fn) for pair in task.get("train", [])],
                "test": [_apply_to_test(item, fn) for item in task.get("test", [])],
                "task_id": task.get("task_id"),
                "transform": name,
            }
            tasks.append(transformed)
        return tasks

    def score(self, task: Task, candidate: Candidate, *, context=None) -> float:
        if isinstance(candidate, dict) and "score" in candidate:
            try:
                return float(candidate["score"])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def select(self, task: Task, candidates: list[Candidate], *, context=None) -> Prediction:
        if not candidates:
            return None
        best = max(candidates, key=lambda cand: self.score(task, cand, context=context))
        return best
