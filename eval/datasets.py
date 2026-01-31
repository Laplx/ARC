"""ARC dataset loaders."""

from __future__ import annotations

import json
import os
import random
from typing import Iterator

from eval.augment import apply_rules_to_task, build_rules
from eval.interfaces import Dataset, Task


class ARCDataset(Dataset):
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

    def __iter__(self) -> Iterator[Task]:
        for path in self._files:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            task_id = os.path.splitext(os.path.basename(path))[0]
            payload["task_id"] = task_id
            yield payload

    def info(self) -> dict:
        return {
            "root": self._root,
            "split": self._split,
            "count": len(self._files),
        }


class AugmentedARCDataset(Dataset):
    def __init__(
        self,
        *,
        base: Dataset,
        samples_per_task: int = 1,
        seed: int = 0,
        use_dihedral: bool = True,
        use_color: bool = True,
        shuffle_pairs: bool = True,
    ) -> None:
        if samples_per_task < 1:
            raise ValueError("samples_per_task must be >= 1")
        self._base = base
        self._samples_per_task = samples_per_task
        self._seed = seed
        self._use_dihedral = use_dihedral
        self._use_color = use_color
        self._shuffle_pairs = shuffle_pairs

    def __iter__(self) -> Iterator[Task]:
        for task in self._base:
            task_id = task.get("task_id")
            for sample_idx in range(self._samples_per_task):
                if sample_idx == 0:
                    yield _copy_task(task, task_id, sample_idx)
                    continue

                rng = random.Random(_make_seed(task_id, sample_idx, self._seed))
                rules = build_rules(rng, use_dihedral=self._use_dihedral, use_color=self._use_color)
                augmented = apply_rules_to_task(task, rules)
                if self._shuffle_pairs:
                    rng.shuffle(augmented["train"])
                    rng.shuffle(augmented["test"])
                augmented["task_id"] = f"{task_id}_aug{sample_idx}"
                yield augmented

    def info(self) -> dict:
        base_info = self._base.info() if hasattr(self._base, "info") else {}
        return {
            **base_info,
            "augmented": True,
            "samples_per_task": self._samples_per_task,
        }


def _copy_task(task: Task, task_id: str | None, sample_idx: int) -> Task:
    copied = {
        "train": [
            {"input": pair["input"], "output": pair["output"]}
            for pair in task.get("train", [])
        ],
        "test": [
            {"input": item["input"], **({"output": item["output"]} if "output" in item else {})}
            for item in task.get("test", [])
        ],
        "task_id": f"{task_id}_orig{sample_idx}" if task_id else None,
    }
    return copied


def _make_seed(task_id: str | None, sample_idx: int, seed: int) -> int:
    base = hash(task_id) if task_id is not None else 0
    return (base + seed * 131 + sample_idx * 17) & 0xFFFFFFFF
