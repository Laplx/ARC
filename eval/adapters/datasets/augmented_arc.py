"""Augmented ARC dataset wrapper."""

from __future__ import annotations

import random
from typing import Iterator

from eval.adapters.augment import apply_rules_to_task, build_rules
from eval.interfaces import Dataset, Task


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
