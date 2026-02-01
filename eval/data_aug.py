"""ARChitects training data helpers (augmentation + SFT-ready datasets)."""

from __future__ import annotations

from typing import Iterable
import random

from eval.core import ARCDataset


def dihedral_transform(grid: list[list[int]], tid: int) -> list[list[int]]:
    if tid == 0:
        return grid
    if tid == 1:
        return [list(row) for row in zip(*grid[::-1])]
    if tid == 2:
        return [list(row[::-1]) for row in grid[::-1]]
    if tid == 3:
        return [list(row) for row in zip(*grid)][::-1]
    if tid == 4:
        return [list(row[::-1]) for row in grid]
    if tid == 5:
        return [list(row) for row in grid[::-1]]
    if tid == 6:
        return [list(row) for row in zip(*grid)]
    if tid == 7:
        return [list(row[::-1]) for row in zip(*grid[::-1])]
    raise ValueError(f"Invalid dihedral id: {tid}")


def color_mapping(grid: list[list[int]], mapping: list[int]) -> list[list[int]]:
    return [[mapping[cell] for cell in row] for row in grid]


def build_rules(rng: random.Random, *, use_dihedral: bool = True, use_color: bool = True) -> list[dict]:
    rules: list[dict] = []
    if use_dihedral:
        rules.append({"type": "dihedral", "settings": {"tid": rng.randint(0, 7)}})
    if use_color:
        colors = list(range(10))
        rng.shuffle(colors)
        rules.append({"type": "color", "settings": {"mapping": colors}})
    return rules


def apply_rules_to_grid(grid: list[list[int]], rules: Iterable[dict]) -> list[list[int]]:
    out = grid
    for rule in rules:
        settings = rule["settings"]
        if rule["type"] == "dihedral":
            out = dihedral_transform(out, settings["tid"])
        elif rule["type"] == "color":
            out = color_mapping(out, settings["mapping"])
        else:
            raise ValueError(f"Unsupported augmentation: {rule['type']}")
    return out


def apply_rules_to_task(task: dict, rules: Iterable[dict]) -> dict:
    transformed = {
        "train": [],
        "test": [],
        "task_id": task.get("task_id"),
    }
    for pair in task.get("train", []):
        transformed["train"].append(
            {
                "input": apply_rules_to_grid(pair["input"], rules),
                "output": apply_rules_to_grid(pair["output"], rules),
            }
        )
    for item in task.get("test", []):
        test_item = {"input": apply_rules_to_grid(item["input"], rules)}
        if "output" in item:
            test_item["output"] = apply_rules_to_grid(item["output"], rules)
        transformed["test"].append(test_item)
    return transformed


def _copy_task(task: dict, task_id: str | None, sample_idx: int) -> dict:
    copied = {
        "train": [
            {"input": pair["input"], "output": pair["output"]}
            for pair in task.get("train", [])
        ],
        "test": [
            {
                "input": item["input"],
                **({"output": item["output"]} if "output" in item else {}),
            }
            for item in task.get("test", [])
        ],
        "task_id": f"{task_id}_orig{sample_idx}" if task_id else None,
    }
    return copied


def _make_seed(task_id: str | None, sample_idx: int, seed: int) -> int:
    base = hash(task_id) if task_id is not None else 0
    return (base + seed * 131 + sample_idx * 17) & 0xFFFFFFFF


class AugmentedARCDataset:
    def __init__(
        self,
        *,
        base: ARCDataset,
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

    def __iter__(self):
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
        return {**base_info, "augmented": True, "samples_per_task": self._samples_per_task}
