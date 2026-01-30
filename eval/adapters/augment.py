"""Data augmentation utilities (dihedral + color)."""

from __future__ import annotations

import random
from typing import Iterable


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
