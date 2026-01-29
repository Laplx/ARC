"""ARC dataset loader (JSON tasks)."""

from __future__ import annotations

import json
import os
from typing import Iterator

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
