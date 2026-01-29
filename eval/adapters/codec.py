"""ARC-agnostic grid codec using constrained tokens."""

from __future__ import annotations

_ALLOWED_CHARS = set("0123456789\n")


class GridCodec:
    def grid_to_text(self, grid: list[list[int]]) -> str:
        return "\n".join("".join(str(cell) for cell in row) for row in grid)

    def serialize_task(self, task: dict, *, test_index: int = 0) -> str:
        parts: list[str] = []
        for pair in task.get("train", []):
            parts.append("<|im_start|>user")
            parts.append(self.grid_to_text(pair["input"]))
            parts.append("<|im_end|>")
            parts.append("<|im_start|>assistant")
            parts.append(self.grid_to_text(pair["output"]))
            parts.append("<|im_end|>")

        test_items = task.get("test", [])
        if not test_items:
            return "\n".join(parts)

        parts.append("<|im_start|>user")
        parts.append(self.grid_to_text(test_items[test_index]["input"]))
        parts.append("<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)

    def deserialize_grid(self, text: str) -> list[list[int]] | None:
        filtered = "".join(ch for ch in text if ch in _ALLOWED_CHARS)
        filtered = filtered.strip("\n")
        if not filtered:
            return None
        lines = [line for line in filtered.splitlines() if line]
        if not lines:
            return None
        return [[int(ch) for ch in line] for line in lines]
