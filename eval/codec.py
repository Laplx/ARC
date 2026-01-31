"""ARC-agnostic grid codec using constrained tokens."""

from __future__ import annotations


_ALLOWED_CHARS = set("0123456789\n")


class GridCodec:
    def grid_to_text(self, grid: list[list[int]]) -> str:
        return "\n".join("".join(str(cell) for cell in row) for row in grid)

    def serialize_task(self, task: dict, *, test_index: int = 0) -> str:
        # NVARC-style formatting: no newline before <|im_end|>
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
