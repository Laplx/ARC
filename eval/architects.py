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
    "identity": (lambda g: g, lambda g: g),
    "rot90": (_rotate90, _rotate270),
    "rot180": (_rotate180, _rotate180),
    "rot270": (_rotate270, _rotate90),
    "flip_h": (_flip_h, _flip_h),
    "flip_v": (_flip_v, _flip_v),
    "transpose": (_transpose, _transpose),
    "anti_transpose": (_anti_transpose, _anti_transpose),
}


def _apply_color(grid, mapping):
    if not mapping:
        return grid
    return [[mapping.get(cell, cell) for cell in row] for row in grid]


def _apply_to_pair(pair, geom_fn, color_map):
    return {
        "input": _apply_color(geom_fn(pair["input"]), color_map),
        "output": _apply_color(geom_fn(pair["output"]), color_map),
    }


def _apply_to_test(item, geom_fn, color_map):
    transformed = {"input": _apply_color(geom_fn(item["input"]), color_map)}
    if "output" in item:
        transformed["output"] = _apply_color(geom_fn(item["output"]), color_map)
    return transformed


def _collect_colors(task: Task) -> list[int]:
    colors: set[int] = set()
    for pair in task.get("train", []):
        for grid in (pair.get("input"), pair.get("output")):
            for row in grid:
                colors.update(row)
    for item in task.get("test", []):
        for grid in (item.get("input"), item.get("output")):
            if grid is None:
                continue
            for row in grid:
                colors.update(row)
    return sorted(colors)


def _color_permutations(colors: list[int], limit: int = 4) -> list[tuple[str, dict[int, int]]]:
    perms: list[tuple[str, dict[int, int]]] = [("identity", {})]
    if len(colors) < 2:
        return perms
    base = colors[:4]
    if len(base) >= 2:
        a, b = base[0], base[1]
        perms.append((f"swap_{a}_{b}", {a: b, b: a}))
    if len(base) >= 3 and len(perms) < limit:
        a, b, c = base[0], base[1], base[2]
        perms.append((f"cycle_{a}_{b}_{c}", {a: b, b: c, c: a}))
    if len(base) >= 4 and len(perms) < limit:
        a, b, c, d = base[0], base[1], base[2], base[3]
        perms.append((f"swap_{c}_{d}", {c: d, d: c}))
    return perms[:limit]


def _invert_map(mapping: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in mapping.items()}


def _canonical_key(candidate: Candidate):
    grid = candidate.get("grid") if isinstance(candidate, dict) else None
    if grid is None:
        return ""
    return ";".join("".join(str(cell) for cell in row) for row in grid)


class NVARCArchitect(Architect):
    def transform(self, task: Task) -> list[dict]:
        colors = _collect_colors(task)
        color_perms = _color_permutations(colors)
        transforms: list[dict] = []
        for geom_name, (geom_fn, inv_geom) in _TRANSFORMS.items():
            for perm_name, perm_map in color_perms:
                inv_map = _invert_map(perm_map)

                def _inverse(grid, g=inv_geom, m=inv_map):
                    return _apply_color(g(grid), m)

                transformed_task = {
                    "train": [
                        _apply_to_pair(pair, geom_fn, perm_map)
                        for pair in task.get("train", [])
                    ],
                    "test": [
                        _apply_to_test(item, geom_fn, perm_map)
                        for item in task.get("test", [])
                    ],
                    "task_id": task.get("task_id"),
                }
                transforms.append(
                    {
                        "task": transformed_task,
                        "inverse": _inverse,
                        "meta": {"geom": geom_name, "color_perm": perm_name},
                    }
                )
        return transforms

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
        ranked = sorted(
            candidates,
            key=lambda cand: (-self.score(task, cand, context=context), _canonical_key(cand)),
        )
        best = ranked[0]
        return best.get("grid") if isinstance(best, dict) else best
