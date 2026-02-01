"""NVARC-style architect DFS utilities for notebooks."""

from __future__ import annotations

import math
import os
from typing import Mapping

import torch

from eval.core import GridCodec
from eval.models import HuggingFaceTextGenerator

MODEL_REGISTRY = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    # "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "deepseek-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def resolve_model_id(model_key: str, model_path: str | None = None) -> tuple[str, bool]:
    if model_path:
        path = os.path.expanduser(model_path)
        if os.path.isfile(path):
            return os.path.dirname(path), True
        return path, True
    return MODEL_REGISTRY.get(model_key, model_key), False


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


def _collect_colors(task) -> list[int]:
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


def _canonical_key(candidate: dict) -> str:
    grid = candidate.get("grid") if isinstance(candidate, dict) else None
    if grid is None:
        return ""
    return ";".join("".join(str(cell) for cell in row) for row in grid)


class NVARCArchitect:
    def transform(self, task: dict) -> list[dict]:
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

    def score(self, task: dict, candidate: dict, *, context=None) -> float:
        if isinstance(candidate, dict) and "score" in candidate:
            try:
                return float(candidate["score"])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def select(self, task: dict, candidates: list[dict], *, context=None):
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda cand: (-self.score(task, cand, context=context), _canonical_key(cand)),
        )
        best = ranked[0]
        return best.get("grid") if isinstance(best, dict) else best


def _allowed_token_ids(tokenizer) -> list[int]:
    allowed: set[int] = set()
    for ch in "0123456789\n":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            allowed.add(ids[0])
    return sorted(allowed)


def _dfs_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    allowed_ids: list[int],
    max_steps: int,
    max_candidates: int,
    top_k: int,
    max_nodes: int,
    min_prob: float,
) -> list[tuple[str, float]]:
    if not allowed_ids:
        return []

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    start_ids = encoded["input_ids"]
    stack: list[tuple[torch.Tensor, float, list[int]]] = [(start_ids, 0.0, [])]
    results: list[tuple[str, float]] = []
    nodes = 0

    log_minprob = float("-inf")
    if min_prob is not None and min_prob > 0.0:
        log_minprob = math.log(min_prob)

    end_token_ids = {
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }

    with torch.no_grad():
        while stack and len(results) < max_candidates and nodes < max_nodes:
            input_ids, score, generated = stack.pop()
            nodes += 1
            logits = model(input_ids=input_ids).logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            options = [(tid, log_probs[tid].item()) for tid in allowed_ids]
            options.sort(key=lambda item: item[1], reverse=True)
            for tid, lp in options[:top_k]:
                new_generated = generated + [tid]
                new_score = score + lp
                if new_score < log_minprob:
                    continue
                text = tokenizer.decode(new_generated, skip_special_tokens=False)
                if tid in end_token_ids or len(new_generated) >= max_steps:
                    results.append((text, new_score))
                    continue
                next_ids = torch.tensor([[tid]], device=input_ids.device)
                stack.append((torch.cat([input_ids, next_ids], dim=1), new_score, new_generated))

    return results


class ArchitectSolver:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        codec: GridCodec,
        architect: NVARCArchitect,
        max_steps: int = 128,
        max_candidates: int = 50,
        top_k: int = 5,
        max_nodes: int = 512,
        min_prob: float = 0.05,
        lambda_weight: float = 0.0,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._codec = codec
        self._architect = architect
        self._max_steps = max_steps
        self._max_candidates = max_candidates
        self._top_k = top_k
        self._max_nodes = max_nodes
        self._min_prob = min_prob
        self._lambda_weight = lambda_weight
        self._allowed_ids = _allowed_token_ids(tokenizer)

    def solve(self, task, *, context: Mapping[str, object] | None = None):
        aggregates: dict[tuple[tuple[int, ...], ...], dict] = {}

        for record in self._architect.transform(task):
            transformed = record["task"]
            inverse = record["inverse"]
            meta = record.get("meta", {})
            transform_key = (meta.get("geom"), meta.get("color_perm"))

            prompt = self._codec.serialize_task(transformed, test_index=0)
            candidates = _dfs_generate(
                self._model,
                self._tokenizer,
                prompt,
                allowed_ids=self._allowed_ids,
                max_steps=self._max_steps,
                max_candidates=self._max_candidates,
                top_k=self._top_k,
                max_nodes=self._max_nodes,
                min_prob=self._min_prob,
            )

            for text, logprob in candidates:
                grid = self._codec.deserialize_grid(text)
                if grid is None:
                    continue
                grid = inverse(grid)
                key = tuple(tuple(row) for row in grid)
                entry = aggregates.setdefault(
                    key,
                    {
                        "grid": grid,
                        "count": 0,
                        "transform_stats": {},
                        "transforms": [],
                    },
                )
                entry["count"] += 1
                stats = entry["transform_stats"].setdefault(
                    transform_key, {"logprob_sum": 0.0, "count": 0}
                )
                stats["logprob_sum"] += float(logprob)
                stats["count"] += 1
                entry["transforms"].append(meta)

        candidates_out: list[dict] = []
        for entry in aggregates.values():
            transform_means = []
            for stats in entry["transform_stats"].values():
                transform_means.append(stats["logprob_sum"] / stats["count"])
            mean_logprob = (
                sum(transform_means) / len(transform_means) if transform_means else float("-inf")
            )
            score = mean_logprob + self._lambda_weight * entry["count"]
            candidates_out.append(
                {
                    "grid": entry["grid"],
                    "score": score,
                    "meta": {
                        "count": entry["count"],
                        "mean_logprob": mean_logprob,
                        "transforms": entry["transforms"],
                    },
                }
            )

        return candidates_out


def build_architect_solver(
    *,
    model_key: str,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
    max_new_tokens: int = 512,
    max_steps: int = 128,
    max_candidates: int = 50,
    top_k: int = 5,
    max_nodes: int = 512,
    min_prob: float = 0.05,
    lambda_weight: float = 0.0,
):
    model_id, from_local = resolve_model_id(model_key, model_path)
    generator = HuggingFaceTextGenerator(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        tokenizer_id=tokenizer_path,
        local_files_only=from_local,
    )
    model, tokenizer = generator.get_backend()
    return ArchitectSolver(
        model=model,
        tokenizer=tokenizer,
        codec=GridCodec(),
        architect=NVARCArchitect(),
        max_steps=max_steps,
        max_candidates=max_candidates,
        top_k=top_k,
        max_nodes=max_nodes,
        min_prob=min_prob,
        lambda_weight=lambda_weight,
    )

