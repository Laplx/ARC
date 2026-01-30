"""Architect wrapper solver."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from eval.interfaces import Candidate, Prediction, Solver


class ArchitectSolver(Solver):
    def __init__(
        self,
        *,
        base_solver: Solver,
        architect,
        max_steps: int = 128,
        max_candidates: int = 50,
        top_k: int = 5,
        max_nodes: int = 512,
        min_prob: float = 0.05,
        lambda_weight: float = 0.0,
    ) -> None:
        self._base_solver = base_solver
        self._architect = architect
        self._max_steps = max_steps
        self._max_candidates = max_candidates
        self._top_k = top_k
        self._max_nodes = max_nodes
        self._min_prob = min_prob
        self._lambda_weight = lambda_weight

    def solve(self, task, *, context: Mapping[str, object] | None = None) -> Prediction:
        model, tokenizer = _get_backend(self._base_solver)
        codec = _get_codec(self._base_solver)
        allowed_ids = _allowed_token_ids(tokenizer)

        aggregates: dict[tuple[tuple[int, ...], ...], dict] = {}

        for record in self._architect.transform(task):
            transformed = record["task"]
            inverse = record["inverse"]
            meta = record.get("meta", {})
            transform_key = (meta.get("geom"), meta.get("color_perm"))

            prompt = codec.serialize_task(transformed, test_index=0)
            candidates = _dfs_generate(
                model,
                tokenizer,
                prompt,
                allowed_ids=allowed_ids,
                max_steps=self._max_steps,
                max_candidates=self._max_candidates,
                top_k=self._top_k,
                max_nodes=self._max_nodes,
                min_prob=self._min_prob,
            )

            for text, logprob in candidates:
                grid = codec.deserialize_grid(text)
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
                    transform_key,
                    {"logprob_sum": 0.0, "count": 0},
                )
                stats["logprob_sum"] += float(logprob)
                stats["count"] += 1
                entry["transforms"].append(meta)

        candidates_out: list[Candidate] = []
        for entry in aggregates.values():
            transform_means = []
            for stats in entry["transform_stats"].values():
                transform_means.append(stats["logprob_sum"] / stats["count"])
            mean_logprob = sum(transform_means) / len(transform_means) if transform_means else float("-inf")
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


def _get_backend(base_solver):
    model = getattr(base_solver, "_model", None)
    if model is None or not hasattr(model, "get_backend"):
        raise RuntimeError("Base solver does not expose a model backend")
    return model.get_backend()


def _get_codec(base_solver):
    codec = getattr(base_solver, "_codec", None)
    if codec is None:
        raise RuntimeError("Base solver does not expose a codec")
    return codec


def _allowed_token_ids(tokenizer):
    token_strings = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "\n",
        "user",
        "assistant",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
    ]

    ids: list[int] = []
    for token in token_strings:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            ids.append(token_id)
            continue
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return sorted(set(ids))


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
