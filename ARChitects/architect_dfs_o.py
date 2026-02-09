from __future__ import annotations

import math
import os
from typing import Mapping

import torch

from eval.core_C import GridCodec
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
    """
    与原版不同：我们仍保留变换集，但只用来评分，不再在生成阶段复制任务。
    transform() 现在同时返回正变换（forward）和逆变换（inverse）便于双向应用。
    """

    def transform(self, task: dict) -> list[dict]:
        colors = _collect_colors(task)
        color_perms = _color_permutations(colors)
        transforms: list[dict] = []
        for geom_name, (geom_fn, inv_geom) in _TRANSFORMS.items():
            for perm_name, perm_map in color_perms:
                inv_map = _invert_map(perm_map)

                def _inverse(grid, g=inv_geom, m=inv_map):
                    return _apply_color(g(grid), m)

                def _forward(grid, g=geom_fn, m=perm_map):
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
                        "forward": _forward,
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
    for ch in "0123456789Ċ":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        allowed.update(ids)
        
    end_token_ids = {
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }
    allowed.update(end_token_ids)
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
    device = model.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    print(
        "[architect_dfs_o] DFS start",
        f"device={device}",
        f"allowed={len(allowed_ids)}",
        f"max_steps={max_steps}",
        f"top_k={top_k}",
        f"max_nodes={max_nodes}",
    )

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
            if nodes % 100 == 0:
                print(
                    f"[architect_dfs_o] DFS progress: nodes={nodes}, results={len(results)}, stack={len(stack)}"
                )

    return results


def _logprob_given(
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    *,
    allowed_ids: set[int] | None = None,
) -> float:
    """
    计算在给定 prompt 下逐 token teacher-forcing 生成 target_text 的对数概率总和。
    若出现不在 allowed_ids 的 token，则返回 -inf。
    """

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if allowed_ids is not None:
        for tid in target_ids:
            if tid not in allowed_ids:
                return float("-inf")

    logprob = 0.0
    with torch.no_grad():
        for step, tid in enumerate(target_ids, 1):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_logits = outputs.logits[:, -1, :]
            next_logprobs = torch.log_softmax(next_logits, dim=-1)
            logprob += next_logprobs[0, tid].item()
            next_token = torch.tensor([[tid]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, device=input_ids.device)], dim=1
                )
            if step <= 3 or step == len(target_ids):
                # 打印前几步与最后一步，便于观察概率
                print(
                    f"[architect_dfs_o] logprob step {step}/{len(target_ids)} "
                    f"tid={tid} lp={next_logprobs[0, tid].item():.3f}"
                )
    return logprob


class ArchitectSolver:
    """
    变体版本：
    - 生成阶段只在原任务上 DFS 一次。
    - 评分阶段对每个候选解遍历所有几何+颜色变换，计算平均对数生成概率。
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        codec: GridCodec,
        architect: NVARCArchitect,
        max_steps: int = 931,
        max_candidates: int = 16,
        top_k: int = 5,
        max_nodes: int = 1024,
        min_prob: float = 0,
        lambda_weight: float = 0.0,  # 保留参数但评分中不再使用 count
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
        print("[architect_dfs_o] start solve")
        # 1) 预生成：仅对原任务做一次 DFS，得到候选网格。
        prompt_orig = self._codec.serialize_task(task, test_index=0)
        print("[architect_dfs_o] prompt sample (orig):")
        print(prompt_orig[:300] + ("..." if len(prompt_orig) > 300 else ""))
        print("[architect_dfs_o] running DFS on original task")
        raw_candidates = _dfs_generate(
            self._model,
            self._tokenizer,
            prompt_orig,
            allowed_ids=self._allowed_ids,
            max_steps=self._max_steps,
            max_candidates=self._max_candidates,
            top_k=self._top_k,
            max_nodes=self._max_nodes,
            min_prob=self._min_prob,
        )
        print(f"[architect_dfs_o] DFS done, raw candidates: {len(raw_candidates)}")
        if raw_candidates:
            for i, (txt, lp) in enumerate(raw_candidates[:3], 1):
                print(f"[architect_dfs_o] raw cand {i}: logprob={lp:.2f}")
                print(txt[:200])

        # 解析为网格
        parsed_candidates = []
        for text, logprob in raw_candidates:
            grid = self._codec.deserialize_grid(text)
            if grid is None:
                print("[architect_dfs_o] deserialize failed, text head:", text[:120])
                continue
            parsed_candidates.append((grid, logprob))

        print(f"[architect_dfs_o] parsed candidates: {len(parsed_candidates)}")
        
        if not parsed_candidates:
            # 如果首次 DFS 没有解析出候选，尝试放宽条件重试几次
            max_retries = 1
            for retry in range(1, max_retries + 1):
                print(f"[architect_dfs_o] no candidates, retry {retry}/{max_retries}")
                raw_candidates = _dfs_generate(
                    self._model,
                    self._tokenizer,
                    prompt_orig,
                    allowed_ids=self._allowed_ids,
                    max_steps=self._max_steps,
                    max_candidates=self._max_candidates,
                    top_k=len(self._allowed_ids), # min(self._top_k + retry * 2, len(self._allowed_ids)),
                    max_nodes=self._max_nodes, # self._max_nodes * (retry + 1),
                    min_prob=max(self._min_prob / (retry + 1) ** 2, 0.0), # max(self._min_prob / (retry + 1) ** 2, 0.0),
                )
                parsed_candidates = []
                for text, logprob in raw_candidates:
                    grid = self._codec.deserialize_grid(text)
                    if grid is None:
                        print(
                            "[architect_dfs_o] retry deserialize failed, text head:",
                            text[:120],
                        )
                        continue
                    parsed_candidates.append((grid, logprob))
                print(
                    f"[architect_dfs_o] retry {retry} parsed candidates: {len(parsed_candidates)}"
                )
                if parsed_candidates:
                    break

        if not parsed_candidates:
            return []

        # 2) 评分：对每个候选，遍历所有变换，计算平均对数概率
        candidates_out: list[dict] = []
        
        # 若只有一个候选，直接返回，不再耗时评分
        if len(parsed_candidates) == 1:
            grid, lp0 = parsed_candidates[0]
            print("[architect_dfs_o] single candidate, skip scoring")
            candidates_out.append({"grid": grid, "score": lp0, "meta": {"single": True}})
            return candidates_out
        
        transforms = self._architect.transform(task)
        print(f"[architect_dfs_o] transforms to score: {len(transforms)}")
        allowed_set = set(self._allowed_ids)

        for idx, (grid, _) in enumerate(parsed_candidates, 1):
            print(f"[architect_dfs_o] scoring candidate {idx}/{len(parsed_candidates)}")
            logprobs: list[float] = []
            for t_idx, record in enumerate(transforms, 1):
                transformed_task = record["task"]
                forward = record["forward"]
                prompt_t = self._codec.serialize_task(transformed_task, test_index=0)
                transformed_grid = forward(grid)
                target_text = self._codec.grid_to_text(transformed_grid)
                lp = _logprob_given(
                    self._model,
                    self._tokenizer,
                    prompt_t,
                    target_text,
                    allowed_ids=allowed_set,
                )
                logprobs.append(lp)
                if t_idx % 4 == 0 or t_idx == len(transforms):
                    print(
                        f"[architect_dfs_o] candidate {idx} scored {t_idx}/{len(transforms)} transforms"
                    )

            mean_logprob = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
            print(
                f"[architect_dfs_o] candidate {idx} mean_logprob={mean_logprob:.2f} "
                f"min={min(logprobs):.2f} max={max(logprobs):.2f}"
            )
            candidates_out.append(
                {
                    "grid": grid,
                    "score": mean_logprob,  # 仅平均对数概率
                    "meta": {
                        "mean_logprob": mean_logprob,
                        "transform_logprobs": logprobs,
                    },
                }
            )

        return candidates_out


def build_architect_solver(
    *,
    model_key: str,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
    max_new_tokens: int = 931,
    max_steps: int = 931,
    max_candidates: int = 16,
    top_k: int = 5,
    max_nodes: int = 1024,
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
    # 确保尽量使用 GPU
    try:
        import torch

        if torch.cuda.is_available():
            target_device = torch.device("cuda")
            model = model.to(target_device)
            print("[architect_dfs_o] model moved to", target_device)
        else:
            print("[architect_dfs_o] cuda not available, stay on CPU")
    except Exception as exc:  # 防止因模型分布式加载报错
        print(f"[architect_dfs_o] warn: failed to move model to cuda: {exc}")
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
