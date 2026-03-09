from __future__ import annotations

import math
import os
from typing import Mapping

import torch
import torch.nn.functional as F

from eval.core_C import GridCodec
from eval.models import HuggingFaceTextGenerator


from pathlib import Path
LOG_PATH = Path("outputs/architect_cand.log")

def _log_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _ORIG_PRINT(*args, **kwargs)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

import builtins

# 防止重复包裹：仅首次保存原 print
if not hasattr(builtins, "_orig_print_archbeam"):
    builtins._orig_print_archbeam = builtins.__dict__["print"]

_ORIG_PRINT = builtins._orig_print_archbeam
builtins.print = _log_print


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


def _allowed_token_ids(tokenizer) -> set[int]:
    allowed: set[int] = set()
    for ch in "0123456789Ċ":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        allowed.update(ids)
        
    end_token_ids = {
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }
    allowed.update(end_token_ids)
    return allowed


def _beam_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    allowed_ids: set[int],
    max_steps: int,
    max_candidates: int,
    top_k: int,
    # beam_width: int,
    # beam_num: int,
    max_rollouts: int,
    min_prob: float,
    # relax_factor: float = 1.0,
) -> list[tuple[str, float]]:
    """
    
    """
    if not allowed_ids:
        return []

    encoded = tokenizer(prompt, return_tensors="pt")
    device = model.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    print(
        "[architect_beam] Beam/DFS start",
        f"device={device}",
        f"allowed={len(allowed_ids)}",
        f"max_steps={max_steps}",
        f"top_k={top_k}",
        # f"beam_width={beam_width}",
        # f"beam_num={beam_num}",
    )

    start_ids = encoded["input_ids"][0]
    from collections import deque

    stack: list[torch.Tensor] = [start_ids]
    results: set[tuple[str, float]] = set()
    rollouts = 0

    log_minprob = float("-inf")
    if min_prob is not None and min_prob > 0.0:
        log_minprob = math.log(min_prob)

    end_token_ids = [
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ]

    with torch.no_grad():
        input_ids = stack.pop()
        while len(results) < 2 and rollouts < 4:
            rollout_result, new_text_ids = _rollout(
                model,
                tokenizer,
                input_ids.unsqueeze(0),
                allowed_ids=allowed_ids,
                end_token_ids=end_token_ids,
                max_steps=max_steps,
                top_k=top_k,
                log_minprob=log_minprob,
                do_sample=True,
                need_prune=False,  # 初始探索阶段不剪枝，保证有足够候选进入后续阶段
                prompt_orig_len=len(start_ids),
            )
            rollouts += 1
            if rollout_result is not None and rollout_result not in results:
                # print(f"[architect_beam] initial rollout found candidate, logprob={rollout_result[1]:.3f}, seq head={tokenizer.decode(rollout_result[0][len(start_ids):len(start_ids)+20], skip_special_tokens=False)}")
                results.add(rollout_result)
                stack.extend(new_text_ids)
            print(f"[architect_beam] initial rollouts, progress: rollouts={rollouts}, results={len(results)}, stack={len(stack)}")
            
            # 计算第一个 result 的
        
        while stack and rollouts < max_rollouts:
            if len(results) < max_candidates // 2:
                relax_factor = 2.0
            else:
                relax_factor = 1.0
                
            input_ids = stack.pop()
            rollout_result, new_text_ids = _rollout(
                model,
                tokenizer,
                input_ids.unsqueeze(0),
                allowed_ids=allowed_ids,
                end_token_ids=end_token_ids,
                max_steps=max_steps,
                top_k=top_k,
                log_minprob=log_minprob * relax_factor,  # 根据当前候选数量动态调整剪枝阈值
                do_sample=False,
                prompt_orig_len=len(start_ids),
            )
            
            rollouts += 1
            if rollout_result is not None and rollout_result not in results:
                if len(results) >= max_candidates:
                    min_item = min(results, key=lambda x: x[1])
                    if rollout_result[1] <= min_item[1]:
                        continue
                    results.remove(min_item)
                results.add(rollout_result)
                stack.extend(new_text_ids)

            print(
                f"[architect_beam] progress: rollouts={rollouts}, results={len(results)}, stack={len(stack)}"
            )
        
        candidates = []
        for result in results:
            candidates.append((tokenizer.decode(result[0][len(start_ids):], skip_special_tokens=False), result[1]))
        
    return candidates


def _rollout(
    model,
    tokenizer,
    partial_generated_ids: torch.Tensor,
    *,
    allowed_ids: set[int] | None = None,
    end_token_ids: list[int] | None = None,
    max_steps: int,
    top_k: int,
    log_minprob: float,
    do_sample: bool = False,
    need_prune: bool = True,
    prompt_orig_len: int,
) -> tuple[tuple[torch.Tensor, float], list[torch.Tensor]]:
    """
    从 partial_generated 的当前状态继续生成（非贪婪），直到遇到结束 token 或达到 max_steps。
    返回最终生成的文本及其对数概率（若未被剪枝），以及生成过程中不确定性较大的步骤（待再生成的部分文本）。
    """

    # 阈值
    # gap_thr = 0.10  # 概率差阈值
    entropy_thr = 1.0  # 较高熵才触发
    # pmax_thr = 0.8  # 最高概率过低时更可能触发
    
    prompt_len = len(partial_generated_ids[0])-prompt_orig_len
    # print(f"[architect_beam] rollout start, prompt_len={prompt_len}, prompt_orig_len={prompt_orig_len}")
    if prompt_len >= max_steps:
        return None, []
    
    output = model.generate(
        input_ids=partial_generated_ids,
        max_new_tokens=max_steps-prompt_len,
        eos_token_id=end_token_ids,
        do_sample=do_sample,
        temperature=1.25,
        output_scores=True,  # 关键：获取概率
        return_dict_in_generate=True,
    )
    # print(f"{output.sequences.shape}, {len(output.scores)}")
    
    scores_tensor = torch.stack(output.scores).squeeze(1)  # [steps, vocab]
    log_probs = F.log_softmax(scores_tensor, dim=-1).squeeze(1)  # [steps, vocab]

    # 计算熵
    # entropy_list = -torch.sum(log_probs.exp() * log_probs, dim=-1)  # [steps]
    
    allow_ids_tensor = torch.tensor(list(allowed_ids), device=scores_tensor.device)
    allow_scores = scores_tensor[:, allow_ids_tensor]  # [steps, len(allow_ids)]
    
    eps = 1e-12
    allow_probs = F.softmax(allow_scores, dim=-1)
    allow_log_probs = torch.log(allow_probs + eps)

    entropy_list = -torch.sum(allow_probs * allow_log_probs, dim=-1) # [steps]
    entropy_list = torch.nan_to_num(entropy_list, nan=0.0)

    # 剪枝（直接用log_probs）
    cumulative = 0
    token_ids = output.sequences[0, prompt_len+prompt_orig_len:]
    
    allowed_mask = torch.zeros(log_probs.size(-1), device=model.device, dtype=torch.bool)
    allowed_mask[allow_ids_tensor] = True
    if not allowed_mask[token_ids].all():
        return None, []

    token_log_probs = log_probs[torch.arange(len(token_ids), device=model.device), token_ids]
    cumulative = token_log_probs.sum().item()
        
    if cumulative < log_minprob and need_prune:
        print(f"[architect_beam] rollout pruned by logprob {cumulative:.3f} < {log_minprob:.3f}")
        return None, []
    
    # if token_ids[-1].item() not in end_token_ids:
    #     print(f"warning: rollout ended without EOS, last token_id={token_ids[-1].item()}")
    
    flag = False
    max_ent = entropy_list.max().item()
    threshold = max(0.4, max_ent * 0.5) if prompt_len < 64 else max(0.9, max_ent * 0.7)  # 两个条件的并集
    print(f"[architect_beam] rollout entropy stats: max={max_ent:.3f}, threshold={threshold:.3f}")

    # 找到第一个满足条件的位置
    for i, ent in enumerate(entropy_list):
        if ent > threshold:
            flag = True
            print(f"[architect_beam] trigger beam @len={i} entropy={ent:.3f} (max={max_ent:.3f}) prompt_len={prompt_len} cum logprob={cumulative:.3f}")
            break
        
    # # 先选取最大熵的位置，若该位置的熵也不高，则不触发
    # if max_ent > threshold:
    #     flag = True
    #     i = entropy_list.argmax().item()
    #     print(f"[architect_beam] trigger beam by max entropy @len={i} entropy={max_ent:.3f} threshold={threshold:.3f} prompt_len={prompt_len} cum logprob={cumulative:.3f}")    
        
    if not flag:
        print(f"[architect_beam] no beam triggered, max entropy={max_ent:.3f}, threshold={threshold:.3f}, cum logprob={cumulative:.3f}")
        return (tuple(output.sequences[0].tolist()), round(cumulative, 3)), []
        
    # 获取该位置的对数概率
    # target_logits = scores_tensor[i]  # [vocab_size]
    target_log_probs = log_probs[i]  # [vocab_size]

    # 在allow_ids中取top_k
    # allow_ids = torch.tensor(allowed_ids)
    # allow_log_probs = target_log_probs[allow_ids]
    topk_values, topk_indices = torch.topk(target_log_probs[allow_ids_tensor], k=top_k)

    # 获取实际的token_id
    topk_token_ids = [allow_ids_tensor[idx] for idx in topk_indices.tolist()]

    # 构建新文本段列表
    new_text_ids = []
    base_seq = output.sequences[0, :prompt_len + prompt_orig_len + i]
    
    for token_id in topk_token_ids:
        token_tensor = torch.tensor([token_id], device=base_seq.device)
        new_seq = torch.cat([base_seq, token_tensor])
        print(f"new seq for beam: {tokenizer.decode(new_seq[prompt_orig_len:], skip_special_tokens=False)}, logprob={target_log_probs[token_id].item():.3f}")
        new_text_ids.append(new_seq)

    return (tuple(output.sequences[0].tolist()), cumulative), new_text_ids[::-1]  # 后生成的先探索


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

    # 编码整个序列
    encoded = tokenizer(prompt + target_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # 编码 prompt 部分
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].size(1)
    target_len = input_ids.size(1) - prompt_len
    
    # 检查 allowed_ids
    if allowed_ids is not None:
        target_ids = input_ids[0, prompt_len:].tolist()
        for tid in target_ids:
            if tid not in allowed_ids:
                return float("-inf")
    
    with torch.no_grad():
        # 单次前向传播计算所有 token 的 logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 计算每个目标 token 的对数概率
        logprobs = torch.log_softmax(logits, dim=-1)
        
        # 提取目标 token 对应的对数概率
        target_logprobs = []
        for i in range(target_len):
            pos = prompt_len - 1 + i
            tid = input_ids[0, pos + 1].item()
            lp = logprobs[0, pos, tid].item()
            target_logprobs.append((tid, lp))
    
    # 求和
    logprob = sum(lp for _, lp in target_logprobs)
    print(f"[architect_beam] logprob_given: target_len={target_len}, total_logprob={logprob:.3f}")
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
        top_k: int = 3,
        # beam_width: int = 4,
        # beam_num: int = 1,
        max_rollouts: int = 16,
        min_prob: float = 0,
        # lambda_weight: float = 0.0,  # 保留参数但评分中不再使用 count
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._codec = codec
        self._architect = architect
        self._max_steps = max_steps
        self._max_candidates = max_candidates
        self._top_k = top_k
        # self._beam_width = beam_width
        # self._beam_num = beam_num
        self._max_rollouts = max_rollouts
        self._min_prob = min_prob
        # self._lambda_weight = lambda_weight
        self._allowed_ids = _allowed_token_ids(tokenizer)

    def solve(self, task, *, context: Mapping[str, object] | None = None):
        print("[architect_beam] start solve")
        # 1) 预生成：仅对原任务做一次 Beam，得到候选网格。
        prompt_orig = self._codec.serialize_task(task, test_index=0)
        print("[architect_beam] prompt sample (orig):")
        print(prompt_orig[:300] + ("..." if len(prompt_orig) > 300 else ""))
        print("[architect_beam] running Beam on original task")

        raw_candidates = _beam_generate(
            self._model,
            self._tokenizer,
            prompt_orig,
            allowed_ids=self._allowed_ids,
            max_steps=self._max_steps,
            max_candidates=self._max_candidates,
            top_k=self._top_k,
            min_prob=self._min_prob,
            max_rollouts=self._max_rollouts,
        )
        
        print(f"[architect_beam] Beam done, raw candidates: {len(raw_candidates)}")
        if raw_candidates:
            for i, (txt, lp) in enumerate(raw_candidates[:3], 1):
                print(f"[architect_beam] raw cand {i}: logprob={lp:.2f}")
                print(txt[:200])

        # 解析为网格
        parsed_candidates = []
        for text, logprob in raw_candidates:
            grid = self._codec.deserialize_grid(text)
            if grid is None:
                print("[architect_beam] deserialize failed, text head:", text[:120],
                      "\n logprob:", logprob)
                continue
            parsed_candidates.append((grid, logprob))
        print(f"[architect_beam] parsed candidates: {len(parsed_candidates)}")

        # if not parsed_candidates:
        #     return []

        # 2) 评分：对每个候选，不再：遍历所有变换，计算平均对数概率
        candidates_out: list[dict] = []
        
        for grid, lp0 in parsed_candidates:
            candidates_out.append({"grid": grid, "score": lp0, "meta": {"single": True}})
        
        return candidates_out

        for idx, (grid, _) in enumerate(parsed_candidates, 1):
            print(f"[architect_beam] scoring candidate {idx}/{len(parsed_candidates)}")
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
                        f"[architect_beam] candidate {idx} scored {t_idx}/{len(transforms)} transforms"
                    )

            mean_logprob = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
            print(
                f"[architect_beam] candidate {idx} mean_logprob={mean_logprob:.2f} "
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
    top_k: int = 3,
    # beam_width: int = 4,
    # beam_num: int = 1,
    max_rollouts: int = 16,
    min_prob: float = 0.05,
    # lambda_weight: float = 0.0,
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
            print("[architect_beam] model moved to", target_device)
        else:
            print("[architect_beam] cuda not available, stay on CPU")
    except Exception as exc:  # 防止因模型分布式加载报错
        print(f"[architect_beam] warn: failed to move model to cuda: {exc}")
    return ArchitectSolver(
        model=model,
        tokenizer=tokenizer,
        codec=GridCodec(),
        architect=NVARCArchitect(),
        max_steps=max_steps,
        max_candidates=max_candidates,
        top_k=top_k,
        # beam_width=beam_width,
        # beam_num=beam_num,
        max_rollouts=max_rollouts,
        min_prob=min_prob,
        # lambda_weight=lambda_weight,
    )
