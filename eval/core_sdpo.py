"""
SDPO (Self-Distillation Policy Optimization) 实用组件。

设计目标：
- 复用已有 GridCodec / dataset 逻辑，把候选采样报告转成 SDPO 可训练样本。
- 提供学生/教师前向输入构造、KL / JS 蒸馏损失计算。
- 仅生成新文件，不改动现有评估或采样代码。
"""

from __future__ import annotations

import json
import os
import inspect
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# from eval.core_cand import GridCodec, _compare_grid

# 日志相关
LOG_PATH = os.path.join("outputs", "sdpo.log")
_ORIG_PRINT = print


def _log_print(*args, **kwargs):
    """轻量日志：写文件并同步控制台，避免一次输出过大。"""
    text = " ".join(str(a) for a in args)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass
    _ORIG_PRINT(*args, **kwargs)
    
    
class GridCodec:
    def grid_to_text(self, grid: list[list[int]]) -> str:
        return "Ċ".join("".join(str(cell) for cell in row) for row in grid)
    
    def comptext_to_grid(self, text: str) -> list[list[int]] | None:
        # 将 report 中形如 "323232 | 787878 | 232323 | 878787 | 323232 | 787878" 的文本转换回二维整数列表
        rows = text.split(" | ")
        grid = []
        for row in rows:
            try:
                grid.append([int(ch) for ch in row])
            except ValueError:
                return None
        return grid

    def serialize_task(self, task: dict, *, test_index: int = 0) -> str:
        segments: list[str] = []
        for pair in task.get("train", []):
            grid_in = self.grid_to_text(pair["input"])
            grid_out = self.grid_to_text(pair["output"])
            segments.append(
                f"<|im_start|>userĊ{grid_in}<|im_end|><|im_start|>assistantĊ{grid_out}<|im_end|>"
            )

        test_items = task.get("test", [])
        if not test_items:
            return "Ċ".join(segments)

        grid_in = self.grid_to_text(test_items[test_index]["input"])
        segments.append(f"<|im_start|>userĊ{grid_in}<|im_end|><|im_start|>assistantĊ")
        return "Ċ".join(segments)
    
    def extract_to_answer(self, task: dict, *, test_index: int = 0) -> str | None:
        test_items = task.get("test", [])
        if not test_items:
            return None
        grid_out = test_items[test_index].get("input")
        if grid_out is None:
            return None
        return self.grid_to_text(grid_out)

    def deserialize_grid(self, text: str) -> list[list[int]] | None:
        # keep only allowed chars (digits + our newline marker) and strip leading/trailing separators
        # unify newline markers and keep only allowed chars
        text = text.replace("\n", "Ċ")
        text = text.replace("<|im_end|>", "Ċ")
        filtered = "".join(ch for ch in text if ch in _ALLOWED_CHARS).strip("Ċ")
        # print(f"Deserializing grid from text:\n{filtered}\n")
        if not filtered:
            return None

        lines = filtered.split("Ċ")
        rows: list[list[int]] = []

        # Extract only the first contiguous block of consistent-width rows.
        expected_width: int | None = None
        for line in lines:
            # stop at blank separators that may appear inside model output
            if not line.strip():
                break
            row = [int(ch) for ch in line if ch.isdigit()]
            if not row:
                continue
            if expected_width is None:
                expected_width = len(row)
                if expected_width < 1 or expected_width > 30:
                    return None
            # if width changes, we treat it as the start of garbage and stop collecting
            if len(row) != expected_width:
                break
            rows.append(row)
            # limit rows to 30 to match ARC constraints
            if len(rows) >= 30:
                break

        if not rows:
            return None
        if len(rows) > 30:
            rows = rows[:30]
        return rows
    

def _normalize_prediction(prediction: Any) -> Any:
    if isinstance(prediction, Mapping) and "grid" in prediction:
        return prediction.get("grid")
    if isinstance(prediction, list) and len(prediction) == 1:
        return prediction[0]
    return prediction

def _compare_grid(prediction: Any, truth: Any) -> bool:
    prediction = _normalize_prediction(prediction)
    if prediction is None or truth is None:
        return False
    return prediction == truth


# -----------------------------
# 数据结构
# -----------------------------


@dataclass
class SDPOExample:
    """单条 SDPO 训练样本"""

    task_id: str
    question_prompt: str  # 序列化后的“问题”文本（包含示例对 + 测试输入）
    wrong_text: str  # 错误答案序列化文本（目标序列）
    ground_truth_text: str  # 正确答案序列化文本（仅用于教师上下文）
    to_answer: str


class SDPODataset(Dataset):
    """将 SDPOExample 列表包装为 PyTorch Dataset"""

    def __init__(self, examples: Sequence[SDPOExample]):
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> SDPOExample:
        return self._examples[index]


# -----------------------------
# 样本构造
# -----------------------------


def _load_task(dataset_root: str, task_id: str) -> dict:
    path = os.path.join(dataset_root, f"{task_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _grid_to_text(codec: GridCodec, grid) -> str:
    """兼容 grid 为 list[list[int]] 或已压缩的字符串（'a | b | c'）。"""
    if isinstance(grid, str):
        return grid
    return codec.grid_to_text(grid)


def _grid_from_compact(codec: GridCodec, grid):
    """
    将 report 中压缩后的 grid 字段还原为 list[list[int]]。
    若已是列表直接返回。
    """
    if isinstance(grid, list):
        return grid
    if isinstance(grid, str):
        # 兼容 core_cand._compact_grids 产生的 "row1 | row2" 文本
        lines = [line.strip() for line in grid.split("|")]
        rows = []
        for line in lines:
            if not line:
                continue
            rows.append([int(ch) for ch in line if ch.isdigit()])
        return rows
    return None


def build_sdpo_examples(
    *,
    report_path: str,
    dataset_root: str,
    codec: GridCodec | None = None,
) -> List[SDPOExample]:
    """
    从评估报告（如 core_cand 生成的 report_xxx.json）中提取 SDPO 样本。

    规则：
    - 仅保留候选集中至少 1 条正确 + 1 条错误的题目。
    - 为每个错误候选生成一条样本（三元组 question, wrong, ground_truth）。
    """

    codec = codec or GridCodec()
    _log_print(f"[sdpo] load report: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    records = report.get("records", [])
    examples: list[SDPOExample] = []

    for idx, record in enumerate(records, 1):
        task_id = record.get("task_id")
        candidates = record.get("candidates") or []
        if not task_id or not candidates:
            continue

        if idx % 20 == 0:
            _log_print(f"[sdpo] scanning record #{idx} task={task_id}")

        task = _load_task(dataset_root, task_id)
        to_answer = task.get("test", [{}])[0].get("input")
        gt_grid = task.get("test", [{}])[0].get("output")
        if gt_grid is None or to_answer is None:
            continue

        ta_text = _grid_to_text(codec, to_answer)
        gt_text = _grid_to_text(codec, gt_grid)
        question_prompt = codec.serialize_task(task, test_index=0)

        correct, wrong = [], []
        for cand in candidates:
            grid_raw = cand.get("grid")
            grid = _grid_from_compact(codec, grid_raw)
            if grid is None:
                continue
            if _compare_grid(grid, gt_grid):
                correct.append(grid)
            else:
                wrong.append(grid)

        if not correct or not wrong:
            # 无可训练信号
            continue

        wrong_seen = set()
        for w in wrong:
            w_text = _grid_to_text(codec, w)
            key = w_text
            if key in wrong_seen:
                continue
            wrong_seen.add(key)

            examples.append(
                SDPOExample(
                    task_id=task_id,
                    question_prompt=question_prompt,
                    wrong_text=w_text,
                    ground_truth_text=gt_text,
                    to_answer=ta_text,
                )
            )

    _log_print(f"[sdpo] build_sdpo_examples: total {len(examples)} samples from {len(records)} records")
    return examples


# -----------------------------
# 文本 → 模型输入
# -----------------------------


@dataclass
class SDPOBatch:
    student_input_ids: torch.Tensor
    student_attention: torch.Tensor
    teacher_input_ids: torch.Tensor
    teacher_attention: torch.Tensor
    labels: torch.Tensor  # 与 student 序列等长；prompt 位置为 -100

    def to(self, device: torch.device) -> "SDPOBatch":
        return SDPOBatch(
            student_input_ids=self.student_input_ids.to(device),
            student_attention=self.student_attention.to(device),
            teacher_input_ids=self.teacher_input_ids.to(device),
            teacher_attention=self.teacher_attention.to(device),
            labels=self.labels.to(device),
        )


def build_sdpo_batch(
    tokenizer,
    examples: Sequence[SDPOExample],
    *,
    device: torch.device | None = None,
) -> SDPOBatch:
    """
    将一批 SDPOExample 转成张量。

    - student_prompt = question + wrong
    - teacher_prompt = question + ground_truth + to_answer + wrong
      并通过 labels 只对 wrong 部分做损失。
    """

    student_texts = []
    teacher_texts = []
    label_tensors = []

    for ex in examples:
        student_text = ex.question_prompt + ex.wrong_text
        teacher_text = ex.question_prompt + ex.ground_truth_text + ex.to_answer + ex.wrong_text

        student_enc = tokenizer(student_text, return_tensors=None)  # lists
        teacher_enc = tokenizer(teacher_text, return_tensors=None)

        student_len = len(student_enc["input_ids"])
        # teacher_len = len(teacher_enc["input_ids"])

        # wrong_answer token 数
        wrong_ids = tokenizer(ex.wrong_text, return_tensors=None)["input_ids"]
        wrong_len = len(wrong_ids)

        # 构造 labels（仅学生序列长度）
        labels = torch.full((student_len,), -100, dtype=torch.long)
        labels[-wrong_len:] = torch.tensor(wrong_ids, dtype=torch.long)

        # 统一收集
        student_texts.append(student_enc)
        teacher_texts.append(teacher_enc)
        label_tensors.append(labels)

    # pad 对齐
    student_batch = tokenizer.pad(student_texts, return_tensors="pt")
    teacher_batch = tokenizer.pad(teacher_texts, return_tensors="pt")
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        label_tensors, batch_first=True, padding_value=-100
    )

    batch = SDPOBatch(
        student_input_ids=student_batch["input_ids"],
        student_attention=student_batch["attention_mask"],
        teacher_input_ids=teacher_batch["input_ids"],
        teacher_attention=teacher_batch["attention_mask"],
        labels=labels_padded,
    )
    if device is not None:
        return batch.to(device)
    return batch


# -----------------------------
# SDPO 损失
# -----------------------------


def _truncate_to_target(logits: torch.Tensor, target_len: int) -> torch.Tensor:
    """裁剪到序列末尾的 target_len 长度（用于教师对齐学生的目标长度）。"""
    if logits.size(1) == target_len:
        return logits
    return logits[:, -target_len:, :]


def _model_supports_forward_kwarg(model, kwarg: str) -> bool:
    """
    判断 model.forward 是否可接收指定 kwarg。
    对于 PEFT 这类仅暴露 **kwargs 的包装层也返回 True。
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    if kwarg in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _compute_sdpo_loss_from_topk(
    *,
    student_top_logits: torch.Tensor,
    teacher_top_logits: torch.Tensor,
    loss_type: str = "js",
) -> torch.Tensor:
    """在教师 top-k 子空间上计算 SDPO 损失。"""
    student_log_probs = torch.log_softmax(student_top_logits, dim=-1)
    teacher_probs = torch.softmax(teacher_top_logits, dim=-1)

    if loss_type == "js":
        student_probs = torch.exp(student_log_probs)
        m = 0.5 * (teacher_probs + student_probs)
        kl_t = F.kl_div(torch.log(m + 1e-8), teacher_probs, reduction="none")
        kl_s = F.kl_div(student_log_probs, m, reduction="none")
        token_loss = 0.5 * (kl_t.sum(-1) + kl_s.sum(-1))
    else:
        token_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(-1)

    return token_loss.mean()


def _forward_with_optional_kwargs(model, *, input_ids, attention_mask, forward_kwargs: dict):
    """优先使用可选 forward kwargs；若模型不支持则自动回退。"""
    if not forward_kwargs:
        return model(input_ids=input_ids, attention_mask=attention_mask)
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, **forward_kwargs)
    except TypeError as exc:
        # 兼容“外层有 **kwargs，但底层模型不支持 logits_to_keep”的情况
        if "logits_to_keep" in forward_kwargs and "logits_to_keep" in str(exc):
            return model(input_ids=input_ids, attention_mask=attention_mask)
        raise


def _get_lm_backbone_and_head(model):
    """
    获取 CausalLM 的 backbone 与 lm_head。
    兼容 PEFT 包装（通过 get_base_model 解包）。
    """
    base_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm_head = getattr(base_lm, "lm_head", None)
    backbone = None
    for attr in ("model", "transformer", "backbone"):
        candidate = getattr(base_lm, attr, None)
        if candidate is not None:
            backbone = candidate
            break
    if backbone is None or lm_head is None or not hasattr(lm_head, "weight"):
        return None, None
    return backbone, lm_head


def _forward_backbone_last_hidden(backbone, *, input_ids, attention_mask):
    """前向 backbone 并返回 last_hidden_state。"""
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_dict": True,
    }
    try:
        kwargs["use_cache"] = False
        out = backbone(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        out = backbone(**kwargs)
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return out[0]
    return None


def _student_top_logits_from_hidden(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
    teacher_top_idx: torch.Tensor,
) -> torch.Tensor | None:
    """
    仅在教师 top-k 子词表上计算学生 logits，避免构造全词表 logits。
    返回形状 [N, K]，N 为有效标签 token 数。
    """
    backbone, lm_head = _get_lm_backbone_and_head(model)
    if backbone is None or lm_head is None:
        return None

    student_hidden = _forward_backbone_last_hidden(
        backbone,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    if student_hidden is None:
        return None

    target_len = teacher_mask.size(1)
    if student_hidden.size(1) < target_len:
        return None
    student_hidden = _truncate_to_target(student_hidden, target_len)
    if teacher_mask.sum() == 0:
        return None

    hidden_masked = student_hidden[teacher_mask]  # [N, H]
    del student_hidden

    # 按 token 动态 gather lm_head 权重：W[idx] -> [N, K, H]
    weight = lm_head.weight.to(hidden_masked.dtype)
    selected_weight = weight[teacher_top_idx]
    student_top = torch.einsum("nh,nkh->nk", hidden_masked, selected_weight)

    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        student_top = student_top + bias[teacher_top_idx].to(student_top.dtype)

    return student_top


def _is_gradient_checkpointing_enabled(model) -> bool:
    """兼容不同封装层判断 gradient checkpointing 开关。"""
    for attr in ("is_gradient_checkpointing", "gradient_checkpointing"):
        try:
            value = getattr(model, attr)
        except Exception:
            continue
        if isinstance(value, bool):
            return value
    return False


def _set_gradient_checkpointing(model, enabled: bool) -> None:
    """兼容不同封装层切换 gradient checkpointing。"""
    method = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    fn = getattr(model, method, None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def compute_sdpo_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "js",  # "js" 或 "kl"
    top_k: int | None = 100,
) -> torch.Tensor:
    """
    基于 wrong_answer 部分的 KL / JS 蒸馏损失。

    Args:
        student_logits: [B, Ls, V]
        teacher_logits: [B, Lt, V]（可能比学生长，会自动裁剪到学生长度）
        labels:        [B, Ls]，prompt 为 -100
        loss_type:     "js" (Jensen-Shannon) 或 "kl" (forward KL)
        top_k:         仅在教师的 top-k token 上计算，可节省显存；None 表示全量
    """

    if student_logits.size(1) != labels.size(1):
        labels = labels[:, -student_logits.size(1) :]

    target_len = labels.size(1)
    teacher_logits = _truncate_to_target(teacher_logits, target_len)
    student_logits = _truncate_to_target(student_logits, target_len)

    mask = labels.ne(-100)  # 只在 wrong_answer token 计算
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    # 仅保留掩码位置，减少长序列带来的显存开销
    student_logits = student_logits[mask]  # [N, V]
    teacher_logits = teacher_logits[mask]  # [N, V]

    if top_k is not None and top_k > 0:
        # 取教师 top-k token，保证学生在同一索引上对齐
        top_vals, top_idx = teacher_logits.topk(top_k, dim=-1)
        # gather 对学生 logits
        student_top = torch.gather(student_logits, -1, top_idx)
        teacher_top = top_vals

        student_log_probs = torch.log_softmax(student_top, dim=-1)
        teacher_probs = torch.softmax(teacher_top, dim=-1)
    else:
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)

    if loss_type == "js":
        # Jensen-Shannon: 0.5*KL(p||m)+0.5*KL(q||m)
        student_probs = torch.exp(student_log_probs)
        m = 0.5 * (teacher_probs + student_probs)
        kl_t = F.kl_div(torch.log(m + 1e-8), teacher_probs, reduction="none")
        kl_s = F.kl_div(student_log_probs, m, reduction="none")
        token_loss = 0.5 * (kl_t.sum(-1) + kl_s.sum(-1))
    else:
        token_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(-1)

    loss = token_loss.mean()
    return loss


def _drop_long_samples_from_batch(
    batch: SDPOBatch, *, max_seq_len: int | None
) -> tuple[SDPOBatch | None, int]:
    """
    丢弃 student token 长度超阈值的样本，并压缩剩余 batch 的 padding 长度。
    返回 (新 batch 或 None, 丢弃数量)。
    """
    if max_seq_len is None or max_seq_len <= 0 or not hasattr(batch, "student_attention"):
        return batch, 0

    student_lens = batch.student_attention.sum(dim=1)
    keep_mask = student_lens.le(max_seq_len)
    dropped = int((~keep_mask).sum().item())
    if dropped == 0:
        return batch, 0

    kept = int(keep_mask.sum().item())
    _log_print(
        f"[sdpo] drop long samples: dropped={dropped} kept={kept} max_seq_len={max_seq_len}"
    )
    if kept == 0:
        return None, dropped

    student_input_ids = batch.student_input_ids[keep_mask]
    student_attention = batch.student_attention[keep_mask]
    teacher_input_ids = batch.teacher_input_ids[keep_mask]
    teacher_attention = batch.teacher_attention[keep_mask]
    labels = batch.labels[keep_mask]

    # 重新压缩 padding 宽度，避免被被丢弃的超长样本拖大计算图。
    student_keep_len = int(student_attention.sum(dim=1).max().item())
    teacher_keep_len = int(teacher_attention.sum(dim=1).max().item())
    student_input_ids = student_input_ids[:, :student_keep_len]
    student_attention = student_attention[:, :student_keep_len]
    labels = labels[:, :student_keep_len]
    teacher_input_ids = teacher_input_ids[:, :teacher_keep_len]
    teacher_attention = teacher_attention[:, :teacher_keep_len]

    return (
        SDPOBatch(
            student_input_ids=student_input_ids,
            student_attention=student_attention,
            teacher_input_ids=teacher_input_ids,
            teacher_attention=teacher_attention,
            labels=labels,
        ),
        dropped,
    )


# -----------------------------
# 训练单步示例
# -----------------------------


def sdpo_forward(
    model,
    tokenizer,
    batch: SDPOBatch,
    *,
    loss_type: str = "js",
    top_k: int | None = 100,
    max_seq_len: int | None = 850,
    return_aux: bool = False,
    keep_only_target_logits: bool = False,
):
    """
    计算一批样本的 SDPO loss，返回 (loss, aux_info)。
    教师 logits 使用 torch.no_grad() 防止梯度回传。
    """
    batch_device = batch.student_input_ids.device
    batch, _dropped = _drop_long_samples_from_batch(batch, max_seq_len=max_seq_len)
    if batch is None:
        zero = torch.zeros((), device=batch_device, requires_grad=True)
        _log_print(f"[sdpo] skip batch: all samples dropped (>{max_seq_len}), loss=0.0000")
        return zero, None

    forward_kwargs = {}
    if keep_only_target_logits and hasattr(batch, "labels"):
        target_tokens = int(batch.labels.ne(-100).sum(dim=1).max().item())
        if target_tokens > 0:
            if _model_supports_forward_kwarg(model, "logits_to_keep"):
                forward_kwargs["logits_to_keep"] = target_tokens

    teacher_forward_kwargs = forward_kwargs
    student_forward_kwargs = forward_kwargs
    # Qwen3 + gradient checkpointing 在训练反传阶段对 logits_to_keep 兼容性不稳定，
    # 学生分支禁用该参数以避免 checkpoint 元数据不一致。
    if (
        not return_aux
        and "logits_to_keep" in forward_kwargs
        and _is_gradient_checkpointing_enabled(model)
    ):
        student_forward_kwargs = {
            k: v for k, v in forward_kwargs.items() if k != "logits_to_keep"
        }

    # 显存优化路径：
    # 在训练场景（return_aux=False）且使用 top-k 时，先把教师 logits 压缩到 top-k，
    # 避免教师全量 logits 与学生前向激活长时间重叠。
    if (
        not return_aux
        and top_k is not None
        and top_k > 0
        and hasattr(batch, "labels")
    ):
        labels = batch.labels
        gc_was_enabled = _is_gradient_checkpointing_enabled(model)
        if gc_was_enabled:
            _set_gradient_checkpointing(model, enabled=False)
        try:
            with torch.no_grad():
                teacher_out = _forward_with_optional_kwargs(
                    model,
                    input_ids=batch.teacher_input_ids,
                    attention_mask=batch.teacher_attention,
                    forward_kwargs=teacher_forward_kwargs,
                )
                teacher_logits = teacher_out.logits  # [B, Lt, V]
                del teacher_out
        finally:
            if gc_was_enabled:
                _set_gradient_checkpointing(model, enabled=True)

        teacher_logits = _truncate_to_target(teacher_logits, labels.size(1))
        labels_for_teacher = labels[:, -teacher_logits.size(1) :]
        teacher_mask = labels_for_teacher.ne(-100)
        if teacher_mask.sum() == 0:
            zero = torch.tensor(0.0, device=batch.student_input_ids.device)
            _log_print("[sdpo] forward batch: seq=0 loss=0.0000")
            return zero, None

        teacher_masked = teacher_logits[teacher_mask]  # [N, V]
        teacher_top_vals, teacher_top_idx = teacher_masked.topk(top_k, dim=-1)
        del teacher_masked
        del teacher_logits

        student_top = _student_top_logits_from_hidden(
            model=model,
            input_ids=batch.student_input_ids,
            attention_mask=batch.student_attention,
            teacher_mask=teacher_mask,
            teacher_top_idx=teacher_top_idx,
        )
        if student_top is None:
            print("[sdpo] Warning: model does not support efficient top-k forward; falling back to full logits")
            # 回退：若无法拿到 backbone/lm_head，则走原始全词表 logits 路径
            student_out = _forward_with_optional_kwargs(
                model,
                input_ids=batch.student_input_ids,
                attention_mask=batch.student_attention,
                forward_kwargs=student_forward_kwargs,
            )
            student_logits = student_out.logits  # [B, Ls, V]
            del student_out

            student_logits = _truncate_to_target(student_logits, teacher_mask.size(1))
            student_masked = student_logits[teacher_mask]  # [N, V]
            del student_logits

            student_top = torch.gather(student_masked, -1, teacher_top_idx)
            del student_masked
        else:
            pass

        del teacher_top_idx

        loss = _compute_sdpo_loss_from_topk(
            student_top_logits=student_top,
            teacher_top_logits=teacher_top_vals,
            loss_type=loss_type,
        )
        del student_top
        del teacher_top_vals

        seq = teacher_mask.size(1)
        _log_print(f"[sdpo] forward batch: seq={seq} loss={loss.item():.4f}")
        return loss, None

    gc_was_enabled = _is_gradient_checkpointing_enabled(model)
    if gc_was_enabled:
        _set_gradient_checkpointing(model, enabled=False)
    try:
        with torch.no_grad():
            teacher_out = _forward_with_optional_kwargs(
                model,
                input_ids=batch.teacher_input_ids,
                attention_mask=batch.teacher_attention,
                forward_kwargs=teacher_forward_kwargs,
            )
            teacher_logits = teacher_out.logits  # [B, Lt, V]
            del teacher_out
    finally:
        if gc_was_enabled:
            _set_gradient_checkpointing(model, enabled=True)

    student_out = _forward_with_optional_kwargs(
        model,
        input_ids=batch.student_input_ids,
        attention_mask=batch.student_attention,
        forward_kwargs=student_forward_kwargs,
    )
    student_logits = student_out.logits  # [B, Ls, V]
    del student_out

    loss = compute_sdpo_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=batch.labels,
        loss_type=loss_type,
        top_k=top_k,
    )

    if hasattr(batch, "labels"):
        seq = batch.labels.size(1)
    else:
        seq = student_logits.size(1)
    _log_print(f"[sdpo] forward batch: seq={seq} loss={loss.item():.4f}")

    if return_aux:
        return loss, {"student_logits": student_logits, "teacher_logits": teacher_logits}

    del student_logits
    del teacher_logits
    return loss, None
