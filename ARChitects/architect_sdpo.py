"""
SDPO 训练辅助模块。

特点：
- 不改动现有 architect / core 文件；仅新增便于 Notebook 复用的封装。
- 依赖 eval.core_sdpo 提供的样本构造与损失计算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from eval.core_sdpo import (
    SDPODataset,
    SDPOExample,
    build_sdpo_batch,
    build_sdpo_examples,
    sdpo_forward,
    _log_print,
)


# -----------------------------
# 配置
# -----------------------------


@dataclass
class SDPOConfig:
    model_path: str
    tokenizer_path: str
    report_path: str
    dataset_root: str
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 10
    max_steps: int = 1000
    loss_type: str = "js"
    top_k: int | None = 100
    use_8bit_optimizer: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 数据加载
# -----------------------------


def load_sdpo_dataset(config: SDPOConfig) -> SDPODataset:
    examples = build_sdpo_examples(
        report_path=config.report_path,
        dataset_root=config.dataset_root,
    )
    _log_print(f"[sdpo] dataset ready: {len(examples)} examples")
    return SDPODataset(examples)


def build_dataloader(
    tokenizer,
    dataset: SDPODataset,
    *,
    device: torch.device,
    batch_size: int,
) -> DataLoader:
    def _collate(batch: Sequence[SDPOExample]):
        # DataLoader 阶段保持在 CPU，避免预取/缓存占用 GPU 显存
        return build_sdpo_batch(tokenizer, batch, device=None)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)


# -----------------------------
# 训练器
# -----------------------------


class SDPOTrainer:
    def __init__(self, model, tokenizer, config: SDPOConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb

                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                _log_print("[sdpo] optimizer: bitsandbytes AdamW8bit")
            except Exception as exc:
                _log_print(f"[sdpo] optimizer fallback to torch AdamW: {exc}")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            _log_print("[sdpo] optimizer: torch AdamW")

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

    def train_step(self, batch):
        self.model.train()
        batch = batch.to(self.device)
        loss, _ = sdpo_forward(
            self.model,
            self.tokenizer,
            batch,
            loss_type=self.config.loss_type,
            top_k=self.config.top_k,
            return_aux=False,
            keep_only_target_logits=True,
        )
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        _log_print(f"[sdpo] train_step loss={loss.item():.4f}")
        return loss.item()

    @torch.no_grad()
    def quick_eval(self, batch):
        """快速检查教师/学生对 wrong_answer 的 logprob 提升情况。"""
        self.model.eval()
        batch = batch.to(self.device)
        loss, aux = sdpo_forward(
            self.model,
            self.tokenizer,
            batch,
            loss_type=self.config.loss_type,
            top_k=self.config.top_k,
            return_aux=True,
            keep_only_target_logits=False,
        )

        # 计算学生/教师在标签上的平均 logprob
        labels = batch.labels
        mask = labels.ne(-100)
        target_len = mask.sum(dim=1, keepdim=True)

        student_logp = torch.log_softmax(aux["student_logits"], dim=-1)
        teacher_logp = torch.log_softmax(aux["teacher_logits"], dim=-1)
        # 对齐长度
        if teacher_logp.size(1) != student_logp.size(1):
            teacher_logp = teacher_logp[:, -student_logp.size(1) :, :]

        idx = labels.clamp_min(0).unsqueeze(-1)
        student_lp = torch.gather(student_logp, -1, idx).squeeze(-1)
        teacher_lp = torch.gather(teacher_logp, -1, idx).squeeze(-1)

        student_mean = (student_lp * mask).sum(dim=1) / target_len.squeeze(1)
        teacher_mean = (teacher_lp * mask).sum(dim=1) / target_len.squeeze(1)

        metrics = {
            "sdpo_loss": loss.item(),
            "student_logprob": student_mean.mean().item(),
            "teacher_logprob": teacher_mean.mean().item(),
        }
        _log_print(
            "[sdpo] quick_eval",
            f"loss={metrics['sdpo_loss']:.4f}",
            f"student_lp={metrics['student_logprob']:.3f}",
            f"teacher_lp={metrics['teacher_logprob']:.3f}",
        )
        return metrics


# -----------------------------
# 模型装载入口
# -----------------------------


def load_model_and_tokenizer(config: SDPOConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    model_kwargs = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    # 确保有 pad_token 以便批量 padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    _log_print(f"[sdpo] model/tokenizer loaded from {config.model_path}")
    return model, tokenizer
