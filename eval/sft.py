"""SFT helpers for ARC-style grid prompting."""

from __future__ import annotations

from typing import Any

import torch


def build_sft_samples(dataset, codec) -> list[dict[str, Any]]:
    """Build prompt/target samples from a dataset using the provided codec."""
    samples: list[dict[str, Any]] = []
    for task in dataset:
        tests = task.get("test", [])
        for test_index, test in enumerate(tests):
            if "output" not in test:
                continue
            prompt = codec.serialize_task(task, test_index=test_index)
            target = codec.grid_to_text(test["output"])
            samples.append(
                {
                    "task_id": task.get("task_id"),
                    "test_index": test_index,
                    "prompt": prompt,
                    "target": target,
                }
            )
    return samples


class ArcSFTDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, *, max_seq_len: int) -> None:
        self._samples = samples
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._samples[idx]
        prompt_ids = self._tokenizer(item["prompt"], add_special_tokens=False)[
            "input_ids"
        ]
        target_ids = self._tokenizer(item["target"], add_special_tokens=False)[
            "input_ids"
        ]

        eos = self._tokenizer.eos_token_id
        input_ids = prompt_ids + target_ids + ([eos] if eos is not None else [])
        labels = [-100] * len(prompt_ids) + target_ids + ([eos] if eos is not None else [])

        if len(input_ids) > self._max_seq_len:
            input_ids = input_ids[-self._max_seq_len :]
            labels = labels[-self._max_seq_len :]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


def collate_sft(batch: list[dict[str, torch.Tensor]], *, tokenizer) -> dict[str, torch.Tensor]:
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
