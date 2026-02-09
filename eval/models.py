"""Hugging Face text generation wrapper (with mirror endpoint)."""

from __future__ import annotations

import os
from typing import Any


def _allowed_token_ids(tokenizer) -> list[int]:
    """Collect token ids that correspond to digits and newline (if single-token encoded)."""
    allowed: set[int] = set()
    for ch in "0123456789\n":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            allowed.add(ids[0])
    return sorted(allowed)


class HuggingFaceTextGenerator:
    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int = 512,
        tokenizer_id: str | None = None,
        chat_template_path: str | None = None,
        local_files_only: bool = False,
        restrict_digits: bool = False,
    ) -> None:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._tokenizer_id = tokenizer_id
        self._chat_template_path = chat_template_path
        self._local_files_only = local_files_only
        self._tokenizer = None
        self._model = None
        self._restrict_digits = restrict_digits
        self._allowed_ids: list[int] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer_id = self._tokenizer_id or self._model_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, local_files_only=self._local_files_only
        )
        if self._chat_template_path:
            with open(self._chat_template_path, "r", encoding="utf-8") as handle:
                self._tokenizer.chat_template = handle.read()
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            dtype=getattr(torch, "bfloat16", None),
            device_map="auto",
            local_files_only=self._local_files_only,
        )
        if self._restrict_digits:
            self._allowed_ids = _allowed_token_ids(self._tokenizer)

    def predict(self, inputs: Any, *, context=None) -> Any:
        self._load()
        return _generate_transformers(
            self._model,
            self._tokenizer,
            inputs,
            max_new_tokens=self._max_new_tokens,
            allowed_ids=self._allowed_ids,
        )

    def get_backend(self):
        self._load()
        return self._model, self._tokenizer


class HuggingFaceBackendTextGenerator:
    """Wrap a preloaded HF model/tokenizer to match the evaluation interface."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        max_new_tokens: int = 256,
        restrict_digits: bool = False,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._restrict_digits = restrict_digits
        self._allowed_ids: list[int] | None = None
        if self._restrict_digits:
            self._allowed_ids = _allowed_token_ids(self._tokenizer)

    def predict(self, inputs: Any, *, context=None) -> Any:
        return _generate_transformers(
            self._model,
            self._tokenizer,
            inputs,
            max_new_tokens=self._max_new_tokens,
            allowed_ids=self._allowed_ids,
        )

    def get_backend(self):
        return self._model, self._tokenizer


def _generate_transformers(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    allowed_ids: list[int] | None = None,
) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    prefix_allowed_tokens_fn = None
    if allowed_ids:
        allowed_set = set(allowed_ids)

        def _fn(batch_id, input_ids):
            return list(allowed_set)

        prefix_allowed_tokens_fn = _fn

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    input_len = encoded["input_ids"].shape[1]
    generated = output[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=False)
