"""ModelScope text generation wrapper."""

from __future__ import annotations

from typing import Any

from eval.interfaces import Model


class ModelScopeTextGenerator(Model):
    def __init__(self, *, model_id: str, max_new_tokens: int = 256) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self._pipeline = pipeline(Tasks.text_generation, model=self._model_id)

    def predict(self, inputs: Any, *, context=None) -> Any:
        self._load()
        result = self._pipeline(
            inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            num_beams=1,
            top_p=1.0,
            top_k=0,
        )
        return _extract_text(result)


def _extract_text(result: Any) -> str:
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        if "generated_text" in result:
            return result["generated_text"]
        if "text" in result:
            return result["text"]
    return str(result)
