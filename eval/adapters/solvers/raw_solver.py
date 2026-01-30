"""Raw solver adapter."""

from __future__ import annotations

from typing import Mapping

from eval.interfaces import Candidate, Prediction, Solver


class RawSolver(Solver):
    def __init__(self, *, model, codec) -> None:
        self._model = model
        self._codec = codec

    def solve(self, task, *, context: Mapping[str, Any] | None = None) -> Prediction:
        tests = task.get("test", [])
        candidates: list[Candidate] = []

        if not tests:
            return candidates

        for index in range(len(tests)):
            prompt = self._codec.serialize_task(task, test_index=index)
            output = self._model.predict(prompt, context=context)
            snippet = _extract_assistant_content(output, prompt)
            grid = self._codec.deserialize_grid(snippet)
            if grid is None:
                candidates.append({})
            else:
                candidates.append({"grid": grid, "raw": output})

        return candidates


def _extract_assistant_content(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        text = text[len(prompt) :]

    end_markers = ["<|im_end|>", "<|im_start|>user"]
    end_positions = [text.find(mark) for mark in end_markers if text.find(mark) != -1]
    if end_positions:
        text = text[: min(end_positions)]
    return text
