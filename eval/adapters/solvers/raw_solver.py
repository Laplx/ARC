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
            grid = self._codec.deserialize_grid(_strip_prompt(output, prompt))
            if grid is None:
                candidates.append({})
            else:
                candidates.append({"grid": grid, "raw": output})

        return candidates


def _strip_prompt(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :]
    return text
