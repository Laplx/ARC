"""Lightweight solvers for ARC evaluation."""

from __future__ import annotations

from typing import Mapping


def _extract_assistant_content(output, prompt: str) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict) and "text" in output:
        return str(output["text"])
    text = str(output)
    if prompt and text.startswith(prompt):
        return text[len(prompt) :]
    return text


class RawSolver:
    def __init__(self, *, model, codec) -> None:
        self._model = model
        self._codec = codec

    def solve(self, task, *, context: Mapping[str, object] | None = None):
        tests = task.get("test", [])
        candidates: list[dict] = []

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
