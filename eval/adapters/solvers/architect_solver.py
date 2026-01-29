"""Architect wrapper solver."""

from __future__ import annotations

from typing import Mapping

from eval.interfaces import Candidate, Prediction, Solver


class ArchitectSolver(Solver):
    def __init__(self, *, base_solver: Solver, architect) -> None:
        self._base_solver = base_solver
        self._architect = architect

    def solve(self, task, *, context: Mapping[str, Any] | None = None) -> Prediction:
        candidates: list[Candidate] = []
        for transformed in self._architect.transform(task):
            outputs = self._base_solver.solve(transformed, context=context)
            if isinstance(outputs, list):
                outputs_list = outputs
            else:
                outputs_list = [outputs]
            for output in outputs_list:
                candidate = output if isinstance(output, dict) else {"grid": output}
                meta = dict(candidate.get("meta") or {})
                meta["transform"] = transformed.get("transform")
                candidate["meta"] = meta
                candidates.append(candidate)
        return candidates
