"""Interfaces for evaluation components.

All interfaces are intentionally minimal and ARC-agnostic. Concrete implementations
should live outside this module.
"""
from __future__ import annotations

from typing import Any, Iterator, Mapping, Protocol, TypedDict


Task = Any
Prediction = Any


class Candidate(TypedDict, total=False):
    """A candidate output for evaluation.

    The evaluation layer treats ``grid`` as the primary comparable output. Other
    fields are optional and carry auxiliary info for analysis.
    """

    grid: Any
    program: Any
    prompt: Any
    trace: Any
    score: float
    meta: Mapping[str, Any]


class Dataset(Protocol):
    def __iter__(self) -> Iterator[Task]:
        ...

    def info(self) -> dict:
        ...


class Solver(Protocol):
    def solve(self, task: Task, *, context: Mapping[str, Any] | None = None) -> Prediction:
        ...


class Model(Protocol):
    def predict(self, inputs: Any, *, context: Mapping[str, Any] | None = None) -> Any:
        ...


class Architect(Protocol):
    def transform(self, task: Task) -> list[Task]:
        ...

    def score(
        self,
        task: Task,
        candidate: Candidate,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        ...

    def select(
        self,
        task: Task,
        candidates: list[Candidate],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Prediction:
        ...
