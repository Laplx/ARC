"""Evaluation entrypoint."""

from __future__ import annotations

import argparse

from eval import failure, metrics, postprocess, runner, visualize
from eval.adapters.architects import NVARCArchitect
from eval.adapters.codec import GridCodec
from eval.adapters.datasets import ARCDataset
from eval.adapters.models import ModelScopeTextGenerator
from eval.adapters.solvers import ArchitectSolver, RawSolver


MODEL_REGISTRY = {
    "mistral-7b": "modelscope/Mistral-7B-v0.1",
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking",
    "llama-3.2-3b": "LLM-Research/Llama-3.2-3B",
    "llama-3.2-8b": "LLM-Research/Llama-3.2-8B",
}


def _get_truth(task, context):
    tests = task.get("test") or []
    if len(tests) != 1:
        return None
    return tests[0].get("output")


def _get_task_id(task, context):
    return task.get("task_id", context.get("index"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARC baseline evaluation")
    parser.add_argument("--data-root", default="data", help="dataset root")
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--model", default="mistral-7b", help="model key or full ModelScope id")
    parser.add_argument("--max-tasks", type=int, default=0, help="limit tasks (0 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--use-architect", action="store_true")
    args = parser.parse_args(argv)

    model_id = MODEL_REGISTRY.get(args.model, args.model)

    dataset = ARCDataset(root=args.data_root, split=args.split, max_tasks=args.max_tasks)
    codec = GridCodec()
    model = ModelScopeTextGenerator(model_id=model_id, max_new_tokens=args.max_new_tokens)

    base_solver = RawSolver(model=model, codec=codec)
    solver = base_solver

    if args.use_architect:
        architect = NVARCArchitect()
        solver = ArchitectSolver(base_solver=base_solver, architect=architect)

    config = {
        "get_truth": _get_truth,
        "get_task_id": _get_task_id,
        "output_dir": args.output_dir,
    }

    results = runner.run(
        dataset=dataset,
        solver=solver,
        metrics=metrics,
        postprocess=postprocess,
        failure=failure,
        visualize=visualize,
        config=config,
    )

    if results.get("summary"):
        print(results["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
