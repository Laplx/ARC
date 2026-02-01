"""Baseline Evaluation entrypoint."""

from __future__ import annotations

import argparse
import os

from eval.core import ARCDataset, GridCodec, run_evaluation
from eval.models import HuggingFaceTextGenerator
from eval.solvers import RawSolver


MODEL_REGISTRY = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    # "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "deepseek-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def _get_truth(task, context):
    tests = task.get("test") or []
    if len(tests) != 1:
        return None
    return tests[0].get("output")


def _get_task_id(task, context):
    return task.get("task_id", context.get("index"))

def _resolve_model_id(model_key: str, model_path: str | None) -> tuple[str, bool]:
    if model_path:
        path = os.path.expanduser(model_path)
        if os.path.isfile(path):
            return os.path.dirname(path), True
        return path, True
    return MODEL_REGISTRY.get(model_key, model_key), False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARC baseline evaluation")
    parser.add_argument("--data-root", default="data", help="dataset root")
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--model", default="mistral-7b", help="model key or full Hugging Face id")
    parser.add_argument("--model-path", default="", help="local model dir or weights file")
    parser.add_argument("--tokenizer-path", default="", help="local tokenizer dir (optional)")
    parser.add_argument("--local-files-only", action="store_true", help="do not fetch from hub")
    parser.add_argument("--max-tasks", type=int, default=0, help="limit tasks (0 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--viz-failures", action="store_true", help="write failure text report")
    args = parser.parse_args(argv)

    model_id, from_local = _resolve_model_id(args.model, args.model_path or None)
    local_only = bool(args.local_files_only or from_local)
    tokenizer_id = args.tokenizer_path or None

    dataset = ARCDataset(root=args.data_root, split=args.split, max_tasks=args.max_tasks)
    codec = GridCodec()
    model = HuggingFaceTextGenerator(
        model_id=model_id,
        max_new_tokens=args.max_new_tokens,
        tokenizer_id=tokenizer_id,
        local_files_only=local_only,
    )

    solver = RawSolver(model=model, codec=codec)

    results = run_evaluation(
        dataset=dataset,
        solver=solver,
        get_truth=_get_truth,
        get_task_id=_get_task_id,
        output_dir=args.output_dir,
        model_id=model_id,
        model_key=args.model,
        viz_failures=args.viz_failures,
        max_tasks=args.max_tasks,
        cleanup_every=1,
    )

    if results.get("summary"):
        print(results["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
