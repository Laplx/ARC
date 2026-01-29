"""Minimal visualization utilities."""

from __future__ import annotations

import json
import os


def build(reports: dict, *, output_dir: str) -> list[str]:
    """Write a simple JSON report and return the artifact path."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "report.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=True, indent=2, default=str)
    return [output_path]
