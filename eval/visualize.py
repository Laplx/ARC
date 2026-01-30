"""Minimal visualization utilities."""

from __future__ import annotations

import json
import os
import re


def build(reports: dict, *, output_dir: str) -> list[str]:
    """Write a simple JSON report and return the artifact path."""
    os.makedirs(output_dir, exist_ok=True)
    meta = reports.get("meta") if isinstance(reports, dict) else None
    model_key = meta.get("model_key") if isinstance(meta, dict) else None
    model_id = meta.get("model_id") if isinstance(meta, dict) else None
    split = meta.get("split") if isinstance(meta, dict) else None

    name_parts = ["report"]
    if model_key:
        name_parts.append(str(model_key))
    elif model_id:
        name_parts.append(str(model_id))
    if split:
        name_parts.append(str(split))

    raw_name = "_".join(name_parts)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
    output_path = os.path.join(output_dir, f"{safe_name}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=True, indent=2, default=str)
    return [output_path]
