import argparse
import json
from pathlib import Path


def get_shape(grid):
    if not isinstance(grid, list) or len(grid) == 0:
        return None, "output is not a non-empty 2D list"
    row_lengths = []
    for row in grid:
        if not isinstance(row, list):
            return None, "output has a non-list row"
        row_lengths.append(len(row))
    if len(set(row_lengths)) != 1:
        return None, "output rows have inconsistent lengths"
    return (len(grid), row_lengths[0]), None


def collect_outputs(task):
    outputs = []
    for section in ("train", "test"):
        items = task.get(section, [])
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if "output" in item:
                outputs.append((section, idx, item["output"]))
    return outputs


def check_file(path):
    with path.open("r", encoding="utf-8") as f:
        task = json.load(f)

    outputs = collect_outputs(task)
    if not outputs:
        return ["no output found"]

    issues = []
    expected_shape = None
    for section, idx, output in outputs:
        shape, err = get_shape(output)
        label = f"{section}[{idx}]"
        if err:
            issues.append(f"{label}: {err}")
            continue
        if expected_shape is None:
            expected_shape = shape
            continue
        if shape != expected_shape:
            issues.append(
                f"{label}: shape {shape} does not match expected {expected_shape}"
            )
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Check whether outputs in each task have consistent shapes"
    )
    parser.add_argument(
        "--root",
        default="data",
        help="data root directory (default: data)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    json_files = sorted(root.rglob("*.json"))

    if not json_files:
        print(f"No json files found under: {root}")
        return 1

    total = 0
    bad = 0
    for path in json_files:
        total += 1
        issues = check_file(path)
        if issues:
            bad += 1
            print(f"[ERROR] {path}")
            for issue in issues:
                print(f"  - {issue}")

    print()
    print(f"Done: {total} files checked, {bad} with issues")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
