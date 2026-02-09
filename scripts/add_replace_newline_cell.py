import json
from pathlib import Path


def main() -> None:
    nb_path = Path("ARChitects/sft_lora_cut_msg_multi_stream.ipynb")
    content = nb_path.read_bytes().decode("utf-8")
    nb = json.loads(content)

    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        if (
            cell.get("cell_type") == "code"
            and cell.get("source")
            and str(cell["source"][0]).strip().startswith("# Build SFT training samples")
        ):
            insert_idx = i + 1
            break
    if insert_idx is None:
        raise SystemExit("could not find Build SFT training samples cell")
    src = """# Force prompt/target to use Ċ instead of literal newlines
try:
    _orig_iter_samples = iter_samples
except NameError:
    print("iter_samples not defined yet")
else:
    def iter_samples(*args, **kwargs):
        for sample in _orig_iter_samples(*args, **kwargs):
            p = sample.get("prompt")
            t = sample.get("target")
            if isinstance(p, str):
                sample["prompt"] = p.replace("\\n", "Ċ")
            if isinstance(t, str):
                sample["target"] = t.replace("\\n", "Ċ")
            yield sample
    print("iter_samples patched: \\n -> Ċ in prompt/target")
"""
    new_cell = {"cell_type": "code", "metadata": {}, "source": src.split("\n")}
    nb["cells"].insert(insert_idx, new_cell)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Inserted newline-rewrite cell at position {insert_idx}")


if __name__ == "__main__":
    main()
