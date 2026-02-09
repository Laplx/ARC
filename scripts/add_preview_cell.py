import json
from pathlib import Path


def main() -> None:
    nb_path = Path("ARChitects/sft_lora_cut_msg_multi_stream.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        if (
            cell.get("cell_type") == "code"
            and cell.get("source")
            and str(cell["source"][0]).strip().startswith("# Build PyTorch Dataset")
        ):
            insert_idx = i + 1
            break
    if insert_idx is None:
        raise SystemExit("target cell not found")

    new_src = """# Preview a few SFT samples (prompt/target)
from itertools import islice

def _preview(dataset, n=3):
    print(f"dataset type: {type(dataset)}")
    for idx, item in enumerate(islice(dataset, n)):
        # items may be dict (from build_sft_samples) or tensors (ArcSFTDataset)
        prompt = item.get("prompt") if isinstance(item, dict) else item[0]
        target = item.get("target") if isinstance(item, dict) else item[1]
        if prompt is None or target is None:
            prompt = item.get("input_ids")
            target = item.get("labels")
        print(f"--- sample {idx} ---")
        if isinstance(prompt, str):
            print("prompt:", prompt[:200].replace("\\n", "\\\\n"), "...")
        else:
            print("prompt ids len:", len(prompt))
        if isinstance(target, str):
            print("target:", target[:200].replace("\\n", "\\\\n"), "...")
        else:
            print("target ids len:", len(target))

try:
    _preview(train_dataset, n=3)
except Exception as e:
    print("preview failed:", e)
"""
    new_cell = {"cell_type": "code", "metadata": {}, "source": new_src.split("\n")}
    nb["cells"].insert(insert_idx, new_cell)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Inserted preview cell at position {insert_idx}")


if __name__ == "__main__":
    main()
