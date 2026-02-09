import json
from pathlib import Path


def main() -> None:
    nb_path = Path("ARChitects/sft_lora_cut_msg_multi_stream.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    idx = None
    for i, cell in enumerate(nb["cells"]):
        if (
            cell.get("cell_type") == "code"
            and cell.get("source")
            and "Preview a few SFT samples" in cell["source"][0]
        ):
            idx = i
            break
    if idx is None:
        raise SystemExit("preview cell not found")

    new_src = """# Preview a few SFT samples (prompt/target)
from itertools import islice

def decode_ids(ids, is_label=False):
    if not isinstance(ids, (list, tuple)):
        try:
            ids = list(ids)
        except Exception:
            return None
    if is_label:
        ids = [i for i in ids if i != -100]
    ids = [i for i in ids if i >= 0 and i != tokenizer.pad_token_id]
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return None

def _preview(dataset, n=3):
    print(f\"dataset type: {type(dataset)}\")
    for idx, item in enumerate(islice(dataset, n)):
        prompt = item.get(\"prompt\") if isinstance(item, dict) else item[0]
        target = item.get(\"target\") if isinstance(item, dict) else item[1]
        if prompt is None or target is None:
            prompt = item.get(\"input_ids\")
            target = item.get(\"labels\")
        print(f\"--- sample {idx} ---\")
        if isinstance(prompt, str):
            print('prompt:', prompt[:200].replace('\\n','\\\\n'), '...')
        else:
            print('prompt ids len:', len(prompt))
            decoded = decode_ids(prompt)
            if decoded:
                print('prompt decoded:', decoded[:200].replace('\\n','\\\\n'), '...')
        if isinstance(target, str):
            print('target:', target[:200].replace('\\n','\\\\n'), '...')
        else:
            print('target ids len:', len(target))
            decoded_t = decode_ids(target, is_label=True)
            if decoded_t:
                print('target decoded:', decoded_t[:200].replace('\\n','\\\\n'), '...')

try:
    _preview(train_dataset, n=3)
except Exception as e:
    print('preview failed:', e)
"""

    nb["cells"][idx]["source"] = new_src.split("\n")
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"preview cell updated at {idx}")


if __name__ == "__main__":
    main()
