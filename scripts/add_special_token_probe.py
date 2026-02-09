import json
from pathlib import Path


def main() -> None:
    nb_path = Path("ARChitects/sft_lora_cut_msg_multi_stream.ipynb")
    nb = json.loads(nb_path.read_text())

    src = """# Check whether special tokens exist in base/cut tokenizers
from transformers import AutoTokenizer

special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

print('--- base tokenizer (model_path) ---')
try:
    base_tok = AutoTokenizer.from_pretrained(
        cfg.model_path, use_fast=False, local_files_only=True
    )
    for t in special_tokens:
        print(f"{t}:", base_tok.convert_tokens_to_ids(t))
except Exception as e:
    print('load base tokenizer failed:', e)

print('--- cut tokenizer (cut_output_dir) ---')
try:
    cut_tok = AutoTokenizer.from_pretrained(cfg.cut_output_dir, use_fast=False)
    for t in special_tokens:
        print(f"{t}:", cut_tok.convert_tokens_to_ids(t))
except Exception as e:
    print('load cut tokenizer failed:', e)
"""

    cell = {"cell_type": "code", "metadata": {}, "source": src.split("\n")}

    # insert after config cell (index 3)
    nb["cells"].insert(4, cell)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
