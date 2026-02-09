from pathlib import Path

def main():
    p = Path("ARChitects/sft_lora_cut_msg_multi_stream.ipynb")
    data = p.read_bytes()
    print("size:", len(data))
    print("first40:", data[:40])

if __name__ == "__main__":
    main()
