"""Run SemanticShield auditing with a pretrained model."""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from transformers import pipeline

from semanticshield.audit.prompts import build_prompt


def run_audit(
    dataset: str,
    data_dir: Path,
    out_dir: Path,
    model_path: Path,
    device: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)

    generator = pipeline(
        "text-generation",
        model=str(model_path),
        device=device,
    )

    files = glob.glob(f"{data_dir}/*.json")

    for file in tqdm(files, desc="Processing files"):
        filename = os.path.basename(file)
        file_stem = os.path.splitext(filename)[0]
        out_path = out_dir / f"{file_stem}.txt"

        with open(file, "r") as f:
            data = json.load(f)

        with open(out_path, "w") as fout:
            for user_id, items in data.items():
                prompt = build_prompt(dataset, items)

                try:
                    res = generator(
                        [{"role": "user", "content": prompt}],
                        max_new_tokens=512,
                        temperature=0.1,
                        top_p=0.9,
                        top_k=50,
                        return_full_text=False,
                    )[0]["generated_text"].strip()
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    res = f"Error: {exc}"

                print(f"\nUser: {user_id}")
                print(res)
                print("=" * 60)

                fout.write(f"User: {user_id}\n\n")
                fout.write(res + "\n")
                fout.write("=" * 60 + "\n")
                fout.flush()
                os.fsync(fout.fileno())


def parse_args(args: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["Clothing", "MIND", "ml-1M"], help="Dataset name")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to input JSON files")
    parser.add_argument("--out_dir", type=Path, required=True, help="Path to save outputs")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model checkpoint")
    parser.add_argument("--device", type=int, default=0, help="GPU id (or -1 for CPU)")
    return parser.parse_args(args=args)


def main():
    args = parse_args()
    run_audit(args.dataset, args.data_dir, args.out_dir, args.model_path, args.device)


if __name__ == "__main__":
    main()


