"""Compatibility wrapper for merging training datasets."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from semanticshield.training.data_utils import merge_datasets


if __name__ == "__main__":
    fake_file = PROJECT_ROOT / "data" / "datasets" / "original" / "fake.jsonl"
    real_file = PROJECT_ROOT / "data" / "datasets" / "original" / "real.jsonl"
    output_file = PROJECT_ROOT / "data" / "datasets" / "original" / "train.jsonl"

    try:
        total = merge_datasets(fake_file, real_file, output_file)
        print(f"Merged datasets -> {output_file} ({total} records)")
    except Exception as exc:
        print(f"Failed to merge datasets: {exc}")


