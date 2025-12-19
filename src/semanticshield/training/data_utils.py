"""Data utilities for preparing SemanticShield training data."""

import json
from pathlib import Path
from typing import Iterable


def merge_datasets(fake_file: Path, real_file: Path, output_file: Path) -> int:
    """Merge fake and real JSONL files into a single training JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fake_data = _load_jsonl(fake_file)
    real_data = _load_jsonl(real_file)

    merged_data = []
    for item in fake_data:
        merged_data.append({"prompt": item["prompt"], "task": "fake"})
    for item in real_data:
        merged_data.append({"prompt": item["prompt"], "task": "real"})

    with output_file.open("w", encoding="utf-8") as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(merged_data)


def _load_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


__all__ = ["merge_datasets"]


