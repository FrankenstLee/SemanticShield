"""Quick counter for common patterns in training logs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log_path = PROJECT_ROOT / "logs" / "output.log"
targets = [
    "<answer>\nReal\n</answer>",
    "<answer>\nFake\n</answer>",
    "Reward: 1.0",
    "Reward: -1.0",
    "Severe error",
]

with open(log_path, "r", encoding="utf-8") as f:
    content = f.read()

counts = []
for target in targets:
    count = content.count(target)
    counts.append(count)
    print(f'"{target}" count: {count}')

print(f"ratio: {counts[0] / sum(counts[:2])}")
print(f"accuracy: {counts[-3] / (counts[-1] + counts[-2] + counts[-3])}")


