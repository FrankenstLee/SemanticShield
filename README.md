# SemanticShield

SemanticShield implements a two-stage defense for shilling-attack detection: behavioral pre-screening plus LLM-based semantic auditing, strengthened with GRPO fine-tuning.

## Repository layout (project root: `SemanticShield/`)
- `src/semanticshield/` — library code (training + audit).
- `scripts/` — CLI entrypoints (train, merge data, chat templating, audit, summarize, log stats).
- `data/datasets/` — training data (`original/` JSONL).
- `data/audit/` — item-side JSON for auditing (data4llm, extra attack sets).
- `logs/` — log outputs (created during runs).
- `outputs/` — generated audit outputs (created by scripts).
- `checkpoints/` — training artifacts (created during runs).

## Setup
```bash
cd SemanticShield
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training pipeline
```bash
# 1) Merge labeled data
python scripts/merge_dataset.py

# 2) Convert to Qwen chat template
python scripts/chat_template.py

# 3) GRPO multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu scripts/train.py
# checkpoints -> ./checkpoints/model/
```

## Run auditing
```bash
python scripts/audit_users.py \
  --dataset Clothing \
  --data_dir ./data/audit/data4llm/Clothing \
  --out_dir ./outputs/audit/Clothing \
  --model_path ./checkpoints/model \
  --device 0
```

Use the released model instead:
```bash
pip install -U "huggingface_hub[cli]"
hf download Luka772001/SS --local-dir ./SemanticShield --local-dir-use-symlinks False

python scripts/audit_users.py \
  --dataset Clothing \
  --data_dir ./data/audit/data4llm/Clothing \
  --out_dir ./outputs/audit/Clothing \
  --model_path ./SemanticShield \
  --device 0
```

## Summarize audit results
```bash
python scripts/gen.py \
  --input_dir ./outputs/audit/Clothing \
  --summary_path ./outputs/audit/Clothing/summary_report.txt
```

## Notes
- Core logic unchanged; layout is now src+scripts+data to stay conventionally organized.
- Defaults expect the base model at `./Qwen2.5-1.5B-Instruct`; adjust paths as needed.
- Behavior-stage filters are assumed precomputed; this repo focuses on LLM auditing and GRPO refinement.