"""GRPO training entrypoint for SemanticShield."""

import logging
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from semanticshield.training.rewards import (
    consistency_reward,
    format_bonus_reward,
    format_reward,
    nonsense_penalty,
    user_reward_func,
    verbose_think_reward,
)

logger = logging.getLogger(__name__)


def load_train_dataset(dataset_path: Path):
    """Load the training dataset from a JSONL file."""
    try:
        logger.info("Loading training dataset from %s", dataset_path)
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        logger.info("Dataset loaded: %s samples", len(dataset))
        logger.debug("First sample: %s", dataset[0])
        return dataset
    except FileNotFoundError:
        logger.error("Training dataset not found at %s", dataset_path)
        raise
    except Exception as exc:
        logger.error("Failed to load training dataset: %s", exc)
        raise


def build_trainer(
    train_dataset,
    model_path: Path,
    output_dir: Path,
    run_name: str = "semanticshield-grpo",
) -> GRPOTrainer:
    """Configure and create a GRPOTrainer."""
    config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=8192,
        log_completions=True,
        report_to="none",
        run_name=run_name,
        logging_steps=10,
        logging_first_step=True,
        generation_kwargs={
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
        },
    )

    return GRPOTrainer(
        model=str(model_path),
        reward_funcs=[
            user_reward_func,
            format_reward,
            format_bonus_reward,
            verbose_think_reward,
            consistency_reward,
            nonsense_penalty,
        ],
        train_dataset=train_dataset,
        args=config,
    )


def train(
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """Run GRPO training with optional overrides for paths."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    os.environ["WANDB_MODE"] = "disabled"

    project_root = Path(__file__).resolve().parents[3]
    dataset_path = dataset_path or project_root / "data" / "datasets" / "original" / "train_qwen.jsonl"
    model_path = model_path or project_root / "Qwen2.5-1.5B-Instruct"
    output_dir = output_dir or project_root / "checkpoints" / "model"

    train_dataset = load_train_dataset(dataset_path)

    try:
        logger.info("Initializing GRPOTrainer")
        trainer = build_trainer(train_dataset, model_path, output_dir)
        logger.info("Starting training")
        trainer.train()
        logger.info("Training finished")
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise


def main():
    """CLI entrypoint."""
    train()


if __name__ == "__main__":
    main()


