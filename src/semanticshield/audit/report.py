"""Aggregate audit outputs and compute simple accuracy stats."""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple


def is_fake_user(user_id_line: str) -> bool:
    return "fake" in user_id_line.lower()


def extract_llm_answer(text_block: str):
    match = re.search(r"<answer>\s*(Real|Fake)\s*</answer>", text_block, re.IGNORECASE)
    return match.group(1).strip().lower() if match else None


def summarize_directory(llm_out_dir: Path) -> Tuple[List[dict], int, int]:
    txt_files = glob.glob(f"{llm_out_dir}/*.txt")
    all_stats = []
    total_users_all = 0
    correct_predictions_all = 0

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        stats = {
            "real": {"pred_real": 0, "pred_fake": 0},
            "fake": {"pred_real": 0, "pred_fake": 0},
        }

        current_type = None
        buffer_block: List[str] = []

        for line in lines:
            line = line.strip()

            if line.startswith("User:"):
                if current_type is not None and buffer_block:
                    answer = extract_llm_answer("\n".join(buffer_block))
                    if answer:
                        if current_type == "real":
                            if answer == "real":
                                stats["real"]["pred_real"] += 1
                            else:
                                stats["real"]["pred_fake"] += 1
                        else:
                            if answer == "fake":
                                stats["fake"]["pred_fake"] += 1
                            else:
                                stats["fake"]["pred_real"] += 1

                current_type = "fake" if is_fake_user(line) else "real"
                buffer_block = []
            else:
                buffer_block.append(line)

        if current_type is not None and buffer_block:
            answer = extract_llm_answer("\n".join(buffer_block))
            if answer:
                if current_type == "real":
                    if answer == "real":
                        stats["real"]["pred_real"] += 1
                    else:
                        stats["real"]["pred_fake"] += 1
                else:
                    if answer == "fake":
                        stats["fake"]["pred_fake"] += 1
                    else:
                        stats["fake"]["pred_real"] += 1

        file_stem = os.path.splitext(os.path.basename(txt_file))[0]
        all_stats.append({"file_name": file_stem, "real": stats["real"], "fake": stats["fake"]})

        total_real = stats["real"]["pred_real"] + stats["real"]["pred_fake"]
        total_fake = stats["fake"]["pred_real"] + stats["fake"]["pred_fake"]
        total_users_all += total_real + total_fake
        correct_predictions_all += stats["real"]["pred_real"] + stats["fake"]["pred_fake"]

    return all_stats, total_users_all, correct_predictions_all


def write_summary(all_stats: List[dict], total_users_all: int, correct_predictions_all: int, summary_path: Path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        for file_stats in all_stats:
            file_name = file_stats["file_name"]
            stats_real = file_stats["real"]
            stats_fake = file_stats["fake"]

            total_real = stats_real["pred_real"] + stats_real["pred_fake"]
            total_fake = stats_fake["pred_real"] + stats_fake["pred_fake"]

            total_file_users = total_real + total_fake
            correct_file_predictions = stats_real["pred_real"] + stats_fake["pred_fake"]

            real_correct_ratio = (stats_real["pred_real"] / total_real * 100) if total_real else 0
            fake_correct_ratio = (stats_fake["pred_fake"] / total_fake * 100) if total_fake else 0
            file_accuracy = (correct_file_predictions / total_file_users * 100) if total_file_users else 0

            f.write(f"File: {file_name}\n")
            f.write(f"Real Users Accuracy: {real_correct_ratio:.2f}% ({stats_real['pred_real']} / {total_real})\n")
            f.write(f"Fake Users Accuracy: {fake_correct_ratio:.2f}% ({stats_fake['pred_fake']} / {total_fake})\n")
            f.write(f"Overall File Accuracy: {file_accuracy:.2f}% ({correct_file_predictions} / {total_file_users})\n")
            f.write("=" * 60 + "\n\n")

        overall_accuracy = (correct_predictions_all / total_users_all * 100) if total_users_all else 0
        f.write("=" * 60 + "\n")
        f.write(f"Total Users: {total_users_all}\n")
        f.write(f"Total Correct Predictions: {correct_predictions_all}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
        f.write("=" * 60 + "\n")


def generate_report(llm_out_dir: Path, summary_stat_path: Path):
    all_stats, total_users_all, correct_predictions_all = summarize_directory(llm_out_dir)
    write_summary(all_stats, total_users_all, correct_predictions_all, summary_stat_path)
    print(f"\nSummary saved to: {summary_stat_path}")


def parse_args(args: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("./outputs/audit/Clothing"), help="Directory containing model outputs")
    parser.add_argument("--summary_path", type=Path, default=Path("./outputs/audit/Clothing/summary_report.txt"), help="Where to save summary stats")
    return parser.parse_args(args=args)


def main():
    args = parse_args()
    generate_report(args.input_dir, args.summary_path)


if __name__ == "__main__":
    main()


