"""Convert training prompts to the Qwen chat template."""

import json
from pathlib import Path


def process_jsonl(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            entry = json.loads(line.strip())
            prompt = entry["prompt"]
            task = entry["task"]

            first_newline_idx = prompt.find("\n")
            if first_newline_idx == -1:
                system_content = prompt
                user_content = ""
            else:
                system_content = prompt[:first_newline_idx]
                user_content = prompt[first_newline_idx + 1 :]

            qwen_prompt = (
                "<|im_start|>system\n"
                f"{system_content}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{user_content}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant"
            )

            json.dump({"prompt": qwen_prompt, "task": task}, outfile, ensure_ascii=False)
            outfile.write("\n")


def main():
    project_root = Path(__file__).resolve().parents[3]
    input_file = project_root / "data" / "datasets" / "original" / "train.jsonl"
    output_file = project_root / "data" / "datasets" / "original" / "train_qwen.jsonl"
    process_jsonl(input_file, output_file)


if __name__ == "__main__":
    main()


