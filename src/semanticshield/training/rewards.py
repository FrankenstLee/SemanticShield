"""Reward functions used during GRPO training."""

import logging
import re
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"<answer>\s*(Real|Fake)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_label(completion: str) -> Optional[str]:
    """Extract the Real/Fake label from a completion."""
    try:
        match = ANSWER_PATTERN.search(completion)
        if match:
            label = match.group(1).strip().lower()
            logger.debug("Extracted label: %s", label)
            return label
        logger.warning("Unable to extract label from completion: %s", completion)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Label extraction failed: %s", exc)
        return None


def format_reward(completions: Iterable[str], **kwargs) -> List[float]:
    """
    Reward completions that strictly match the required format:
    <think>...</think>\n<answer>\nReal|Fake\n</answer>
    """
    pattern = r"<think>\n.*?\n</think>\n<answer>\n(?:Real|Fake)\n</answer>"
    rewards: List[float] = []
    for i, completion in enumerate(completions):
        content = completion.strip()
        match = re.fullmatch(pattern, content, re.DOTALL)
        reward = 0.5 if match else 0.0
        rewards.append(reward)

        print(f"\n--- Completion #{i + 1} ---")
        print(content)
        print(f" format_reward: {reward}\n")

    return rewards


def user_reward_func(prompts, completions, task, **kwargs) -> List[float]:
    """Primary accuracy reward: +1 for correct label, -1 for incorrect."""
    rewards: List[float] = []
    for prompt, completion, expected_task in zip(prompts, completions, task):
        label = extract_label(completion)
        if label is None:
            logger.warning("Assigning -1 reward because label was not found: %s", completion)
            rewards.append(-1.0)
            continue

        expected = str(expected_task).strip().lower()
        if expected in ("real", "fake"):
            if label == expected:
                reward = 1.0
                logger.debug("Task=%s, prediction=%s, reward=%s", expected, label, reward)
            else:
                # Extra penalty when a fake user is predicted as real.
                if expected == "fake" and label == "real":
                    reward = -1.25
                    logger.debug("Task=%s, prediction=%s, severe penalty -> reward %s", expected, label, reward)
                else:
                    reward = -1.0
                    logger.debug("Task=%s, prediction=%s, reward=%s", expected, label, reward)
        else:
            logger.warning("Unknown task type: %s", expected_task)
            reward = 0.0

        rewards.append(reward)
    return rewards


def verbose_think_reward(prompts, completions, **kwargs) -> List[float]:
    """
    Encourage moderately detailed reasoning inside <think> ... </think>.
    Reward +0.25 if the reasoning contains 60-130 English words.
    """
    rewards: List[float] = []
    for i, completion in enumerate(completions):
        try:
            match = re.search(r"<think>\s*(.*?)\s*</think>", completion, re.DOTALL | re.IGNORECASE)
            if not match:
                rewards.append(0.0)
                logger.debug("[think length reward] Sample %s had no <think> content -> reward 0", i)
                continue

            think_text = match.group(1)
            words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", think_text)
            count = len(words)

            reward = 0.25 if (count > 60 and count < 130) else 0.0
            rewards.append(reward)
            logger.debug("[think length reward] Sample %s word_count=%s -> reward %s", i, count, reward)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[think length reward] Failed to parse completion: %s", exc)
            rewards.append(0.0)
    return rewards


def consistency_reward(prompts, completions, **kwargs) -> List[float]:
    """
    Penalize cases where the reasoning mentions the opposite label of the final answer.
    """
    rewards: List[float] = []
    for i, completion in enumerate(completions):
        try:
            answer_match = ANSWER_PATTERN.search(completion)
            answer_label = answer_match.group(1).strip().lower() if answer_match else None

            think_match = re.search(r"<think>(.*?)</think>", completion, re.IGNORECASE | re.DOTALL)
            think_text = think_match.group(1) if think_match else ""

            if answer_label:
                opposite_label = "real" if answer_label == "fake" else "fake"
                if re.search(rf"\b{opposite_label}\b", think_text, re.IGNORECASE):
                    reward = -0.5
                    logger.debug("[consistency reward] Sample %s think mentions opposite label %s -> reward %s", i, opposite_label, reward)
                else:
                    reward = 0.0
                    logger.debug("[consistency reward] Sample %s no opposite label found -> reward %s", i, reward)
            else:
                reward = 0.0
                logger.debug("[consistency reward] Sample %s had no <answer> label -> reward %s", i, reward)

            rewards.append(reward)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[consistency reward] Failed to parse completion: %s", exc)
            rewards.append(0.0)
    return rewards


def format_bonus_reward(prompts, completions, task, **kwargs) -> List[float]:
    """Reward +0.25 if the completion contains numbered list formatting."""
    rewards: List[float] = []
    for i, completion in enumerate(completions):
        if re.search(r"\d+\..*\n\d+\..*\n\d+\..*", completion):
            rewards.append(0.25)
            logger.debug("[format bonus] Sample %s contains numbered list -> reward 0.25", i)
        else:
            rewards.append(0.0)
    return rewards


def nonsense_penalty(prompts, completions, **kwargs) -> List[float]:
    """Penalty -0.5 when <think> contains very long nonsense tokens."""
    rewards: List[float] = []
    for i, completion in enumerate(completions):
        try:
            match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
            if not match:
                rewards.append(0.0)
                continue

            think_text = match.group(1).strip()
            words = re.findall(r"[A-Za-z]+", think_text)

            if any(len(word) > 20 for word in words):
                reward = -0.5
                logger.debug("[nonsense penalty] Sample %s contained overlong token -> reward %s", i, reward)
            else:
                reward = 0.0

            rewards.append(reward)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[nonsense penalty] Failed to parse completion: %s", exc)
            rewards.append(0.0)
    return rewards


__all__ = [
    "extract_label",
    "format_reward",
    "user_reward_func",
    "verbose_think_reward",
    "consistency_reward",
    "format_bonus_reward",
    "nonsense_penalty",
]


