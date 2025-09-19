import re
import logging
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import os
from collections import Counter

os.environ["WANDB_MODE"] = "disabled"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    logging.info("开始加载数据集")
    train_dataset = load_dataset("json", data_files="../datasets/original/train_qwen.jsonl", split="train")
    logging.info(f"数据集加载完成: 训练集 {len(train_dataset)} 条")
    logging.debug(f"数据集示例: {train_dataset[0]}")
except FileNotFoundError:
    logging.error("train.jsonl 文件不存在，请检查路径")
    raise
except Exception as e:
    logging.error(f"数据集加载失败: {str(e)}")
    raise

def extract_label(completion: str):
    try:
        match = re.search(r"<answer>\s*(Real|Fake)\s*</answer>", completion, re.IGNORECASE | re.DOTALL)
        if match:
            label = match.group(1).strip().lower()  # 'real' or 'fake'
            logging.debug(f"提取标签成功: {label}")
            return label
        logging.warning(f"无法提取标签: {completion}")
        return None
    except Exception as e:
        logging.error(f"提取标签失败: {str(e)}")
        return None

def format_reward(completions, **kwargs):
    pattern = r"<think>\n.*?\n</think>\n<answer>\n(?:Real|Fake)\n</answer>"
    rewards = []
    for i, completion in enumerate(completions):
        content = completion.strip()  
        match = re.fullmatch(pattern, content, re.DOTALL)  
        reward = 0.5 if match else 0.0
        rewards.append(reward)

        print(f"\n--- Completion #{i+1} ---")
        print(content)
        print(f" format_reward: {reward}\n")

    return rewards

def user_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for i, (prompt, completion, t) in enumerate(zip(prompts, completions, task)):
        label = extract_label(completion)   
        if label is None:
            logging.warning(f"奖励设为 -1，因为无法提取标签: {completion}")
            rewards.append(-1.0)
            continue

        expected = str(t).strip().lower()  
        if expected in ("real", "fake"):
            if label == expected:
                reward = 1.0
                logging.debug(f"任务: {expected}, 预测: {label}, 奖励: {reward}")
            else:
                # 特殊情况：恶意用户(fake)被识别成正常(real)，额外惩罚 -0.25
                if expected == "fake" and label == "real":
                    reward = -1.25
                    logging.debug(f"任务: {expected}, 预测: {label}, 严重错误 -> 奖励 {reward}")
                else:
                    reward = -1.0
                    logging.debug(f"任务: {expected}, 预测: {label}, 奖励: {reward}")
        else:
            logging.warning(f"未知任务类型: {t}")
            reward = 0.0

        rewards.append(reward)
    return rewards


def verbose_think_reward(prompts, completions, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        try:
            m = re.search(r"<think>\s*(.*?)\s*</think>", completion, re.DOTALL | re.IGNORECASE)
            if not m:
                rewards.append(0.0)
                logging.debug(f"[think长度奖励] 用户 {i} 未找到 <think> 内容，奖励 0")
                continue

            think_text = m.group(1)

            # 仅统计英文单词（包含撇号的词，如 don't）
            words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", think_text)
            count = len(words)

            reward = 0.25 if (count > 60 and count <130) else 0.0
            rewards.append(reward)
            logging.debug(f"[think长度奖励] 用户 {i} 单词数={count} -> 奖励 {reward}")
        except Exception as e:
            logging.error(f"[think长度奖励] 解析失败: {str(e)}")
            rewards.append(0.0)
    return rewards


def consistency_reward(prompts, completions, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        try:
            # 提取 <answer> 最终标签
            answer_match = re.search(r"<answer>\s*(Real|Fake)\s*</answer>", completion, re.IGNORECASE | re.DOTALL)
            answer_label = answer_match.group(1).strip().lower() if answer_match else None

            # 提取 <think> 内容
            think_match = re.search(r"<think>(.*?)</think>", completion, re.IGNORECASE | re.DOTALL)
            think_text = think_match.group(1) if think_match else ""

            if answer_label:
                opposite_label = "real" if answer_label == "fake" else "fake"

                # 检查 think 中是否包含相反标签
                if re.search(rf"\b{opposite_label}\b", think_text, re.IGNORECASE):
                    reward = -0.5
                    logging.debug(f"[一致性奖励] 用户 {i} think 含反标签 {opposite_label} -> 扣分 {reward}")
                else:
                    reward = 0.0
                    logging.debug(f"[一致性奖励] 用户 {i} think 中未发现标签 -> 奖励 {reward}")
            else:
                reward = 0.0
                logging.debug(f"[一致性奖励] 用户 {i} 无 answer 标签 -> 奖励 {reward}")

            rewards.append(reward)
        except Exception as e:
            logging.error(f"[一致性奖励] 解析失败: {str(e)}")
            rewards.append(0.0)
    return rewards


def format_bonus_reward(prompts, completions, task, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        if re.search(r"\d+\..*\n\d+\..*\n\d+\..*", completion):
            rewards.append(0.25)
            logging.debug(f"[格式奖励] 用户 {i} 包含编号格式，奖励+0.25")
        else:
            rewards.append(0.0)
    return rewards

def nonsense_penalty(prompts, completions, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        try:
            m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
            if not m:
                rewards.append(0.0)
                continue

            think_text = m.group(1).strip()
            words = re.findall(r"[A-Za-z]+", think_text)

            if any(len(w) > 20 for w in words):
                reward = -0.5
                logging.debug(f"[无意义词惩罚] 用户 {i} 出现超长单词 -> 扣分 {reward}")
            else:
                reward = 0.0

            rewards.append(reward)
        except Exception as e:
            logging.error(f"[无意义词惩罚] 解析失败: {str(e)}")
            rewards.append(0.0)
    return rewards

# 训练配置
config = GRPOConfig(
    output_dir="/media2/lkh/models/v4",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=8192,
    log_completions=True,
    report_to="wandb",
    run_name="test",
    logging_steps=10,
    logging_first_step=True,
    generation_kwargs={
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
    }
)

try:
    logging.info("开始初始化 GRPOTrainer")
    trainer = GRPOTrainer(
        model="/media2/lkh/models/v3/checkpoint-6441",
        reward_funcs=[user_reward_func, format_reward, format_bonus_reward, verbose_think_reward, consistency_reward, nonsense_penalty],
        train_dataset=train_dataset,
        args=config,
    )
    logging.info("开始训练")
    trainer.train()
    logging.info("训练完成")
except Exception as e:
    logging.error(f"训练失败: {str(e)}")
    raise
