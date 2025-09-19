import os
import re
import glob

llm_out_dir = './extra_out/Clothing'
summary_stat_path = os.path.join(llm_out_dir, 'summary_report.txt')

os.makedirs(llm_out_dir, exist_ok=True)

total_real_all = 6040  
total_fake_all = 60    

def is_fake_user(user_id_line):
    return 'fake' in user_id_line.lower()

def extract_llm_answer(text_block):
    match = re.search(r'<answer>\s*(Real|Fake)\s*</answer>', text_block, re.IGNORECASE)
    return match.group(1).strip().lower() if match else None

all_stats = []
txt_files = glob.glob(f'{llm_out_dir}/*.txt')

for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    stats = {
        'real': {'pred_real': 0, 'pred_fake': 0},
        'fake': {'pred_real': 0, 'pred_fake': 0}
    }

    current_type = None
    buffer_block = []

    for line in lines:
        line = line.strip()

        if line.startswith("User:"):
            if current_type is not None and buffer_block:
                answer = extract_llm_answer("\n".join(buffer_block))
                if answer:
                    if current_type == 'real':
                        if answer == 'real':
                            stats['real']['pred_real'] += 1
                        else:
                            stats['real']['pred_fake'] += 1
                    else:
                        if answer == 'fake':
                            stats['fake']['pred_fake'] += 1
                        else:
                            stats['fake']['pred_real'] += 1
                else:
                    if current_type == 'real':
                        stats['real']['pred_fake'] += 1
                    else:
                        stats['fake']['pred_real'] += 1

            current_type = 'fake' if is_fake_user(line) else 'real'
            buffer_block = []
        else:
            buffer_block.append(line)

    if current_type is not None and buffer_block:
        answer = extract_llm_answer("\n".join(buffer_block))
        if answer:
            if current_type == 'real':
                if answer == 'real':
                    stats['real']['pred_real'] += 1
                else:
                    stats['real']['pred_fake'] += 1
            else:
                if answer == 'fake':
                    stats['fake']['pred_fake'] += 1
                else:
                    stats['fake']['pred_real'] += 1
        else:
            if current_type == 'real':
                stats['real']['pred_fake'] += 1
            else:
                stats['fake']['pred_real'] += 1

    file_stem = os.path.splitext(os.path.basename(txt_file))[0]
    file_stats = {
        'file_name': file_stem,
        'real': stats['real'],
        'fake': stats['fake']
    }
    all_stats.append(file_stats)

total_fake_users_all = 0
total_fake_correct_all = 0
max_real_users = 0
max_real_correct = 0

with open(summary_stat_path, 'w', encoding='utf-8') as f:
    for file_stats in all_stats:
        file_name = file_stats['file_name']
        stats_real = file_stats['real']
        stats_fake = file_stats['fake']

        total_real = stats_real['pred_real'] + stats_real['pred_fake']
        total_fake = stats_fake['pred_real'] + stats_fake['pred_fake']

        if total_real > max_real_users:
            max_real_users = total_real
            max_real_correct = stats_real['pred_real']

        total_fake_users_all += total_fake
        total_fake_correct_all += stats_fake['pred_fake']

        real_correct_ratio = (stats_real['pred_real'] / total_real * 100) if total_real else 0
        fake_correct_ratio = (stats_fake['pred_fake'] / total_fake * 100) if total_fake else 0

        far_ratio = stats_real['pred_fake'] / total_real_all * 100

        real_enter_llm_ratio = (total_real / total_real_all * 100)
        fake_enter_llm_ratio = (total_fake / total_fake_all * 100)

        f.write(f"File: {file_name}\n")

        f.write(f"User Type: REAL USERS\n")
        f.write(f"Total Real Users Evaluated: {total_real}\n")
        f.write(f"LLM Judged Real (Correct): {stats_real['pred_real']}\n")
        f.write(f"LLM Judged Fake (False Positive): {stats_real['pred_fake']}\n")
        f.write(f"Accuracy: {real_correct_ratio:.2f}%\n")
        f.write(f"FAR (False Acceptance Rate): {stats_real['pred_fake']} / {total_real_all} = {far_ratio:.2f}%\n")
        f.write("=" * 40 + "\n")

        f.write(f"User Type: FAKE USERS\n")
        f.write(f"Total Fake Users Evaluated: {total_fake}\n")
        f.write(f"LLM Judged Fake (Correct): {stats_fake['pred_fake']}\n")
        f.write(f"LLM Judged Real (False Negative): {stats_fake['pred_real']}\n")
        f.write(f"Accuracy: {fake_correct_ratio:.2f}%\n")
        f.write("=" * 40 + "\n")

        f.write(f"Real Users Entered LLM Judgment: {total_real} / {total_real_all} ({real_enter_llm_ratio:.2f}%)\n")
        f.write(f"Fake Users Entered LLM Judgment: {total_fake} / {total_fake_all} ({fake_enter_llm_ratio:.2f}%)\n")
        f.write("=" * 60 + "\n\n")

    total_users_all = max_real_users + total_fake_users_all
    correct_predictions_all = max_real_correct + total_fake_correct_all
    overall_accuracy = (correct_predictions_all / total_users_all * 100) if total_users_all else 0

    f.write("=" * 60 + "\n")
    f.write(f"总体真实用户数: {max_real_users}\n")
    f.write(f"总体虚假用户数: {total_fake_users_all}\n")
    f.write(f"总体用户数: {total_users_all}\n")
    f.write(f"总体预测正确数: {correct_predictions_all}\n")
    f.write(f"总体准确率: {overall_accuracy:.2f}%\n")
    f.write("=" * 60 + "\n")

print(f"\n 已完成：每个文件的 FAR 已单独统计，结果保存在：{summary_stat_path}")
