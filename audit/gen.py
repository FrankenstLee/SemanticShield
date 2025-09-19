import os
import re
import glob

llm_out_dir = './extra_out/Clothing'
summary_stat_path = os.path.join(llm_out_dir, 'summary_report.txt')

os.makedirs(llm_out_dir, exist_ok=True)

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

    file_stem = os.path.splitext(os.path.basename(txt_file))[0]
    file_stats = {
        'file_name': file_stem,
        'real': stats['real'],
        'fake': stats['fake']
    }
    all_stats.append(file_stats)

# 汇总
total_users_all = 0
correct_predictions_all = 0

with open(summary_stat_path, 'w', encoding='utf-8') as f:
    for file_stats in all_stats:
        file_name = file_stats['file_name']
        stats_real = file_stats['real']
        stats_fake = file_stats['fake']

        total_real = stats_real['pred_real'] + stats_real['pred_fake']
        total_fake = stats_fake['pred_real'] + stats_fake['pred_fake']

        total_file_users = total_real + total_fake
        correct_file_predictions = stats_real['pred_real'] + stats_fake['pred_fake']

        real_correct_ratio = (stats_real['pred_real'] / total_real * 100) if total_real else 0
        fake_correct_ratio = (stats_fake['pred_fake'] / total_fake * 100) if total_fake else 0
        file_accuracy = (correct_file_predictions / total_file_users * 100) if total_file_users else 0

        f.write(f"File: {file_name}\n")
        f.write(f"Real Users Accuracy: {real_correct_ratio:.2f}% ({stats_real['pred_real']} / {total_real})\n")
        f.write(f"Fake Users Accuracy: {fake_correct_ratio:.2f}% ({stats_fake['pred_fake']} / {total_fake})\n")
        f.write(f"Overall File Accuracy: {file_accuracy:.2f}% ({correct_file_predictions} / {total_file_users})\n")
        f.write("=" * 60 + "\n\n")

        total_users_all += total_file_users
        correct_predictions_all += correct_file_predictions

    overall_accuracy = (correct_predictions_all / total_users_all * 100) if total_users_all else 0
    f.write("=" * 60 + "\n")
    f.write(f"Total Users: {total_users_all}\n")
    f.write(f"Total Correct Predictions: {correct_predictions_all}\n")
    f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
    f.write("=" * 60 + "\n")

print(f"\n✅ 已完成：只计算准确率，结果保存在：{summary_stat_path}")
