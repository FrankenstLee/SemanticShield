# count_score.py
log_path = "../logs/output4.log"
targets = ["<answer>\nReal\n</answer>","<answer>\nFake\n</answer>","奖励: 1.0","奖励: -1.0","严重错误"]

with open(log_path, 'r', encoding='utf-8') as f:
    content = f.read()

counts=[]
for target in targets:
    count = content.count(target)
    counts.append(count)
    print(f'🔢 "{target}" 出现次数: {count}')
print(f"ratio: {counts[0]/sum(counts[:2])}")
print(f"accuracy: {counts[-3]/(counts[-1]+counts[-2]+counts[-3])}")