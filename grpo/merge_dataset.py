import json
import os

def merge_datasets(fake_file, real_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    fake_data = []
    real_data = []
    
    try:
        with open(fake_file, 'r', encoding='utf-8') as f:
            for line in f:
                fake_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {fake_file} 不存在")
    except json.JSONDecodeError as e:
        raise ValueError(f"文件 {fake_file} 格式错误: {str(e)}")
    
    try:
        with open(real_file, 'r', encoding='utf-8') as f:
            for line in f:
                real_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {real_file} 不存在")
    except json.JSONDecodeError as e:
        raise ValueError(f"文件 {real_file} 格式错误: {str(e)}")
    
    merged_data = []
    for item in fake_data:
        merged_data.append({"prompt": item["prompt"], "task": "fake"})
    for item in real_data:
        merged_data.append({"prompt": item["prompt"], "task": "real"})
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in merged_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        raise Exception(f"写入 {output_file} 失败: {str(e)}")
    
    print(f"成功合并数据集，生成 {output_file}，共 {len(merged_data)} 条数据")

if __name__ == "__main__":
    fake_file = "../datasets/original/fake.jsonl"
    real_file = "../datasets/original/real.jsonl"
    output_file = "../datasets/original/train.jsonl"
    
    try:
        merge_datasets(fake_file, real_file, output_file)
    except Exception as e:
        print(f"合并数据集失败: {str(e)}")