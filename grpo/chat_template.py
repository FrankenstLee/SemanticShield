import json
import os

def process_jsonl(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            entry = json.loads(line.strip())
            prompt = entry['prompt']
            task = entry['task']
            
            first_newline_idx = prompt.find('\n')
            if first_newline_idx == -1:
                system_content = prompt
                user_content = ""
            else:
                system_content = prompt[:first_newline_idx]
                user_content = prompt[first_newline_idx + 1:]
            
            # Construct the Qwen template
            qwen_prompt = f"<|im_start|>system\n{system_content}\n<|im_end|>\n<|im_start|>user\n{user_content}\n<|im_end|>\n<|im_start|>assistant"
            
            new_entry = {
                "prompt": qwen_prompt,
                "task": task
            }
            
            json.dump(new_entry, outfile, ensure_ascii=False)
            outfile.write('\n')

input_file = '../datasets/original/train.jsonl'
output_file = '../datasets/original/train_qwen.jsonl'

process_jsonl(input_file, output_file)