# SemanticShield

<h2>Using the Model<h2>

If you want to directly use our model, you can download it from Hugging Face as follows:

```bash
# Make sure Hugging Face CLI is installed
pip install -U "huggingface_hub[cli]"

hf download Luka772001/SS --local-dir ./SemanticShield --local-dir-use-symlinks False

# After downloading the model, you can run the auditing script by navigating into the audit folder and executing the script.

cd audit

python audit_users.py \
  --dataset Clothing \
  --data_dir ./data4llm/Clothing \
  --out_dir ./out/Clothing \
  --model_path ../SemanticShield \
  --device 0


<h2>Requirements</h2>
```
python==3.10
torch==2.4.1
torchvision==0.19.1
transformers==4.53.2
huggingface-hub==0.33.4
tokenizers==0.21.2
safetensors==0.5.3
accelerate==1.9.0
tqdm==4.66.5
```
