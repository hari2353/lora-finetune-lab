# LoRA Fine-tune on Colab T4 (free tier)

## 1. Runtime
Runtime → Change runtime type → **T4 GPU**.

## 2. Install
```python
!pip install -q torch transformers peft trl datasets
```

## 3. Generate the dataset (from this repo)
```python
import sys, json
sys.path.insert(0, "/content/lora-finetune-lab/src")
!git clone https://github.com/hari2353/lora-finetune-lab.git /content/lora-finetune-lab

from finetune_lab.data import generate_pairs, to_instruction_format

ID_TO_NAME = {
    "esco__001": "Machine Learning", "esco__004": "Kubernetes", "esco__005": "Apache Spark",
    "esco__007": "Python", "esco__011": "Retrieval-Augmented Generation", "esco__016": "AWS",
}
pairs = generate_pairs(n_per_skill=40, seed=42)
records = to_instruction_format(pairs, ID_TO_NAME)
with open("train.jsonl", "w") as fh:
    for r in records:
        fh.write(json.dumps(r) + "\n")
```

## 4. Emit and run the training script
```python
from finetune_lab.train_lora import build_training_script
script = build_training_script("Qwen/Qwen2.5-0.5B-Instruct", "train.jsonl", "out")
open("run_train.py", "w").write(script)
!python run_train.py
```

## 5. Quick sanity check of the merged adapter
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("out")
model = AutoModelForCausalLM.from_pretrained("out", device_map="auto")
msgs = [{"role": "system", "content": 'Extract skills from the text. Reply JSON: {"skills": [... ]}'},
        {"role": "user", "content": "We need strong kubernetes and aws lambda experience."}]
inputs = tok.apply_chat_template(msgs, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=60)[0], skip_special_tokens=True))
```

Expected: valid `{"skills": ["AWS", "Kubernetes"]}`-style output where the base model often rambles.
