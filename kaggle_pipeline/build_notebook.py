"""
Build a self-contained Kaggle notebook for fine-tuning on Tesla P100 (16GB).

APPROACH: Pure FP16 LoRA — NO bitsandbytes, NO torchao, NO triton dependency.

Failures analyzed (versions 4-9):
  V4-V6: PyTorch 2.10 dropped P100 sm_60 → fixed with torch 2.2.0+cu118
  V7: gradient_checkpointing + PEFT requires enable_input_require_grads()
  V8: 3B FP16 OOM at batch_size=2, max_seq_length=2048
  V9: bitsandbytes 0.43 crashes with 'No module named triton.ops'

Solution: 1.5B model in pure FP16 LoRA. No quantization libraries needed.
  - 1.5B FP16 = ~3GB VRAM (fits easily in 16GB P100)
  - batch_size=1, max_seq_length=512, gradient_checkpointing
  - adamw_torch optimizer (zero external deps)
  - Pinned: transformers==4.44.2, peft==0.13.2, trl==0.11.4
"""

import nbformat as nbf
import json


def create_kaggle_notebook():
    nb = nbf.v4.new_notebook()

    with open("kaggle_pipeline/train_data.jsonl", "r", encoding="utf-8") as f:
        dataset_lines = [json.loads(line) for line in f]

    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 🚀 ETL Interview LLM Fine-Tuning (Kaggle P100)
- **Model**: Qwen/Qwen2.5-1.5B-Instruct (FP16 LoRA)
- **GPU**: Tesla P100 16GB
- **Stack**: transformers + peft + trl (NO bitsandbytes)
"""))

    # ── Cell 1: Fix PyTorch + install deps ───────────────────────
    cells.append(nbf.v4.new_code_cell("""import subprocess, sys

# Step 1: Remove incompatible packages
for pkg in ["torch", "torchvision", "torchaudio", "torchao"]:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg],
                   capture_output=True)

# Step 2: Install PyTorch 2.2.0 with CUDA 11.8 (supports P100 sm_60)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.2.0+cu118", "torchvision==0.17.0+cu118", "torchaudio==2.2.0+cu118",
    "--index-url", "https://download.pytorch.org/whl/cu118"
], check=True)

# Step 3: Install fine-tuning stack (NO bitsandbytes)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.44.2", "peft==0.13.2", "trl==0.11.4",
    "accelerate==0.34.2", "datasets==3.0.1"
], check=True)

print("✅ Packages installed")
"""))

    # ── Cell 2: Verify GPU ───────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""import torch
print(f"PyTorch: {torch.__version__}")
assert torch.cuda.is_available(), "No GPU!"
archs = torch.cuda.get_arch_list()
assert "sm_60" in archs, f"sm_60 not in {archs}"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name} | sm_{cap[0]}{cap[1]} | {mem_gb:.1f} GB")
print("✅ GPU OK")
"""))

    # ── Cell 3: Load model in FP16 ───────────────────────────────
    cells.append(nbf.v4.new_code_cell("""from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Enable gradient checkpointing BEFORE wrapping with PEFT
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
used = torch.cuda.memory_allocated(0) / 1e9
print(f"✅ Model loaded | VRAM: {used:.1f} GB / {mem_gb:.1f} GB")
"""))

    # ── Cell 4: Prepare dataset ──────────────────────────────────
    cells.append(nbf.v4.new_code_cell(f"""import json
from datasets import Dataset

raw_data = {json.dumps(dataset_lines, ensure_ascii=False)}

def format_chat(examples):
    texts = []
    for convo in examples["messages"]:
        texts.append(tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False))
    return {{"text": texts}}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_chat, batched=True)
print(f"✅ {{len(dataset)}} training samples")
"""))

    # ── Cell 5: Train ────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""from trl import SFTTrainer
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_ratio=0.05,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    save_strategy="no",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=args,
)

print("🚀 Training...")
trainer.train()
print("🎉 Done!")
"""))

    # ── Cell 6: Inference test ───────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""prompt = \"\"\"<|im_start|>system
You are a Senior ETL Test Engineer with 4.2 years at HCL Technologies on the Menards retail data warehouse. Answer naturally and concisely.<|im_end|>
<|im_start|>user
What is the difference between ER modeling and Dimensional modeling?<|im_end|>
<|im_start|>assistant
\"\"\"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
"""))

    # ── Cell 7: Save ─────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""model.save_pretrained("etl_lora")
tokenizer.save_pretrained("etl_lora")
print("✅ Saved to etl_lora/")
"""))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12",
        },
    }

    with open("kaggle_pipeline/ETL_Interview_FineTuning.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("✅ Notebook created")


if __name__ == "__main__":
    create_kaggle_notebook()
