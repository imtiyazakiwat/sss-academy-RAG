"""
Build a self-contained Kaggle notebook for fine-tuning on Tesla P100 (16GB).

Environment constraints (discovered through 8 failed versions):
  - Kaggle assigns P100 GPU (sm_60, Pascal architecture)
  - Kaggle's default PyTorch 2.10+ dropped sm_60 support
  - Kaggle's pre-installed torchao 0.10.0 conflicts with latest transformers
  - P100 has 16GB VRAM: 3B FP16 model OOMs with batch_size>=2 at seq_len=2048
  - gradient_checkpointing + PEFT requires enable_input_require_grads()

Solution:
  - Downgrade PyTorch to 2.2.0+cu118 (has sm_60 kernels)
  - Use Qwen2.5-1.5B-Instruct in 4-bit QLoRA (~1GB model VRAM)
  - Pass peft_config to SFTTrainer (it handles prepare_model internally)
  - batch_size=1, gradient_accumulation=8, gradient_checkpointing=True
  - paged_adamw_8bit optimizer
"""

import nbformat as nbf
import json


def create_kaggle_notebook():
    nb = nbf.v4.new_notebook()

    with open("kaggle_pipeline/train_data.jsonl", "r", encoding="utf-8") as f:
        dataset_lines = [json.loads(line) for line in f]

    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 🚀 ETL Interview LLM Fine-Tuning (Kaggle P100 GPU)
- **Model**: Qwen/Qwen2.5-1.5B-Instruct (4-bit QLoRA)
- **GPU**: Tesla P100 16GB
- **Method**: QLoRA via bitsandbytes + SFTTrainer
"""))

    # ── Cell 1: Fix environment ──────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""%%capture
# ================================================================
# ENVIRONMENT FIX (required for Kaggle P100 / sm_60)
#
# Problem: Kaggle ships PyTorch 2.10+ which dropped P100 support.
# Fix: Downgrade to PyTorch 2.2.0 + CUDA 11.8 (has sm_60 kernels).
# Also remove conflicting torchao package.
# ================================================================
!pip uninstall -y torch torchvision torchaudio torchao -q
!pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118 -q

# Install fine-tuning stack (pinned versions tested with torch 2.2 + cu118)
!pip install "transformers==4.44.2" "peft==0.13.2" "trl==0.11.4" "bitsandbytes==0.43.1" "accelerate==0.34.2" "datasets==3.0.1" -q
"""))

    # ── Cell 2: Verify GPU ───────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
archs = torch.cuda.get_arch_list()
print(f"Archs: {archs}")
assert "sm_60" in archs, f"sm_60 missing from {archs}"
gpu = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} | Capability: {cap[0]}.{cap[1]} | VRAM: {mem:.1f} GB")
print("✅ Environment OK")
"""))

    # ── Cell 3: Load model in 4-bit QLoRA ────────────────────────
    cells.append(nbf.v4.new_code_cell("""from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Required for gradient checkpointing with frozen base model
model.config.use_cache = False
model.enable_input_require_grads()

print(f"✅ {model_id} loaded in 4-bit QLoRA")
print(f"   VRAM used: {torch.cuda.memory_allocated(0)/1e9:.1f} GB / {mem:.1f} GB")
"""))

    # ── Cell 4: Prepare dataset ──────────────────────────────────
    cells.append(nbf.v4.new_code_cell(f"""import json
from datasets import Dataset

raw_data = {json.dumps(dataset_lines, ensure_ascii=False)}

def format_chat(examples):
    texts = []
    for convo in examples["messages"]:
        texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
    return {{"text": texts}}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_chat, batched=True)
print(f"✅ {{len(dataset)}} training samples ready")
"""))

    # ── Cell 5: Train ────────────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    warmup_ratio=0.05,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    save_strategy="no",
    report_to="none",
)

# Let SFTTrainer handle PEFT wrapping internally (safer than manual get_peft_model)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args,
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("🎉 Fine-tuning complete!")
"""))

    # ── Cell 6: Test inference ───────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""prompt = \"\"\"<|im_start|>system
You are an experienced Senior ETL Test Engineer in a technical interview. You have 4.2 years of experience at HCL Technologies working on the Menards retail data warehouse project. Speak naturally and concisely.<|im_end|>
<|im_start|>user
What is the difference between ER modeling and Dimensional modeling?<|im_end|>
<|im_start|>assistant
\"\"\"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
print("--- Output ---")
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
"""))

    # ── Cell 7: Save adapter ────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""trainer.save_model("etl_interview_lora")
tokenizer.save_pretrained("etl_interview_lora")
print("✅ LoRA adapter saved to etl_interview_lora/")
"""))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    }

    with open("kaggle_pipeline/ETL_Interview_FineTuning.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("✅ Notebook created")


if __name__ == "__main__":
    create_kaggle_notebook()
