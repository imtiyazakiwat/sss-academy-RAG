"""
Script to create the complete standalone Kaggle Jupyter Notebook.

Targets: Kaggle Tesla P100 GPU (sm_60), Python 3.12
Key fix: Kaggle's pre-installed PyTorch 2.10+ dropped P100 support.
We downgrade to PyTorch 2.2.0+cu118 which still has sm_60 kernels.
"""

import nbformat as nbf
import json

def create_kaggle_notebook():
    nb = nbf.v4.new_notebook()

    with open("kaggle_pipeline/train_data.jsonl", "r", encoding="utf-8") as f:
        dataset_lines = [json.loads(line) for line in f]

    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell("""# 🚀 ETL Test Engineer LLM Fine-Tuning on Kaggle GPU
### Fine-Tuning Qwen2.5-Coder-3B on Tesla P100 with LoRA
- **Base Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Method**: FP16 LoRA (no quantization needed for 3B model on 16GB P100)
- **Domain**: HCL Technologies / Menards Retail DWH ETL Interview
"""))

    # Cell 1: Downgrade PyTorch to support P100 (sm_60) + install deps
    cells.append(nbf.v4.new_code_cell("""%%capture
# ============================================================
# STEP 1: Fix GPU compatibility
# Kaggle's default PyTorch 2.10+ dropped P100 (sm_60) support.
# We must downgrade to PyTorch 2.2.0 with CUDA 11.8 which
# still includes sm_60 kernels for the Tesla P100.
# ============================================================
!pip uninstall -y torch torchvision torchaudio torchao
!pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# STEP 2: Install fine-tuning libraries (pinned for torch 2.2 compat)
!pip install -q "transformers==4.44.2" "peft==0.13.2" "trl==0.11.4" "accelerate==0.34.2" "datasets==3.0.1"
"""))

    # Cell 2: Verify GPU environment
    cells.append(nbf.v4.new_code_cell("""import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Supported Archs: {torch.cuda.get_arch_list()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name} (sm_{cap[0]}{cap[1]})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    assert "sm_60" in torch.cuda.get_arch_list(), "ERROR: sm_60 not in supported archs!"
    print("✅ P100 (sm_60) is supported by this PyTorch build!")
else:
    raise RuntimeError("No GPU detected!")
"""))

    # Cell 3: Load Model & Tokenizer in FP16
    cells.append(nbf.v4.new_code_cell("""from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✅ Model loaded with LoRA adapters!")
"""))

    # Cell 4: Dataset Preparation
    cells.append(nbf.v4.new_code_cell(f"""import json
from datasets import Dataset

raw_data = {json.dumps(dataset_lines, ensure_ascii=False)}

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {{ "text" : texts }}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"✅ Formatted {{len(dataset)}} training samples!")
"""))

    # Cell 5: Train
    cells.append(nbf.v4.new_code_cell("""from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    save_strategy="no",
    report_to="none",
    gradient_checkpointing=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("🎉 Fine-tuning complete!")
"""))

    # Cell 6: Test inference
    cells.append(nbf.v4.new_code_cell("""prompt_text = \"\"\"<|im_start|>system
You are an experienced Senior ETL Test Engineer in a technical interview. You have 4.2 years of experience at HCL Technologies working on the Menards retail data warehouse project. Speak naturally, authoritatively, and concisely in the 1st person. Ground your answers in real tools (Oracle, Informatica, TOAD, HP ALM, Unix) and practical SQL validation techniques (MINUS queries, duplicate checks, SCD2 history tracking, count reconciliation). Never use robotic bullet lists or markdown headers.<|im_end|>
<|im_start|>user
What is the difference between ER modeling and Dimensional modeling?<|im_end|>
<|im_start|>assistant
\"\"\"

inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True)
print("\\n--- Model Output ---")
_ = model.generate(**inputs, streamer=streamer, max_new_tokens=400, temperature=0.7, do_sample=True)
"""))

    # Cell 7: Save
    cells.append(nbf.v4.new_code_cell("""model.save_pretrained("etl_interview_lora")
tokenizer.save_pretrained("etl_interview_lora")
print("✅ LoRA adapter saved!")
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

    print("✅ Notebook created successfully.")

if __name__ == "__main__":
    create_kaggle_notebook()
