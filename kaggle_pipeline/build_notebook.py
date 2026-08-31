"""
Script to create the complete standalone Kaggle Jupyter Notebook.

Targets: Kaggle Tesla T4 GPU (sm_75), Python 3.12, PyTorch 2.x
Fixes applied:
  - Uninstalls incompatible pre-installed torchao before importing transformers
  - Uses adamw_torch optimizer (no bitsandbytes dependency)
  - Uses FP16 LoRA (no 4-bit quantization, no bitsandbytes)
  - Requests T4 via kernel-metadata.json accelerator field
"""

import nbformat as nbf
import json

def create_kaggle_notebook():
    nb = nbf.v4.new_notebook()

    # Load dataset lines to embed in notebook for 100% self-contained execution
    with open("kaggle_pipeline/train_data.jsonl", "r", encoding="utf-8") as f:
        dataset_lines = [json.loads(line) for line in f]

    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell("""# 🚀 ETL Test Engineer LLM Fine-Tuning on Kaggle GPU
### High-Accuracy Model Training on Cloud Tesla T4 GPU
- **Base Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Fine-Tuning**: FP16 LoRA with Hugging Face `peft` + `trl` (SFTTrainer)
- **Domain**: Grounded in HCL Technologies / Menards Retail DWH ETL Interview Experience
"""))

    # Cell 1: Fix environment + install deps
    cells.append(nbf.v4.new_code_cell("""%%capture
# CRITICAL: Remove pre-installed torchao which conflicts with transformers
# Kaggle ships torchao 0.10.0 but transformers requires >= 0.16.0
!pip uninstall -y torchao

# Install the fine-tuning stack (no bitsandbytes needed for FP16 LoRA)
!pip install -q "transformers>=4.45.0" "peft>=0.13.0" "trl>=0.11.0" "accelerate>=0.34.0" datasets
"""))

    # Cell 2: Verify GPU + environment
    cells.append(nbf.v4.new_code_cell("""import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    raise RuntimeError("No GPU detected! Check Kaggle accelerator settings.")
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

# LoRA config targeting all attention + MLP projections
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
print("✅ Model loaded in FP16 with LoRA adapters!")
"""))

    # Cell 4: Dataset Preparation
    cells.append(nbf.v4.new_code_cell(f"""import json
from datasets import Dataset

# Embedded gold-standard training data ({len(dataset_lines)} samples)
raw_data = {json.dumps(dataset_lines, ensure_ascii=False)}

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {{ "text" : texts }}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(formatting_prompts_func, batched = True)
print(f"✅ Formatted {{len(dataset)}} conversational training samples!")
"""))

    # Cell 5: Train with SFTTrainer
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

print("🚀 Starting fine-tuning on Cloud GPU...")
trainer.train()
print("🎉 Fine-tuning completed successfully!")
"""))

    # Cell 6: Test & Verify Accuracy
    cells.append(nbf.v4.new_code_cell("""# 🧪 Run Live Inference Test (ER vs Dimensional Modeling)
prompt_text = \"\"\"<|im_start|>system
You are an experienced Senior ETL Test Engineer in a technical interview. You have 4.2 years of experience at HCL Technologies working on the Menards retail data warehouse project. Speak naturally, authoritatively, and concisely in the 1st person. Ground your answers in real tools (Oracle, Informatica, TOAD, HP ALM, Unix) and practical SQL validation techniques (MINUS queries, duplicate checks, SCD2 history tracking, count reconciliation). Never use robotic bullet lists or markdown headers.<|im_end|>
<|im_start|>user
What is the difference between ER modeling and Dimensional modeling?<|im_end|>
<|im_start|>assistant
\"\"\"

inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True)
print("\\n--- Model Output ---")
_ = model.generate(**inputs, streamer=streamer, max_new_tokens=400, temperature=0.7)
"""))

    # Cell 7: Save and Export
    cells.append(nbf.v4.new_code_cell("""# 💾 Save LoRA Adapter to disk
model.save_pretrained("etl_interview_qwen3b_lora")
tokenizer.save_pretrained("etl_interview_qwen3b_lora")
print("✅ Saved LoRA adapter to etl_interview_qwen3b_lora/!")
"""))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
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

    print("Successfully created kaggle_pipeline/ETL_Interview_FineTuning.ipynb")

if __name__ == "__main__":
    create_kaggle_notebook()
