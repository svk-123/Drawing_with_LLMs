import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers.utils import is_torch_bf16_gpu_available
from get_dataset import build_dataset
from config import LOCAL_MODEL_PATH, LORA_PATH, RANK, MAX_SEQ_LENGTH, EPOCHS, TRAIN_DIR, MAX_ITER_STEPS, OUTPUT_PATH

# ----------------------------
# Load model & tokenizer
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",            
)
model.gradient_checkpointing_enable()  # reduce memory usage

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Build datasets
# ----------------------------
train_dataset, test_dataset = build_dataset()

# ----------------------------
# LoRA config
# ----------------------------
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)

# ----------------------------
# SFT config
# ----------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_PATH,
    num_train_epochs=EPOCHS,
    max_steps=MAX_ITER_STEPS,
    per_device_train_batch_size=2,       # lower batch size
    gradient_accumulation_steps=4,       # keep effective batch size
    max_length=min(MAX_SEQ_LENGTH, 2048),  # cap to 1024 tokens if possible

    optim="paged_adamw_8bit",
    learning_rate=5e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
        
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
        
    bf16=is_torch_bf16_gpu_available(),
    fp16=not is_torch_bf16_gpu_available(),
    dataloader_pin_memory=True,
        
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    warmup_steps=5,
    logging_steps=10,
    eval_steps=1000,
    eval_strategy="steps",
    save_strategy="epoch",
    save_total_limit=3,

    report_to="none",        # disable integrations like WandB
    packing=False,
    remove_unused_columns=False,
    dataset_text_field="text",

)

# ----------------------------
# Trainer
# ----------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=lora_config,
    args=sft_config,
)

trainer.train()
trainer.save_model(LORA_PATH + TRAIN_DIR)
