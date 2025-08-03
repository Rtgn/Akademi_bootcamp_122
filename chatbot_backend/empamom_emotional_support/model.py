import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer 
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset


if torch.cuda.is_available():
    print("✅ CUDA available:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    raise SystemExit("❌ CUDA is not available. Fine-tuning requires a GPU.")


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

#Model ve tokenizer yükleme
model_id = "stabilityai/stablelm-2-zephyr-1_6b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
                                                                                                        
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="chatbot_emotion_examples.jsonl", split="train")

def tokenize(example):
    prompt = f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}{tokenizer.eos_token}"
    tokenized = tokenizer(prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return {k: v.squeeze(0) for k, v in tokenized.items()}

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./stablelm-2-zephyr-1_6b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    optim="adamw_8bit",
    warmup_steps=100,
    max_grad_norm=0.3,
    remove_unused_columns=False,
    report_to="none"
)

#Modeli train etme
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

print("eĞİTİM başladı")
trainer.train()
print("eĞİTİM TAMAMLANDI")

model.save_pretrained("stablelm-2-zephyr-1_6b")
tokenizer.save_pretrained("stablelm-2-zephyr-1_6b")

print("Model ve tokenizer kaydedildi")