# Fine-tuning GPT-2 for text generation. Run in a Python environment, not the Python REPL.
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling

texts = [
    "Artificial intelligence enables computers to perform tasks that normally require human intelligence.",
    "Machine learning is a branch of artificial intelligence that learns patterns from data.",
    "Deep learning uses neural networks with multiple layers to learn complex patterns.",
    "Generative AI can create new text, images, audio and other types of content.",
    "Natural language processing enables computers to understand and generate human language.",
    "Large language models are trained on large collections of text data.",
    "Prompt engineering helps users obtain better results from generative AI models.",
    "Generative AI is widely used in education, healthcare, business and research."
]
dataset = Dataset.from_dict({"text": texts})
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(output_dir="./fine_tuned_gpt2", num_train_epochs=3, per_device_train_batch_size=2, learning_rate=5e-5, logging_steps=1, save_strategy="no", report_to="none")
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, data_collator=data_collator)
trainer.train()
prompt = "Artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=80, num_return_sequences=1, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Prompt:"); print(prompt); print("\nGenerated Text:"); print(generated_text)
