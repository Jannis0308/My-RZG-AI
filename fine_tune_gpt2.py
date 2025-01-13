from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

# 1. Modell und Tokenizer laden
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Padding-Token hinzufügen (falls erforderlich)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# 2. Datensatz laden
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Kein Masked Language Modeling für GPT-2
    )

dataset_path = "dataset.txt"
dataset = load_dataset(dataset_path, tokenizer)

# 3. Trainingseinstellungen
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=torch.cuda.is_available()  # Mixed Precision aktivieren, falls GPU verfügbar
)

# 4. Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator(tokenizer),
    train_dataset=dataset,
)

# 5. Modell trainieren
trainer.train()

# 6. Modell und Tokenizer speichern
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
print("Feintuning abgeschlossen und Modell gespeichert.")
