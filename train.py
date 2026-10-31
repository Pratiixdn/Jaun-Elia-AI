import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def main():
    print("1. Loading dataset...")

    file_path = "data/jaun_elia_data.txt"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file at '{file_path}'. Please check your data folder.")

    dataset = load_dataset("text", data_files={"train": file_path})

    print("2. Loading base model and tokenizer...")
    model_id = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("3. Tokenizing poetry data...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    model = AutoModelForCausalLM.from_pretrained(model_id)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    print("4. Setting up training parameters...")
    training_args = TrainingArguments(
        output_dir="./jaun_elia_model",
        per_device_train_batch_size=2,
        num_train_epochs=20,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="no",
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    print("5. Training model on Jaun Elia's poetry...")
    trainer.train()

    os.makedirs("./jaun_elia_model", exist_ok=True)
    trainer.save_model("./jaun_elia_model")
    tokenizer.save_pretrained("./jaun_elia_model")
    print("\nTraining complete! Fine-tuned AI saved in './jaun_elia_model'.")

if __name__ == "__main__":
    main()