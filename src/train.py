import mlflow
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def prepare_data(tokenizer, dataset):
    """Tokenize and format the dataset."""
    def tokenize_function(examples):
        # Combine instruction and response with a separator
        text = [f"Instruction: {instr}\nResponse: {resp}" 
               for instr, resp in zip(examples['instruction'], examples['response'])]
        
        return tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding='max_length'
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_datasets

def fine_tune_model():
    """
    Fine-tune GPT-Neo on customer support data using instructions and responses.
    """
    # Load dataset
    dataset = load_dataset('csv', data_files='data/raw/customer_support.csv')
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

    # Load model and tokenizer
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Prepare the dataset
    tokenized_datasets = prepare_data(tokenizer, dataset)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not doing masked language modeling
    )

    mlflow.start_run()

    # Log hyperparameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("learning_rate", 2e-5)

    training_args = TrainingArguments(
        output_dir="models/",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="mlflow"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
    )

    trainer.train()

    # Save the model and tokenizer
    model_path = "models/customer_support_gpt"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Log model artifacts
    mlflow.log_artifact(model_path)

    # Log evaluation metrics
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    mlflow.end_run()

if __name__ == "__main__":
    fine_tune_model()