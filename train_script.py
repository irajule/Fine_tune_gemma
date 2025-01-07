import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset
import pandas as pd
from bs4 import BeautifulSoup
import optuna
from datasets import load_metric

# Paths to your dataset files
poems_data_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gemma2/code/Users/j.irabaruta1/poems_data.json"
articles_data_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gemma2/code/Users/j.irabaruta1/cleaned_articlesr.json"

# Helper function to clean HTML content
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Load datasets
poetry_dataset = load_dataset("json", data_files=poems_data_path)
articles_dataset = load_dataset("json", data_files=articles_data_path)

# Preprocess poetry dataset
def preprocess_poem(example):
    text = clean_text(example["texte"])
    metadata = (
        f"Auteur: {example.get('auteur', 'Non spécifié')} | "
        f"Titre: {example.get('title', 'Non spécifié')} | "
        f"Date de publication: {example.get('date_de_publication', 'Non spécifié')} | "
        f"Ère: {example.get('ère', 'Non spécifié')}"
    )
    return {"text": f"{metadata}\n{text}"}

# Preprocess articles dataset
def preprocess_article(example):
    text = clean_text(example["texte"])
    metadata = (
        f"Titre: {example.get('title', 'Non spécifié')} | "
        f"Sujet: {example.get('sujet', 'Non spécifié')}"
    )
    return {"text": f"{metadata}\n{text}"}

# Apply preprocessing functions
poetry_dataset = poetry_dataset.map(preprocess_poem, remove_columns=["texte", "link", "title", "date_de_publication", "auteur", "ère"])
articles_dataset = articles_dataset.map(preprocess_article, remove_columns=["texte", "link", "title", "sujet"])

# Convert datasets to pandas DataFrames
poetry_df = poetry_dataset["train"].to_pandas()
articles_df = articles_dataset["train"].to_pandas()

# Concatenate the datasets
combined_df = pd.concat([poetry_df, articles_df], ignore_index=True)

# Convert the reduced DataFrame back to a Hugging Face Dataset
combined_dataset = Dataset.from_pandas(combined_df)

# Shuffle the combined dataset
combined_dataset = combined_dataset.shuffle(seed=42)

# Tokenizer and Model
local_model_path = "/home/azureuser/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
model = AutoModelForCausalLM.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Add a padding token if the tokenizer does not have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    # Add labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize the combined dataset
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split into training and evaluation datasets
split_dataset = tokenized_dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Load metrics
rouge_metric = load_metric("rouge")
bleu_metric = load_metric("bleu")

# Custom callback to log evaluation metrics after each epoch
class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", None)
        perplexity = torch.exp(torch.tensor(eval_loss)) if eval_loss else None
        
        print(f"Evaluation metrics after epoch {state.epoch}:")
        print(f"Perplexity: {perplexity}")
        print(f"Other metrics: {metrics}")

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    num_train_epochs = trial.suggest_int("num_train_epochs", 5, 15)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    # Define layer freezing (gradual unfreezing)
    num_layers = len(list(model.parameters()))
    for i, param in enumerate(model.parameters()):
        if i < num_layers // 3 * 2:  # Gradually unfreeze
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Initialize the optimizer with discriminative learning rates
    optimizer_grouped_parameters = [
        {"params": [p for i, p in enumerate(model.parameters()) if p.requires_grad],
         "lr": learning_rate if i < num_layers // 2 else learning_rate * 10}  # Higher LR for final layers
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Define the learning rate scheduler (e.g., StepLR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./trial_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8 // batch_size,
        num_train_epochs=num_train_epochs,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[
            LogMetricsCallback(),
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
        ],
    )

    # Train and evaluate the model
    trainer.train()

    # Evaluation metrics
    metrics = trainer.evaluate()
    print(f"Final Evaluation Metrics: {metrics}")

    return metrics["eval_loss"]

# Run 3 trials with Bayesian Optimization
num_trials = 3
study = optuna.create_study(direction="minimize")
for _ in range(num_trials):
    study.optimize(objective, n_trials=1)

# Save the best model
best_trial = study.best_trial
print(f"Best Trial: {best_trial}")
best_learning_rate = best_trial.params["learning_rate"]
best_epochs = best_trial.params["num_train_epochs"]
best_batch_size = best_trial.params["batch_size"]

# Save best model details
training_args = TrainingArguments(
    output_dir="./best_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=best_batch_size,
    per_device_eval_batch_size=best_batch_size,
    gradient_accumulation_steps=8 // best_batch_size,
    num_train_epochs=best_epochs,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=best_learning_rate,
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[
        LogMetricsCallback(),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    ],
)
trainer.train()
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
