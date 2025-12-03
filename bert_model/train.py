import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Step 0: Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load dataset
df = pd.read_csv("/kaggle/input/multidomain-query-intent-classification/multidomain_query_dataset_5800.csv")

# Step 2: Encode intent labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["intent"])
num_labels = len(label_encoder.classes_)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Step 3: Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.1, random_state=42
)

# Step 4: Tokenize text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_texts, train_labels)
val_dataset = IntentDataset(val_texts, val_labels)

# Step 5: Load BERT model with classification head
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels
).to(device)

# Step 6: Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_intent_model",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=50,
    do_eval=True,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2
)

# Step 7: Evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# Step 8: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Step 9: Train the model
trainer.train()

# Step 10: Save final model + tokenizer
trainer.save_model("./bert_intent_model")
tokenizer.save_pretrained("./bert_intent_model")

# Step 11: Plot loss graph
logs = trainer.state.log_history
train_loss = [x['loss'] for x in logs if 'loss' in x and 'epoch' in x]
eval_loss = [x['eval_loss'] for x in logs if 'eval_loss' in x]
epochs_train = [x['epoch'] for x in logs if 'loss' in x and 'epoch' in x]
epochs_eval = [x['epoch'] for x in logs if 'eval_loss' in x]

plt.figure(figsize=(10, 6))
plt.plot(epochs_train, train_loss, label="Training Loss")
plt.plot(epochs_eval, eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()
