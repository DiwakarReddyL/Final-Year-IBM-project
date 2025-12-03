import os
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local model paths
MODEL_PATH = os.path.join(BASE_DIR, 'bert-base-uncased')  # Folder containing config.json, pytorch_model.bin, etc.
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')  # Rename consistently

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# Load label encoder
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

def predict_intent(text):
    """Predicts intent label from text using BERT classifier."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]
