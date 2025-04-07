import os
import pandas as pd
import numpy as np
import torch
from transformers import XLNetTokenizer, XLNetModel, pipeline
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger(log_dir="logs", log_file="sample_feature_extraction.log")

data_dir = "data/processed/"
feature_dir = "features2/"
os.makedirs(feature_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model = XLNetModel.from_pretrained("xlnet-base-cased").to(device)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

SAMPLES = {"train": 50, "val": 10, "test": 30}

def extract_xlnet_features(texts):
    """Extract XLNet embeddings for a batch of texts."""
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = xlnet_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  

def extract_zero_shot_features(texts):
    """Extract zero-shot classification scores for a batch of texts."""
    labels = ["human-written", "AI-generated"]
    scores = []
    for text in texts:
        result = zero_shot_classifier(text, labels)
        scores.append(result["scores"])
    return np.array(scores)  

def process_sample(split, num_rows):
    """Process a sample of train, val, test datasets."""
    input_file = os.path.join(data_dir, f"{split}.csv")
    output_file = os.path.join(feature_dir, f"{split}_sample_features.npy")

    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return
    
    df = pd.read_csv(input_file).head(num_rows)  
    texts = df["cleaned_text"].fillna(" ").tolist()
    
    try:
        xlnet_feat = extract_xlnet_features(texts)  
        zero_shot_feat = extract_zero_shot_features(texts)  
        features = np.hstack((xlnet_feat, zero_shot_feat))  

        np.save(output_file, features)
        logger.info(f"Processed {num_rows} rows from {split}. Features saved to {output_file}")

    except Exception as e:
        logger.error(f"Error processing {split}: {e}")

if __name__ == "__main__":
    for split, num_rows in SAMPLES.items():
        logger.info(f"Extracting sample features from {split} dataset...")
        process_sample(split, num_rows)
    logger.info("Sample feature extraction complete!")
