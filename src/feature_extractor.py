import os
import pandas as pd
import numpy as np
import torch
from transformers import XLNetTokenizer, XLNetModel, pipeline
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger

logger = setup_logger(log_dir="logs", log_file="feature_extraction.log")

data_dir = "data/processed/"
feature_dir = "features/"
os.makedirs(feature_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model = XLNetModel.from_pretrained("xlnet-base-cased").to(device)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

BATCH_SIZE = 50

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
def process_dataset(split):
    """Process train, val, test datasets in batches."""
    input_file = os.path.join(data_dir, f"{split}.csv")
    output_file = os.path.join(feature_dir, f"{split}_features.npy")
    
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    texts = df["cleaned_text"].fillna(" ").tolist()
    
    all_features = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]

        try:
            xlnet_feat = extract_xlnet_features(batch_texts)  
            zero_shot_feat = extract_zero_shot_features(batch_texts)  
            batch_features = np.hstack((xlnet_feat, zero_shot_feat))  

            all_features.append(batch_features)

            logger.info(f"Processed batch {i // BATCH_SIZE + 1} of {len(texts) // BATCH_SIZE + 1}")

            if os.path.exists(output_file):
                existing_features = np.load(output_file)
                combined_features = np.vstack((existing_features, batch_features))
                np.save(output_file, combined_features)
            else:
                np.save(output_file, batch_features)

        except Exception as e:
            logger.error(f"Error processing batch {i // BATCH_SIZE + 1}: {e}")

    logger.info(f"Finished processing {split} dataset! Features saved to {output_file}")


if __name__ == "__main__":
    for split in ["test"]:
        logger.info(f"Processing {split} dataset in batches...")
        process_dataset(split)
    logger.info("Feature extraction complete!")
