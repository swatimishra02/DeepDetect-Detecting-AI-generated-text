import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from preprocessing.preprocessor import split_and_preprocess


from utils.logger import setup_logger  

logger = setup_logger()

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text, remove_stopwords=True):
    """
    Cleans the input text by:
    - Lowercasing
    - Removing special characters & punctuation
    - Removing extra spaces
    - Removing stopwords (optional)
    """
    if not isinstance(text, str):
        return ""  

    text = text.lower()
    text = re.sub(r"\d+", "", text)  
    text = text.translate(str.maketrans("", "", string.punctuation))  
    text = re.sub(r"\s+", " ", text).strip()  

    if remove_stopwords:
        text = " ".join([word for word in text.split() if word not in STOPWORDS])

    return text

def remove_leakage(source_df, target_df, threshold=0.9):
    """
    Removes leaked samples from a target dataset if they are too similar to a source dataset.

    Args:
        source_df (pd.DataFrame): The dataset to check against (e.g., Train)
        target_df (pd.DataFrame): The dataset to clean (e.g., Test/Validation)
        threshold (float): Cosine similarity threshold to flag leakage

    Returns:
        pd.DataFrame: Cleaned target dataset
    """
    source_texts = source_df["cleaned_text"].tolist()
    target_texts = target_df["cleaned_text"].tolist()

    vectorizer = TfidfVectorizer()
    source_vectors = vectorizer.fit_transform(source_texts)
    target_vectors = vectorizer.transform(target_texts)

    similarity_matrix = cosine_similarity(target_vectors, source_vectors)
    max_similarities = similarity_matrix.max(axis=1)

    non_leaked_indices = [i for i, sim in enumerate(max_similarities) if sim < threshold]
    cleaned_target_df = target_df.iloc[non_leaked_indices].reset_index(drop=True)

    leaked_count = len(target_df) - len(cleaned_target_df)
    if leaked_count > 0:
        logger.warning(f"{leaked_count} samples removed due to data leakage.")
    else:
        logger.info("No data leakage detected.")

    return cleaned_target_df

def split_and_preprocess(raw_csv, output_dir, val_split=True):
    """
    Splits dataset into train-test (70-30), optionally train-val (80-20), and preprocesses text.
    Also removes duplicates and prevents data leakage.

    Args:
        raw_csv (str): Path to raw dataset
        output_dir (str): Directory to save processed datasets
        val_split (bool): Whether to create a validation split from training data
    """
    logger.info(f"Loading dataset from {raw_csv}...")
    df = pd.read_csv(raw_csv)
    
    if "text" not in df.columns or "generated" not in df.columns:
        logger.error("Dataset must contain 'text' and 'generated' columns")
        raise ValueError("Dataset must contain 'text' and 'generated' columns")

    logger.info(f"Initial dataset size: {len(df)} samples")

    # Remove exact duplicates
    df.drop_duplicates(subset=["text"], inplace=True)
    logger.info(f"Dataset size after removing duplicates: {len(df)} samples")

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["generated"])

    if val_split:
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["generated"])
    
    logger.info(f"Train size: {len(train_df)} samples")
    logger.info(f"Test size (before leakage check): {len(test_df)} samples")
    if val_split:
        logger.info(f"Validation size (before leakage check): {len(val_df)} samples")

    train_df["cleaned_text"] = train_df["text"].apply(lambda x: clean_text(x))
    test_df["cleaned_text"] = test_df["text"].apply(lambda x: clean_text(x))
    if val_split:
        val_df["cleaned_text"] = val_df["text"].apply(lambda x: clean_text(x))

    test_df = remove_leakage(train_df, test_df)
    if val_split:
        val_df = remove_leakage(train_df, val_df)
        val_df = remove_leakage(test_df, val_df)  

    logger.info(f"Final Train size: {len(train_df)} samples")
    logger.info(f"Final Test size: {len(test_df)} samples")
    if val_split:
        logger.info(f"Final Validation size: {len(val_df)} samples")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    if val_split:
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        logger.info(f"Validation set saved: {output_dir}/val.csv")

    logger.info(f"Train set saved: {output_dir}/train.csv")
    logger.info(f"Test set saved: {output_dir}/test.csv")

if __name__ == "__main__":
    split_and_preprocess("data/Training_Essay_Data.csv", "data/processed", val_split=True)
