import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_text_leakage(train_csv, test_csv, val_csv=None, threshold=0.9):
    """
    Checks for text leakage between train, validation, and test datasets using cosine similarity.

    Args:
        train_csv (str): Path to preprocessed train dataset
        test_csv (str): Path to preprocessed test dataset
        val_csv (str, optional): Path to validation dataset (if applicable)
        threshold (float): Similarity threshold to flag potential leakage

    Returns:
        None (prints leaked samples count and examples)
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv) if val_csv else None

    if "cleaned_text" not in train_df.columns or "cleaned_text" not in test_df.columns:
        raise ValueError("Datasets must contain 'cleaned_text' column")

    train_df.dropna(subset=["cleaned_text"], inplace=True)
    test_df.dropna(subset=["cleaned_text"], inplace=True)
    if val_df is not None:
        val_df.dropna(subset=["cleaned_text"], inplace=True)

    train_texts = train_df["cleaned_text"].astype(str).tolist()
    test_texts = test_df["cleaned_text"].astype(str).tolist()
    val_texts = val_df["cleaned_text"].astype(str).tolist() if val_df is not None else []

    def detect_leakage(source_texts, target_texts, source_name, target_name):
        """
        Compares two datasets for possible text leakage using cosine similarity.
        """
        if not source_texts or not target_texts:
            print(f"Skipping {source_name}-{target_name} check (empty dataset)")
            return

        vectorizer = TfidfVectorizer()
        source_vectors = vectorizer.fit_transform(source_texts)
        target_vectors = vectorizer.transform(target_texts)

        similarity_matrix = cosine_similarity(target_vectors, source_vectors)
        max_similarities = similarity_matrix.max(axis=1)

        leaked_indices = [i for i, sim in enumerate(max_similarities) if sim > threshold]

        if leaked_indices:
            print(f"WARNING: {len(leaked_indices)} samples in {target_name} leaked from {source_name}!")

            print(f"\nExamples of leaked samples from {target_name}:")
            for idx in leaked_indices[:5]:  
                print(f"\n{target_texts[idx][:300]}...\n")
        else:
            print(f"No data leakage detected between {source_name} and {target_name}.")

    detect_leakage(train_texts, test_texts, "Train", "Test")

    if val_df is not None:
        detect_leakage(train_texts, val_texts, "Train", "Validation")
        detect_leakage(val_texts, test_texts, "Validation", "Test")

if __name__ == "__main__":
    check_text_leakage(
        "data/processed/train.csv",
        "data/processed/test.csv",
        "data/processed/val.csv"
    )
