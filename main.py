import os
import logging
from utils.logger import setup_logger
from feature_extractor import process_dataset
from model_trainer import train_model, evaluate_model, ensemble_models
from preprocessing.preprocessor import split_and_preprocess

DATA_PATH = "data/processed"
FEATURE_PATH = "features"
LOG_PATH = "logs/training.log"
EPOCHS = 100
FOLDS = 5

def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = setup_logger(LOG_PATH)
    logger.info("Starting DeepDetect Pipeline")

    logger.info("Preprocessing data...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = preprocess_data(DATA_PATH)
    logger.info("Preprocessing complete.")

    logger.info("Extracting features...")
    for split in ["train", "val", "test"]:
        process_dataset(split)
    logger.info("Feature extraction complete.")

    logger.info("Starting model training with cross-validation...")
    train_model(FEATURE_PATH, train_labels, val_labels, epochs=EPOCHS, folds=FOLDS, logger=logger)
    logger.info("Model training complete.")

    logger.info("Evaluating models on test set...")
    predictions, metrics = evaluate_model(FEATURE_PATH, test_labels, logger=logger)
    logger.info(f"Evaluation Results: {metrics}")

    logger.info("Ensembling predictions...")
    final_preds = ensemble_models(predictions)
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
