import logging
import os

def setup_logger(log_dir="logs", log_file="preprocessing.log"):
    """
    Sets up a logger to record preprocessing steps.

    Args:
        log_dir (str): Directory where logs will beimport os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from utils.logger import setup_logger

# Setup Logger
logger = setup_logger(log_dir="logs", log_file="xlnet_training.log")
logger.info("Starting XLNet training script...")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Paths
feature_dir = "features"
label_dir = "data/processed"
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(model_name)

# Function to load data
def load_data(split):
    features = np.load(os.path.join(feature_dir, f"{split}_features.npy"))
    df = pd.read_csv(os.path.join(label_dir, f"{split}.csv"))
    labels = df["generated"].values

    if len(labels) != features.shape[0]:
        logger.warning(f"⚠️ Mismatch in {split}: Truncating labels ({len(labels)}) to match features ({features.shape[0]})")
        labels = labels[:features.shape[0]]

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Dataset class
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load data
X_train, y_train = load_data("train")
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=16)
test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=16)

logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Initialize XLNet model
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

logger.info("XLNet model initialized.")

# Custom TQDM logging class
class TqdmToLogger(tqdm):
    def __init__(self, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger

    def display(self, msg=None, pos=None):
        if msg:
            self.logger.info(msg)

# Training setup
num_epochs = 100
best_val_loss = float("inf")
early_stopping_patience = 10
no_improve_epochs = 0

logger.info(f"Starting XLNet training for {num_epochs} epochs...")

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss, train_correct = 0, 0
    loop = TqdmToLogger(logger, train_loader, desc=f"Epoch {epoch}/{num_epochs} Training")

    for x_batch, y_batch in loop:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch).logits
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == y_batch).sum().item()

        loop.set_postfix(loss=loss.item())

    train_acc = train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).logits
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    val_loss /= len(val_loader)

    # Learning rate scheduling
    scheduler.step(val_loss)

    logger.info(f"Epoch {epoch}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), "best_xlnet_model.pth")  # Save best model
        logger.info(f"🔹 New best model saved at epoch {epoch}")
    else:
        no_improve_epochs += 1
        logger.info(f"🔸 No improvement in validation loss for {no_improve_epochs} epochs.")

    if no_improve_epochs >= early_stopping_patience:
        logger.info("🚀 Early stopping triggered. Training stopped.")
        break

# Load best model before evaluation
model.load_state_dict(torch.load("best_xlnet_model.pth"))
logger.info("Loaded best XLNet model for final evaluation.")

# Final Evaluation on Test Set
model.eval()
all_preds = []

with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch).logits
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

test_acc = accuracy_score(y_test[:len(all_preds)], all_preds)
logger.info(f"✅ Final Test Accuracy: {test_acc:.4f}")
print(f"\nTest Accuracy: {test_acc:.4f}")
stored.
        log_file (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Logs to console as well
        ],
    )

    return logging.getLogger(__name__)
