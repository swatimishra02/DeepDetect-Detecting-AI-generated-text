import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.logger import setup_logger
logger = setup_logger(log_file="training.log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dir = "features"
label_dir = "data/processed"
splits = ["train", "val", "test"]

def load_data(split):
    features = np.load(os.path.join(feature_dir, f"{split}_features.npy"))
    df = pd.read_csv(os.path.join(label_dir, f"{split}.csv"))
    labels = df["generated"].values

    if len(labels) != features.shape[0]:
        logger.warning(f"Mismatch in {split}: Truncating labels ({len(labels)}) to match features ({features.shape[0]})")
        labels = labels[:features.shape[0]]

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

X_train, y_train = load_data("train")
X_val, y_val = load_data("val")
X_test, y_test = load_data("test")

train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=16)
test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=16)

class FeatureClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = FeatureClassifier(input_dim=X_train.shape[1]).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

num_epochs = 100
best_val_loss = float("inf")
early_stopping_patience = 10
no_improve_epochs = 0

logger.info("Starting training...")

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss, train_correct = 0, 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Training", leave=False)

    for x_batch, y_batch in train_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == y_batch).sum().item()

        train_bar.set_postfix(loss=loss.item())

    train_acc = train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss, val_correct = 0, 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} Validation", leave=False)
    with torch.no_grad():
        for x_batch, y_batch in val_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()

            val_bar.set_postfix(loss=loss.item())

    val_acc = val_correct / len(val_loader.dataset)
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step(val_loss)

    msg = (
        f"Epoch {epoch}/{num_epochs}: "
        f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | "
        f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}"
    )
    logger.info(msg)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), "best_model.pth")
        logger.info(f"Best model saved at epoch {epoch}")
    else:
        no_improve_epochs += 1
        logger.info(f"No improvement for {no_improve_epochs} epoch(s)")

    if no_improve_epochs >= early_stopping_patience:
        logger.info("Early stopping triggered. Training stopped.")
        break

model.load_state_dict(torch.load("best_model.pth"))

model.eval()
all_preds = []

with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

test_acc = accuracy_score(y_test[:len(all_preds)], all_preds)
logger.info(f"\nTest Accuracy: {test_acc:.4f}")

np.savez("logs/metrics.npz", train_loss=train_losses, val_loss=val_losses, train_acc=train_accuracies, val_acc=val_accuracies)
