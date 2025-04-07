import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch

metrics = np.load("logs/metrics.npz")
train_loss = metrics["train_loss"]
val_loss = metrics["val_loss"]
train_acc = metrics["train_acc"]
val_acc = metrics["val_acc"]

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss', marker='o')
plt.plot(val_loss, label='Val Loss', marker='o')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/loss_curve.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Train Accuracy', marker='o')
plt.plot(val_acc, label='Val Accuracy', marker='o')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/accuracy_curve.png")
plt.close()

X_test = torch.tensor(np.load("features/test_features.npy"), dtype=torch.float32)
df = pd.read_csv("data/processed/test.csv")
y_test = df["generated"].values
from model_trainer import FeatureClassifier, device  

model = FeatureClassifier(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

all_preds = []
with torch.no_grad():
    for i in range(0, len(X_test), 16):
        x_batch = X_test[i:i+16].to(device)
        preds = model(x_batch).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

cm = confusion_matrix(y_test[:len(all_preds)], all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig("logs/confusion_matrix.png")
plt.close()
