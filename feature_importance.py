import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

feature_dir = "features/"
data_dir = "data/processed/"
feature_file = os.path.join(feature_dir, "val_features.npy")
label_csv = os.path.join(data_dir, "val.csv")

if not os.path.exists(feature_file):
    raise FileNotFoundError(f"Feature file not found at: {feature_file}")
X_val = np.load(feature_file)
print(f"Validation features shape: {X_val.shape}")

if not os.path.exists(label_csv):
    raise FileNotFoundError(f"Label CSV file not found at: {label_csv}")
label_df = pd.read_csv(label_csv)
print(f"Validation labels shape: {label_df.shape}")

if label_df.shape[1] > 1:
    y_val = label_df.iloc[:, -1].values  
else:
    y_val = label_df.iloc[:, 0].values

if X_val.shape[0] != len(y_val):
    print(f"Label count ({len(y_val)}) != Feature count ({X_val.shape[0]}), slicing labels.")
    y_val = y_val[:X_val.shape[0]]
print(f"Final label shape (after alignment): {y_val.shape}")

clf = LogisticRegression(max_iter=1000)
clf.fit(X_val, y_val)
y_pred = clf.predict(X_val)

val_acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {val_acc:.4f}")

print("\nCalculating permutation importance (this may take a moment)...")
importance = permutation_importance(clf, X_val, y_val, n_repeats=10, random_state=42)

importances = importance.importances_mean
top_indices = np.argsort(importances)[::-1][:10]

print("\nTop 10 Important Feature Indices:")
for idx in top_indices:
    print(f"Feature {idx}: Importance = {importances[idx]:.6f}")

np.save(os.path.join(feature_dir, "val_feature_importance.npy"), importances)
print("\nFeature importance saved to features/val_feature_importance.npy")
