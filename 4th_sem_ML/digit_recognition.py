"""
============================================================
  Digit Recognition using Support Vector Machine (SVM)
  Dataset: MNIST (Kaggle - Digit Recognizer)
============================================================

SETUP INSTRUCTIONS:
    pip install pandas numpy matplotlib seaborn scikit-learn

DATASET:
    Download from: https://www.kaggle.com/competitions/digit-recognizer/data
    Files needed: train.csv  (and optionally test.csv)
    Place them in the same directory as this script.

    Alternatively, this script auto-falls back to sklearn's
    built-in digits dataset so you can run it immediately
    without downloading anything.
============================================================
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.datasets import load_digits   # fallback only

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# 1.  DATASET  –  Load & Explore
# ─────────────────────────────────────────────

KAGGLE_PATH = "train.csv"          # path to Kaggle MNIST CSV

def load_data(path: str):
    """
    Tries to load the Kaggle MNIST CSV.
    Falls back to sklearn's digits dataset when the file is absent.
    Returns X (pixel values) and y (labels) as numpy arrays.
    """
    if os.path.exists(path):
        print(f"[INFO] Loading Kaggle MNIST from '{path}' …")
        df = pd.read_csv(path)

        # ── Initial Exploration ──────────────────────────────────────
        print("\n── Dataset shape ──────────────────────────────────────")
        print(f"  Rows × Cols : {df.shape}")

        print("\n── First 5 rows (label + first 5 pixels) ──────────────")
        print(df.iloc[:5, :6])

        print("\n── Null values per column (sum) ───────────────────────")
        null_sum = df.isnull().sum().sum()
        print(f"  Total nulls : {null_sum}")

        print("\n── Class distribution ─────────────────────────────────")
        print(df["label"].value_counts().sort_index().to_string())

        X = df.drop("label", axis=1).values.astype(np.float32)
        y = df["label"].values
    else:
        print("[WARN] Kaggle CSV not found. Using sklearn built-in 'digits' dataset.")
        print("       (8×8 images, 1 797 samples, 10 classes)\n")
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target

        print(f"  Shape : {X.shape}")
        print(f"  Classes : {np.unique(y)}")

    return X, y


X_raw, y = load_data(KAGGLE_PATH)

# ─────────────────────────────────────────────
# 2.  DATA PREPROCESSING
# ─────────────────────────────────────────────

print("\n\n══════════════════════════════════════════")
print("  STEP 2 – Preprocessing")
print("══════════════════════════════════════════")

# 2-a  Normalise pixel values to [0, 1]
X = X_raw / 255.0 if X_raw.max() > 1.0 else X_raw
print(f"  Pixel range after normalisation : [{X.min():.2f}, {X.max():.2f}]")

# 2-b  Train / Test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# 2-c  Visualise sample images before training
n_side   = int(np.sqrt(X.shape[1]))          # 28 for MNIST, 8 for sklearn
img_size = (n_side, n_side)

print(f"\n  Image grid size inferred : {img_size}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample Training Images", fontsize=14, fontweight="bold")
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_train[i].reshape(img_size), cmap="gray")
    ax.set_title(f"Label: {y_train[i]}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=120)
plt.show()
print("  [Saved] sample_images.png")

# ─────────────────────────────────────────────
# 3.  MODEL BUILDING  –  SVM with two kernels
# ─────────────────────────────────────────────

print("\n\n══════════════════════════════════════════")
print("  STEP 3 – Model Building")
print("══════════════════════════════════════════")

# For large datasets (Kaggle full MNIST = 42 000 rows) SVM can be slow.
# We cap training size at 10 000 for the initial kernel comparison;
# the full set is used for the optimised model later.
MAX_TRAIN = 10_000
if X_train.shape[0] > MAX_TRAIN:
    print(f"  [INFO] Subsetting training set to {MAX_TRAIN} rows for fast kernel demo.")
    idx = np.random.choice(X_train.shape[0], MAX_TRAIN, replace=False)
    X_tr_small, y_tr_small = X_train[idx], y_train[idx]
else:
    X_tr_small, y_tr_small = X_train, y_train

kernels = {
    "linear": SVC(kernel="linear", C=1.0, random_state=42),
    "rbf"   : SVC(kernel="rbf",    C=10.0, gamma="scale", random_state=42),
}

kernel_results = {}
for name, clf in kernels.items():
    print(f"\n  Training SVM [{name}] …")
    clf.fit(X_tr_small, y_tr_small)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    kernel_results[name] = {"model": clf, "acc": acc, "pred": y_pred}

    print(f"    Accuracy  : {acc * 100:.2f} %")
    print(f"    Report    :")
    print(classification_report(y_test, y_pred, zero_division=0))

# ─────────────────────────────────────────────
# 4.  MODEL EVALUATION  –  Best kernel so far
# ─────────────────────────────────────────────

print("\n\n══════════════════════════════════════════")
print("  STEP 4 – Model Evaluation")
print("══════════════════════════════════════════")

best_kernel = max(kernel_results, key=lambda k: kernel_results[k]["acc"])
best_base   = kernel_results[best_kernel]
print(f"  Best kernel : {best_kernel}  (Acc = {best_base['acc']*100:.2f} %)")

# Confusion matrix
cm = confusion_matrix(y_test, best_base["pred"])
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=np.unique(y_test))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix – SVM [{best_kernel}]",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix_base.png", dpi=120)
plt.show()
print("  [Saved] confusion_matrix_base.png")

# ─────────────────────────────────────────────
# 5.  OPTIMISATION  –  GridSearchCV
# ─────────────────────────────────────────────

print("\n\n══════════════════════════════════════════")
print("  STEP 5 – Hyperparameter Optimisation")
print("══════════════════════════════════════════")

# Grid is intentionally small to keep runtime manageable.
param_grid = {
    "C"      : [1, 10, 50],
    "gamma"  : ["scale", "auto"],
    "kernel" : ["rbf", "linear"],
}

# Use a subset for GridSearch if dataset is large
MAX_GRID = 5_000
if X_tr_small.shape[0] > MAX_GRID:
    idx_g = np.random.choice(X_tr_small.shape[0], MAX_GRID, replace=False)
    X_g, y_g = X_tr_small[idx_g], y_tr_small[idx_g]
else:
    X_g, y_g = X_tr_small, y_tr_small

print(f"  Running GridSearchCV on {X_g.shape[0]} samples …")
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_g, y_g)

best_params = grid_search.best_params_
best_score  = grid_search.best_score_
print(f"\n  Best CV accuracy : {best_score * 100:.2f} %")
print(f"  Best parameters  : {best_params}")

# Retrain best model on the full training set
print("\n  Retraining best model on full training set …")
best_svm = SVC(**best_params, random_state=42)
best_svm.fit(X_train, y_train)           # full training set here
y_pred_best = best_svm.predict(X_test)
acc_best    = accuracy_score(y_test, y_pred_best)
print(f"  Test Accuracy (optimised) : {acc_best * 100:.2f} %")
print("\n  Full Classification Report :")
print(classification_report(y_test, y_pred_best, zero_division=0))

# ─────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────

print("\n\n══════════════════════════════════════════")
print("  STEP 6 – Visualisation")
print("══════════════════════════════════════════")

# 6-a  Sample predictions
fig, axes = plt.subplots(3, 5, figsize=(14, 8))
fig.suptitle("Sample Predictions – Optimised SVM",
             fontsize=14, fontweight="bold")
wrong_indices  = np.where(y_pred_best != y_test)[0]
correct_indices= np.where(y_pred_best == y_test)[0]

for i, ax in enumerate(axes.flatten()):
    if i < 10:                            # first 10 = correct
        idx = correct_indices[i]
        border_colour = "green"
    else:                                 # last 5 = wrong
        idx = wrong_indices[i - 10] if len(wrong_indices) > (i - 10) else correct_indices[i]
        border_colour = "red"

    ax.imshow(X_test[idx].reshape(img_size), cmap="gray")
    ax.set_title(
        f"True:{y_test[idx]}  Pred:{y_pred_best[idx]}",
        fontsize=9,
        color=border_colour,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(border_colour)
        spine.set_linewidth(2)
    ax.axis("off")

fig.text(0.5, 0.01,
         "Green border = Correct  |  Red border = Wrong",
         ha="center", fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("sample_predictions.png", dpi=120)
plt.show()
print("  [Saved] sample_predictions.png")

# 6-b  Confusion matrix heatmap (optimised model)
cm_best = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    cm_best,
    annot=True, fmt="d",
    cmap="YlOrRd",
    linewidths=0.5,
    xticklabels=np.unique(y_test),
    yticklabels=np.unique(y_test),
    ax=ax,
)
ax.set_title("Confusion Matrix Heatmap – Optimised SVM",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label",      fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=120)
plt.show()
print("  [Saved] confusion_matrix_heatmap.png")

# 6-c  Kernel accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
names  = list(kernel_results.keys()) + ["SVM (optimised)"]
scores = [v["acc"] * 100 for v in kernel_results.values()] + [acc_best * 100]
colours= ["#4C72B0", "#DD8452", "#55A868"]
bars   = ax.bar(names, scores, color=colours, edgecolor="black", linewidth=0.8)
ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("SVM Kernel & Optimisation Comparison", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("kernel_comparison.png", dpi=120)
plt.show()
print("  [Saved] kernel_comparison.png")

# ─────────────────────────────────────────────
# 7. (Optional)  COMPARISON  –  Logistic Regression
# ─────────────────────────────────────────────

print("\n\n══════════════════════════════════════════")
print("  STEP 7 (Optional) – Comparison: Logistic Regression")
print("══════════════════════════════════════════")

lr = LogisticRegression(
    max_iter=1000,
    solver="saga",
    n_jobs=-1,
    random_state=42,
)
print("  Training Logistic Regression …")
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr     = accuracy_score(y_test, y_pred_lr)
print(f"  LR Test Accuracy : {acc_lr * 100:.2f} %")
print("\n  Classification Report :")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# Final comparison chart
fig, ax = plt.subplots(figsize=(8, 5))
all_names   = list(kernel_results.keys()) + ["SVM (optimised)", "Logistic Regression"]
all_scores  = [v["acc"] * 100 for v in kernel_results.values()] + [acc_best * 100, acc_lr * 100]
all_colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
bars = ax.bar(all_names, all_scores, color=all_colours, edgecolor="black", linewidth=0.8)
ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)
ax.set_ylim(0, 108)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Comparison: SVM vs Logistic Regression",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
plt.show()
print("  [Saved] model_comparison.png")

# ─────────────────────────────────────────────
# 8.  CONCLUSION
# ─────────────────────────────────────────────

print("\n\n══════════════════════════════════════════")
print("  CONCLUSION")
print("══════════════════════════════════════════")

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Model Performance Summary                              │
  ├──────────────────────────────┬──────────────────────────┤
  │  SVM (linear kernel)         │  {kernel_results['linear']['acc']*100:>6.2f} %              │
  │  SVM (RBF kernel)            │  {kernel_results['rbf']['acc']*100:>6.2f} %              │
  │  SVM (optimised via Grid)    │  {acc_best*100:>6.2f} %              │
  │  Logistic Regression         │  {acc_lr*100:>6.2f} %              │
  └──────────────────────────────┴──────────────────────────┘

  ✔  Strengths of SVM for digit recognition
     • High accuracy even in high-dimensional pixel spaces.
     • The kernel trick (RBF) captures non-linear boundaries
       between digits (e.g., 4 vs 9, 3 vs 8) effectively.
     • Robust to overfitting when C is properly tuned.
     • Works well with relatively small training sets.

  ✘  Limitations of SVM for digit recognition
     • Training time scales poorly (O(n²–n³)) with dataset size;
       the full 42 000-sample MNIST can be slow.
     • Predicting a single sample requires computing the kernel
       against all support vectors → latency grows with n.
     • No spatial/translational invariance; CNNs vastly outperform
       SVMs on raw pixels (CNNs reach >99 % on MNIST).
     • Memory-intensive for large kernelised models.
     • Hyper-parameter search (C, γ) adds significant cost.

  💡  Recommendation
     For production digit recognition, prefer a CNN (e.g. LeNet-5)
     or a pre-trained model.  SVMs remain an excellent, interpretable
     baseline that is fast to implement and competitive at ≤10 k samples.
""")
