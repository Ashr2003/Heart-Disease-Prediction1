import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ============================================
#  Load Dataset
# ============================================
path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_disease.csv"
df = pd.read_csv(path)

print("Columns:", df.columns.tolist())
print(" Shape:", df.shape)

# ============================================
#  Features & Target
# ============================================
X = df.drop(columns=["num"])
y = df["num"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
#  Train Models
# ============================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = []
y_pred_prob = None  # هنخزن احتمالات آخر موديل علشان ROC

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append([name, acc, prec, rec, f1])

    # ناخد احتمالات آخر موديل (SVM) علشان ROC
    y_pred_prob = model.predict_proba(X_test)

# ============================================
#  Save Results
# ============================================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n Model Performance:\n", results_df)

# Create results folder if not exists
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/supervised_metrics.csv", index=False)
print("\n Results saved in 'results/' folder")

# ============================================
#  ROC Curves - All Classes + Separate
# ============================================
# Binarize target
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# ---------- 1) ROC Curve (All Classes in one plot) ----------
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
plt.title("ROC Curves - All Classes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("results/roc_all_classes.png")
plt.show()

# ---------- 2) ROC Curve (Each Class Separate) ----------
plt.figure(figsize=(15, 10))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.subplot(2, 3, i + 1)  # شبكة 2 × 3
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.title(f"ROC Curve - Class {i}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("results/roc_classes_subplots.png")
plt.show()
