import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

#  المسار لملف CSV
path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_disease.csv"

#  تحميل البيانات
df = pd.read_csv(path)
print("Columns:", df.columns.tolist(), "\n Shape:", df.shape)

#  تقسيم البيانات
X = df.drop(["num", "id"], axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  GridSearchCV ل RandomForest
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 4]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)

#  حفظ النتائج النصية
os.makedirs("results", exist_ok=True)
with open("results/hyperparam_tuning_results.txt", "w") as f:
    f.write(f"Best Random Forest Params: {grid.best_params_}\n")
    f.write(f"Best RF Accuracy: {accuracy_score(y_test, y_pred)}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\n Results saved to results/hyperparam_tuning_results.txt")

#  رسم ROC Curves
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

# One-vs-Rest
ovr = OneVsRestClassifier(best_rf)
ovr.fit(X_train, label_binarize(y_train, classes=np.unique(y)))
y_score = ovr.predict_proba(X_test)

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=np.unique(y))[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.legend(loc="lower right")

#  حفظ الصورة
plt.savefig("results/roc_curve.png")
plt.close()
print(" ROC Curve saved to results/roc_curve.png")
