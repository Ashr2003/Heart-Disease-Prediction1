# ==============================
# 03_feature_selection.ipynb
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest

# ====================================================
#  1) Load Dataset
# ====================================================
path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_disease.csv"
df = pd.read_csv(path)

print("FEATURE_SELECTION")
print("======================================")
print(" Columns:", df.columns.tolist())
print(" Shape before cleaning:", df.shape)

# ====================================================
#  2) Separate Features (X) & Target (y)
# ====================================================
X = df.drop(columns=["num", "id"], errors="ignore")  # بنشيل target + id
y = df["num"]

# ====================================================
#  3) تحويل target لــ Classes
# ====================================================

# 🔹 Option 1: Binary Classification
y_binary = (y > 0).astype(int)

# 🔹 Option 2: Multi-Class Classification
y_multi = y.astype(int)

# ← هنا نحدد نوع التصنيف
y = y_multi  

print(" Target classes:", np.unique(y))

# ====================================================
# 4) Feature Importance باستخدام RandomForest
# ====================================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)

# plot + save
plt.figure(figsize=(10, 8))
feat_importances.sort_values().plot(kind='barh')
plt.title("Feature Importance (RandomForest)")

results_dir = r"D:\Level 5\ping-pong\Heart_Disease_Project\results"
os.makedirs(results_dir, exist_ok=True)

rf_plot_path = os.path.join(results_dir, "feature_importance_randomforest.png")
plt.savefig(rf_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# ====================================================
#  5) Recursive Feature Elimination (RFE)
# ====================================================
log_reg = LogisticRegression(max_iter=5000)
selector = RFE(log_reg, n_features_to_select=5)
selector = selector.fit(X, y)

rfe_features = X.columns[selector.support_].tolist()

# ====================================================
#  6) Chi-Square Test
# ====================================================
X_pos = X.apply(lambda col: col.abs())  # لازم قيم موجبة
chi2_selector = SelectKBest(score_func=chi2, k=5)
chi2_selector.fit(X_pos, y)

chi2_features = X.columns[chi2_selector.get_support()].tolist()

# ====================================================
# Save results in a text file
# ====================================================
report_path = os.path.join(results_dir, "feature_selection_report.txt")
with open(report_path, "w") as f:
    f.write("========== FEATURE SELECTION REPORT ==========\n\n")
    f.write("RandomForest Top Features:\n")
    f.write(", ".join(feat_importances.sort_values(ascending=False).head(10).index.tolist()) + "\n\n")
    f.write(f"RFE Selected Features: {rfe_features}\n\n")
    f.write(f"Chi-Square Selected Features: {chi2_features}\n\n")

# ====================================================
# Summary
# ====================================================
print("\n========== SUMMARY ==========")
print(f"✅ RandomForest plot saved to: {rf_plot_path}")
print(f"✅ Feature selection report saved to: {report_path}")
print("RFE Selected Features:", rfe_features)
print("Chi-Square Selected Features:", chi2_features)
