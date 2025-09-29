# ==============================
# 02_pca_analysis.ipynb
# ==============================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load cleaned dataset
path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_cleaned.csv"  
df = pd.read_csv(path)

print("Cleaned Data Shape:", df.shape)

# بنفصل الـ Features عن الـ Target (لو target اسمه num موجود هنشيله)
if "num" in df.columns:
    X = df.drop("num", axis=1)
    y = df["num"]
else:
    X = df.copy()
    y = None

# 2. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# 3. Explained Variance Ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

print("\nExplained Variance for first 10 components:")
for i, var in enumerate(explained_variance[:10], start=1):
    print(f"PC{i}: {var:.4f}")

# ===============================
# Save Results (Visualizations)
# ===============================
results_dir = r"D:\Level 5\ping-pong\Heart_Disease_Project\results"
os.makedirs(results_dir, exist_ok=True)

# 4. Plot cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained by PCA")
plt.grid(True)
cumulative_plot_path = os.path.join(results_dir, "pca_cumulative_variance.png")
plt.savefig(cumulative_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# 5. Scatter plot of first 2 components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y if y is not None else "blue", cmap="viridis", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Scatter Plot (First 2 Components)")
if y is not None:
    plt.colorbar(label="Target (num)")
scatter_plot_path = os.path.join(results_dir, "pca_scatter_plot.png")
plt.savefig(scatter_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# 6. Save PCA-transformed dataset
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)])
if y is not None:
    pca_df["target"] = y.values

output_path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_pca.csv"
pca_df.to_csv(output_path, index=False)

# ===============================
# Summary
# ===============================
print("\n========== PCA ANALYSIS ==========")
print(f"✅ PCA-transformed dataset saved to: {output_path}")
print(f"✅ Cumulative variance plot saved to: {cumulative_plot_path}")
print(f"✅ PCA scatter plot saved to: {scatter_plot_path}")
