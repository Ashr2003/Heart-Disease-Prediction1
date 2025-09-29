import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ===============================
# PREPROCESSOR
# ===============================

# 1. Load dataset
path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_disease.csv"
df = pd.read_csv(path)

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# 2. Handle missing values
print("\nMissing values per column before cleaning:")
print(df.isnull().sum())

df = df.fillna(df.median(numeric_only=True))

print("\nMissing values per column after cleaning:")
print(df.isnull().sum())

# 3. Encoding categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)
print("\nAfter Encoding. Shape:", df_encoded.shape)

# تحويل أي Boolean لـ int (عشان ما يبوظش الحفظ)
for col in df_encoded.select_dtypes(include='bool').columns:
    df_encoded[col] = df_encoded[col].astype(int)

# 4. Standardization
scaler = StandardScaler()
numeric_cols = df_encoded.select_dtypes(include=np.number).columns
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

print("\nAfter Scaling. Example rows:")
print(df_encoded.head())

# ===============================
# Save Results (EDA Visualizations)
# ===============================
results_dir = r"D:\Level 5\ping-pong\Heart_Disease_Project\results"
os.makedirs(results_dir, exist_ok=True)

# Histogram
plt.figure(figsize=(12, 10))
df.hist(figsize=(12, 10))
plt.suptitle("Histograms of Features", fontsize=16)
hist_path = os.path.join(results_dir, "histograms.png")
plt.savefig(hist_path, dpi=300, bbox_inches="tight")
plt.close()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
heatmap_path = os.path.join(results_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

# Boxplot
plt.figure(figsize=(15, 8))
df.boxplot(rot=90)
plt.title("Boxplots of Features")
boxplot_path = os.path.join(results_dir, "boxplots.png")
plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
plt.close()

# 6. Save cleaned dataset
output_path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_cleaned.csv"
df_encoded.to_csv(output_path, index=False)

# ===============================
# Summary
# ===============================
print("\n========== PREPROCESSOR ==========")
print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"✅ Histograms saved to: {hist_path}")
print(f"✅ Heatmap saved to: {heatmap_path}")
print(f"✅ Boxplots saved to: {boxplot_path}")
