import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score

# ============================================
# 📂 Load Dataset
# ============================================
path = r"D:\Level 5\ping-pong\Heart_Disease_Project\heart_disease.csv"
df = pd.read_csv(path)

print("Columns:", df.columns.tolist())
print(" Shape:", df.shape)

# Features & Target
X = df.drop(columns=["num"])
y = df["num"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 1️⃣ K-Means Clustering + Elbow Method
# ============================================
sse = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, sse, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.savefig("results/kmeans_elbow.png")
plt.show()

# Apply KMeans with chosen k (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# ============================================
# 2️⃣ Hierarchical Clustering (Dendrogram)
# ============================================
plt.figure(figsize=(8, 5))
Z = linkage(X_scaled, method="ward")
dendrogram(Z, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.savefig("results/hierarchical_dendrogram.png")
plt.show()

# Assign clusters (Agglomerative Clustering)
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
df["hier_cluster"] = agg.fit_predict(X_scaled)

# ============================================
# 3️⃣ Compare Clusters with True Labels
# ============================================
ari_kmeans = adjusted_rand_score(y, df["kmeans_cluster"])
ari_hier = adjusted_rand_score(y, df["hier_cluster"])

print(f"\n🔹 Adjusted Rand Index (KMeans vs True Labels): {ari_kmeans:.3f}")
print(f"🔹 Adjusted Rand Index (Hierarchical vs True Labels): {ari_hier:.3f}")

# Visualization of KMeans clusters
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1],
                hue=df["kmeans_cluster"], palette="Set1")
plt.title("KMeans Clusters (first 2 PCA components)")
plt.savefig("results/kmeans_clusters.png")
plt.show()

# Visualization of Hierarchical clusters
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1],
                hue=df["hier_cluster"], palette="Set2")
plt.title("Hierarchical Clusters (first 2 PCA components)")
plt.savefig("results/hierarchical_clusters.png")
plt.show()
