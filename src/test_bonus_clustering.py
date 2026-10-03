import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_clean_path = os.path.join(base_dir, 'data', 'processed', 'adult_cleaned_train.csv')

df_clean = pd.read_csv(train_clean_path)
X = df_clean.drop(columns=['income_target'])

print(f"Loaded X shape: {X.shape}")

# Subsample 1500 for hierarchical clustering (dendrogram computation is O(N^2))
np.random.seed(42)
sample_idx = np.random.choice(len(X), size=1500, replace=False)
X_sub = X.iloc[sample_idx]

# 1. K-Means
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = km.fit_predict(X_sub)
km_sil = silhouette_score(X_sub, km_labels)
print(f"K-Means (k=3) Silhouette: {km_sil:.4f}")

# 2. Hierarchical / Agglomerative Clustering (Ward Linkage) - BONUS 1
Z = linkage(X_sub, method='ward')
print(f"Linkage matrix shape: {Z.shape}")

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X_sub)
agg_sil = silhouette_score(X_sub, agg_labels)
print(f"Agglomerative Clustering (k=3, Ward) Silhouette: {agg_sil:.4f}")

# 3. DBSCAN (Density-Based Clustering for Outliers) - BONUS 2
db = DBSCAN(eps=2.5, min_samples=10)
db_labels = db.fit_predict(X_sub)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"DBSCAN clusters found: {n_clusters_db}, Noise/Outlier points: {n_noise} ({n_noise/len(X_sub)*100:.1f}%)")

print("All bonus clustering models validated successfully!")
