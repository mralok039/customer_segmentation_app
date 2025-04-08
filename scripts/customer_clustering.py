import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Step 1: Load RFM Data
rfm_path = "outputs/rfm_scores.csv"
if not os.path.exists(rfm_path):
    raise FileNotFoundError(f"{rfm_path} not found. Please generate RFM scores first.")

rfm = pd.read_csv(rfm_path)

# Step 2: Select Features for Clustering
features = rfm[['Recency', 'Frequency', 'Monetary']]

# Step 3: Standardize the Features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(features)

# Step 4: Elbow Method to Determine Optimal k
sse = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

# Step 5: Plot the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K, sse, 'bo--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.xticks(K)
plt.tight_layout()
plt.show()

# Step 6: Apply KMeans with Optimal k
optimal_k = 4  # You can change this after observing the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Step 7: Visualize Clusters (Recency vs Monetary)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=rfm,
    x='Recency',
    y='Monetary',
    hue='Cluster',
    palette='Set2',
    s=70
)
plt.title('Customer Segments (Recency vs Monetary)')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Step 8: Save the clustered RFM data
output_path = "outputs/rfm_with_clusters.csv"
rfm.to_csv(output_path, index=False)
print(f"✅ RFM with clusters saved to {output_path}")
