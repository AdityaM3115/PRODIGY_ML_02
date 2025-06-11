import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Directories
DATA_PATH = os.path.join("data", "Retail_Transactions_Dataset.csv")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load the dataset
df = pd.read_csv(DATA_PATH)

# Step 2: Select relevant features
features = ['Total_Items', 'Total_Cost']
X = df[features]

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Determine optimal number of clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot and save the elbow plot
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method: Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
elbow_plot_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")
plt.savefig(elbow_plot_path)
print(f"Elbow plot saved to: {elbow_plot_path}")
plt.close()

# Step 5: Apply KMeans with chosen k (e.g., 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Visualize the clusters using a pairplot
pairplot = sns.pairplot(df, hue='Cluster', vars=features, palette='Set2')
cluster_plot_path = os.path.join(OUTPUT_DIR, "clusters_plot.png")
pairplot.fig.suptitle('Customer Segments by K-Means Clustering', y=1.02)
pairplot.savefig(cluster_plot_path)
print(f"Cluster visualization saved to: {cluster_plot_path}")
plt.close()

# Step 7: Save the clustered data
output_csv_path = os.path.join(OUTPUT_DIR, "op.csv")
df.to_csv(output_csv_path, index=False)
print(f"Clustered data saved to: {output_csv_path}")
