import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Define the SoftKMeans class as provided
class SoftKMeans(nn.Module):
    def __init__(self, num_clusters, feature_dim, temperature=1.0):
        super(SoftKMeans, self).__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.centroids = nn.Parameter(torch.randn(num_clusters, feature_dim))

    def forward(self, x):
        distance_matrix = torch.cdist(x, self.centroids)
        soft_assignments = F.softmax(-distance_matrix / self.temperature, dim=1)
        return soft_assignments

    def compute_loss(self, x, soft_assignments):
        expanded_x = x.unsqueeze(1)
        expanded_centroids = self.centroids.unsqueeze(0)
        distances = torch.sum((expanded_x - expanded_centroids) ** 2, dim=2)
        loss = torch.sum(soft_assignments * distances)
        return loss / x.size(0)

# Create synthetic dataset
n_samples = 1000
n_features = 2
n_clusters = 5

X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)
X = torch.tensor(X, dtype=torch.float32)

# Initialize SoftKMeans
num_clusters = 5
temperature = 1.0
soft_kmeans = SoftKMeans(num_clusters=num_clusters, feature_dim=n_features, temperature=temperature)

# Training
optimizer = torch.optim.Adam(soft_kmeans.parameters(), lr=0.01)
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    soft_assignments = soft_kmeans(X)
    loss = soft_kmeans.compute_loss(X, soft_assignments)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Get the final soft assignments
soft_assignments = soft_kmeans(X).detach().numpy()
hard_assignments = np.argmax(soft_assignments, axis=1)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=hard_assignments, cmap='viridis', marker='o', s=50, alpha=0.6)
plt.scatter(soft_kmeans.centroids.detach().numpy()[:, 0], soft_kmeans.centroids.detach().numpy()[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Soft KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
