import torch
import torch.nn as nn
import torch.nn.functional as F

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

def train_soft_kmeans(X, num_clusters, num_epochs=300, temperature=1.0, learning_rate=0.01):
    feature_dim = X.size(1)
    soft_kmeans = SoftKMeans(num_clusters=num_clusters, feature_dim=feature_dim, temperature=temperature)
    optimizer = torch.optim.Adam(soft_kmeans.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        soft_assignments = soft_kmeans(X)
        loss = soft_kmeans.compute_loss(X, soft_assignments)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    return soft_kmeans
