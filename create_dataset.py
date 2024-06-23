import torch
from sklearn.datasets import make_blobs

def create_dataset(n_samples=300, n_features=2, n_clusters=3, random_state=42):
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state)
    X = torch.tensor(X, dtype=torch.float32)
    return X, y_true