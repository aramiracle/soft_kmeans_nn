import matplotlib.pyplot as plt
import numpy as np

def plot_results(X, model):
    soft_assignments = model(X).detach().numpy()
    hard_assignments = np.argmax(soft_assignments, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=hard_assignments, cmap='viridis', marker='o', s=50, alpha=0.6)
    plt.scatter(model.centroids.detach().numpy()[:, 0], model.centroids.detach().numpy()[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('Soft KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()