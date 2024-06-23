from create_dataset import create_dataset
from train import train_soft_kmeans
from visualize import plot_results

if __name__ == "__main__":
    # Create dataset
    X, y_true = create_dataset(n_samples=500, n_features=2, n_clusters=5, random_state=42)

    # Train the model
    num_clusters = 5
    num_epochs = 1000
    temperature = 1.0
    learning_rate = 0.01
    model = train_soft_kmeans(X, num_clusters, num_epochs, temperature, learning_rate)

    # Plot the results
    plot_results(X, model)

