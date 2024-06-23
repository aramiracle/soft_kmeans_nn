# Soft KMeans Clustering

This project demonstrates the implementation of Soft KMeans clustering using PyTorch. Soft KMeans is a variant of KMeans clustering that assigns soft probabilities (soft assignments) to data points instead of hard assignments. This allows for more nuanced clustering where data points can belong to multiple clusters with varying degrees of membership.

## Overview

### Parts of the Project:

1. **Dataset Creation**:
   - Generates a synthetic dataset using `make_blobs` from scikit-learn.
   - Converts the dataset into a PyTorch tensor for compatibility with PyTorch-based models.

2. **Model Training**:
   - Implements the Soft KMeans algorithm as a PyTorch `nn.Module`.
   - Defines the model's architecture, including the calculation of distances and soft assignments using the softmax function.
   - Utilizes the Adam optimizer to minimize the clustering loss, which is computed based on the distances between data points and cluster centroids.

3. **Result Visualization**:
   - Plots the clustered data points and centroids after training.
   - Displays how the Soft KMeans algorithm partitions the data into clusters based on learned centroids.

### Workflow:

- **Dataset Creation**: 
  - Generates a synthetic dataset with specified characteristics (number of samples, features, clusters).
  - Converts the dataset into a format suitable for training machine learning models in PyTorch.

- **Model Training**:
  - Initializes the Soft KMeans model with parameters such as the number of clusters and temperature (a hyperparameter controlling the softness of assignments).
  - Iteratively updates the model parameters (centroids) using backpropagation and the Adam optimizer.
  - Monitors the training process by printing the loss at regular intervals to track convergence.

- **Result Visualization**:
  - After training, visualizes the clustering results by plotting the data points colored according to their cluster assignments.
  - Overlays the learned centroids on the plot to show the central points of each cluster.

### Requiremnets

To install requirements using pip you can run this command in your terminal:

```
pip install -r requirements.txt
```

### Usage:

To run this project:
- Ensure Python 3.x and required libraries (PyTorch, scikit-learn, matplotlib, numpy) are installed.
- Execute the main script, which orchestrates dataset creation, model training, and result visualization in sequence.

### Purpose:

This project serves as a practical implementation of Soft KMeans clustering, showcasing its application in partitioning data points into clusters with soft assignments. It demonstrates fundamental concepts in machine learning such as model training, loss optimization, and result visualization using widely-used libraries in the Python ecosystem.

## Conclusion

Soft KMeans clustering provides a flexible approach to clustering data by allowing data points to belong to multiple clusters simultaneously. By implementing Soft KMeans in PyTorch and visualizing its results, this project illustrates how machine learning techniques can be applied to real-world data clustering tasks.

## Medium

You can also go to this [link](https://medium.com/@a.r.amouzad.m/soft-kmeans-with-neural-networks-000e42131086) in medium site for see more details.
