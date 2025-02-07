"""
This file performs tSNE dim reduction and visualization.
By running this file, you get answers for question 23
"""

from load_images import load_images_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":

    X, y = load_images_data()

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tSNE = tsne.fit_transform(X)  # Reduce to 2D

    # Plot results
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tSNE[:, 0], X_tSNE[:, 1], c=y, cmap="jet", alpha=0.7)
    plt.colorbar(scatter, label="ground-truth labels")
    plt.title("t-SNE Visualization of tf_flowers Dataset")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
