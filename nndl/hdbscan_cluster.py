"""
This file implements the hdbscan clustering algorithm.
By running this file, you will get the answer for question 15, 16
"""

import hdbscan
from all_tfidf_labels import get_tfidf_labels
from k_mean import cluster_measures, k_mean_evaluate
from scipy.spatial.distance import pdist, squareform
from umap_reduction import umap_dim_reduce

def hdbscan_clustering(X, min_cluster_size=10, min_samples=5, metric='euclidean'):
    if metric != 'cosine':
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
        y_pred = clustering.fit_predict(X)
    else:
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
        distance_matrix = squareform(pdist(X, metric="cosine"))
        y_pred = clustering.fit_predict(distance_matrix)
    return y_pred, clustering

if __name__ == "__main__":
    X, y = get_tfidf_labels()
    _, X = umap_dim_reduce(X, 20, "cosine")
    min_cluster_size_list = [20, 100, 200]
    for min_cluster_size in min_cluster_size_list:
        y_pred, _ = hdbscan_clustering(X, min_cluster_size, metric='cosine')
        measurement = cluster_measures(y, y_pred, f'min cluster size = {min_cluster_size}')

    # min_cluster_size = 200 is the best
    y_pred, _ = hdbscan_clustering(X, 200, metric='cosine')
    k_mean_evaluate(y, y_pred, f' for min cluster size = 200')
