"""
This file performs a comparative study of different dim reduction and cluster algorithms over text dataset.
By setting parameters and running this file multiple times, you get data for question 17.
"""

from all_tfidf_labels import get_tfidf_labels
from k_mean import k_means_clustering, cluster_measures
from agg_cluster import agg
from umap_reduction import umap_dim_reduce
from pca_svd import PCA_SVD
from nmf import nmf_dim_reduction
from hdbscan_cluster import hdbscan_clustering

if __name__ == "__main__":

    X, y = get_tfidf_labels()

    dim_reduction = 'SVD' # SVD, NMF, UMAP, None
    n_components = 5 # [5, 20, 200]
    cluster = 'Agg' # KMeans, Agg, HDBSCAN

    if dim_reduction == 'SVD':
        _, X = PCA_SVD(X, n_components=n_components)
    elif dim_reduction == 'NMF':
        _, X = nmf_dim_reduction(X, n_components=n_components)
    elif dim_reduction == 'UMAP':
        _, X = umap_dim_reduce(X, n_components=n_components, metric='cosine')
    else:
        pass

    if cluster == 'KMeans':
        for k in [10, 20, 50]:
            y_pred, _ = k_means_clustering(X, k=k)
            measurement = cluster_measures(y, y_pred, f'for k means and k={k}, {dim_reduction}, n_components={X.shape[1]}')
    elif cluster == 'Agg':
        y_pred, _ = agg(X, n_clusters=20)
        measurement = cluster_measures(y, y_pred, f'for agg and n_clusters={20}, {dim_reduction}, n_components={X.shape[1]}')
    elif cluster == 'HDBSCAN':
        for min_cluster_size in [100, 200]:
            y_pred, _ = hdbscan_clustering(X, min_cluster_size=min_cluster_size, min_samples=5, metric='cosine')
            measurement = cluster_measures(y, y_pred, f'for hdbscan and min_cluster_size={min_cluster_size}, {dim_reduction}, n_components={X.shape[1]}')