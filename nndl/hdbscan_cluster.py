import hdbscan
from scipy.spatial.distance import pdist, squareform

def hdbscan_clustering(X, min_cluster_size=10, min_samples=5, metric='euclidean'):
    if metric != 'cosine':
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
        y_pred = clustering.fit_predict(X)
    else:
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
        distance_matrix = squareform(pdist(X, metric="cosine"))
        y_pred = clustering.fit_predict(distance_matrix)
    return y_pred, clustering