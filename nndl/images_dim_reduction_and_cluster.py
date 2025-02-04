from load_images import load_images_data
from sklearn.metrics import adjusted_rand_score
from k_mean import k_means_clustering
from agg_cluster import agg
from umap_reduction import umap_dim_reduce
from pca_svd import PCA_SVD
from torch_model import Autoencoder
from hdbscan_cluster import hdbscan_clustering
import numpy as np

if __name__ == "__main__":

    X, y = load_images_data()
    X_auto =Autoencoder(n_components=50).fit_transform(X)
    _, X_svd = PCA_SVD(X, n_components=50)
    _, X_umap = umap_dim_reduce(X, n_components=50, metric='cosine')
    X_all = [X, X_svd, X_umap, X_auto]
    rand_scores = np.zeros((4, 3))
    for i in range(4):
        for j in range(3):
            if j == 0:
                y_pred, _ = k_means_clustering(X_all[i], k=5)
            elif j == 1:
                y_pred, _ = agg(X_all[i], n_clusters=5)
            elif j == 2:
                y_pred, _ = hdbscan_clustering(X_all[i], min_cluster_size=10, min_samples=5, metric='cosine')
            rand_scores[i][j] = adjusted_rand_score(y, y_pred)
    print(rand_scores)

    # [
    # [0.18919803 0.27310444 0.03047664]
    # [0.18977365 0.18877595 0.04045094]
    # [0.46351037 0.46195567 0.17351392]
    # [0.18585654 0.17665086 0.01327472]
    # ]


    