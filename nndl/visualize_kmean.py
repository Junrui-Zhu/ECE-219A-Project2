import matplotlib.pyplot as plt
from k_mean import k_means_clustering
from nmf import nmf_dim_reduction
from pca_svd import PCA_SVD

from all_tfidf_labels import get_tfidf_labels

def visualize_clusters(tfidf_matrix, true_labels, r_svd=10, r_nmf=8, k=2):
    """
    Visualize the clustering results using SVD and NMF.
    Answer for problem 8 and 9
    """

    # Step 1: Perform dimensionality reduction (SVD)
    svd_model, SVD_matrix = PCA_SVD(tfidf_matrix, n_components=r_svd)
    nmf_model, NMF_matrix = nmf_dim_reduction(tfidf_matrix, n_components=r_nmf)

    # Step 2: Further reduce to 2D using SVD for visualization
    _, svd_vis = PCA_SVD(SVD_matrix, n_components=2)
    _, nmf_vis = nmf_dim_reduction(NMF_matrix, n_components=2)

    # Step 3: Apply K-Means clustering
    labels_svd, _ = k_means_clustering(SVD_matrix)
    labels_nmf, _ = k_means_clustering(NMF_matrix)

    # Step 4: Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # SVD - Ground Truth Labels
    axs[0, 0].scatter(svd_vis[:, 0], svd_vis[:, 1], c=true_labels, cmap="viridis", alpha=0.6)
    axs[0, 0].set_title("SVD Projection - Ground Truth Labels")

    # SVD - K-Means Clustering Labels
    axs[0, 1].scatter(svd_vis[:, 0], svd_vis[:, 1], c=labels_svd, cmap="tab10", alpha=0.6)
    axs[0, 1].set_title("SVD Projection - K-Means Clusters")

    # NMF - Ground Truth Labels
    axs[1, 0].scatter(nmf_vis[:, 0], nmf_vis[:, 1], c=true_labels, cmap="viridis", alpha=0.6)
    axs[1, 0].set_title("NMF Projection - Ground Truth Labels")

    # NMF - K-Means Clustering Labels
    axs[1, 1].scatter(nmf_vis[:, 0], nmf_vis[:, 1], c=labels_nmf, cmap="tab10", alpha=0.6)
    axs[1, 1].set_title("NMF Projection - K-Means Clusters")

    plt.tight_layout()
    plt.show()

# 示例调用
if __name__ == "__main__":
    tfidf_matrix, true_labels = get_tfidf_labels()
    visualize_clusters(tfidf_matrix, true_labels)
