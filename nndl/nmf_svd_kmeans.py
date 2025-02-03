import numpy as np
import matplotlib.pyplot as plt
from nmf import nmf_dim_reduction
from pca_svd import PCA_SVD
from k_mean import k_means_clustering, cluster_measures
from all_tfidf_labels import get_tfidf_labels

def evaluate_nmf_svd(tfidf, true_labels, r_values, k=2):
    results_nmf = {"Homogeneity": [], "Completeness": [], "V-measure": [], "Adjusted Rand Index": [], "Adjusted Mutual Information Score": []}
    results_svd = {"Homogeneity": [], "Completeness": [], "V-measure": [], "Adjusted Rand Index": [], "Adjusted Mutual Information Score": []}
    svd_model, SVD_matrix = PCA_SVD(tfidf, n_components=max(r_values))
    reduced_tfidf_dict = {r: SVD_matrix[:, :r] for r in r_values}  # 预存降维后的结果

    for r in r_values:
        # NMF 降维
        nmf_model, W = nmf_dim_reduction(tfidf, n_components=r)
        labels_nmf, _ = k_means_clustering(W, k=k)
        metrics_nmf = cluster_measures(true_labels, labels_nmf)
        for key in results_nmf.keys():
            results_nmf[key].append(metrics_nmf[key])

        reduced_tfidf = reduced_tfidf_dict[r]
        labels_svd, _ = k_means_clustering(reduced_tfidf, k=k)
        metrics_svd = cluster_measures(true_labels, labels_svd)
        for key in results_svd.keys():
            results_svd[key].append(metrics_svd[key])
    return results_nmf, results_svd

def plot_results(r_values, results_nmf, results_svd):
    """
    Plot the evaluation metrics for NMF and SVD.
    """
    print("start to plot")
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    # Homogeneity
    axs[0, 0].plot(r_values, results_nmf["Homogeneity"], marker='o', label="NMF")
    axs[0, 0].plot(r_values, results_svd["Homogeneity"], marker='s', label="SVD")
    axs[0, 0].set_title("Homogeneity vs r")
    axs[0, 0].set_xlabel("r")
    axs[0, 0].set_ylabel("Homogeneity")
    axs[0, 0].legend()
    # Completeness
    axs[0, 1].plot(r_values, results_nmf["Completeness"], marker='o', label="NMF")
    axs[0, 1].plot(r_values, results_svd["Completeness"], marker='s', label="SVD")
    axs[0, 1].set_title("Completeness vs r")
    axs[0, 1].set_xlabel("r")
    axs[0, 1].set_ylabel("Completeness")
    axs[0, 1].legend()
    # V-measure
    axs[1, 0].plot(r_values, results_nmf["V-measure"], marker='o', label="NMF")
    axs[1, 0].plot(r_values, results_svd["V-measure"], marker='s', label="SVD")
    axs[1, 0].set_title("V-measure vs r")
    axs[1, 0].set_xlabel("r")
    axs[1, 0].set_ylabel("V-measure")
    axs[1, 0].legend()
    # Adjusted Rand Index
    axs[1, 1].plot(r_values, results_nmf["Adjusted Rand Index"], marker='o', label="NMF")
    axs[1, 1].plot(r_values, results_svd["Adjusted Rand Index"], marker='s', label="SVD")
    axs[1, 1].set_title("Adjusted Rand Index vs r")
    axs[1, 1].set_xlabel("r")
    axs[1, 1].set_ylabel("Adjusted Rand Index")
    axs[1, 1].legend()
    # Adjusted Mutual Information Score
    axs[2, 0].plot(r_values, results_nmf["Adjusted Mutual Information Score"], marker='o', label="NMF")
    axs[2, 0].plot(r_values, results_svd["Adjusted Mutual Information Score"], marker='s', label="SVD")
    axs[2, 0].set_title("Adjusted Mutual Information Score vs r")
    axs[2, 0].set_xlabel("r")
    axs[2, 0].set_ylabel("Adjusted Mutual Information Score")
    axs[2, 0].legend()
    plt.tight_layout()
    plt.show()

# 示例调用
if __name__ == "__main__":
    tfidf_matrix, true_labels = get_tfidf_labels()
    r_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100] #to make this code run faster, delete 300
    results_nmf, results_svd = evaluate_nmf_svd(tfidf_matrix, true_labels, r_values)
    plot_results(r_values, results_nmf, results_svd)
