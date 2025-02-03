"""
This file performs agglomerative clustering and evaluates its performance.
By running this file, you can get answer for question 14
"""
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from all_tfidf_labels import get_tfidf_labels
from umap_reduction import umap_dim_reduce

def agg(X_tfidf, n_clusters=20, linkage = "ward"):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage = linkage)
    y_pred = clustering.fit_predict(X_tfidf)
    return y_pred, clustering

def agg_measure(true_labels, pred_labels, note):
    metrics = {
        "Homogeneity": homogeneity_score(true_labels, pred_labels),
        "Completeness": completeness_score(true_labels, pred_labels),
        "V-measure": v_measure_score(true_labels, pred_labels),
        "Adjusted Rand Index": adjusted_rand_score(true_labels, pred_labels),
        "Adjusted Mutual Information Score": adjusted_mutual_info_score(true_labels, pred_labels)
    }
    print(note)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics

if __name__ == "__main__":
    tfidf, ground_truth = get_tfidf_labels()
    _, X_reduced = umap_dim_reduce(tfidf, 200, "cosine") #do not change 200 and cosine
    linkages = ["ward", "single"]
    for linkage in linkages:
        y_pred, _ = agg(X_reduced, n_clusters = 20, linkage = linkage)
        agg_measure(ground_truth, y_pred, note = "measure for " + linkage)
