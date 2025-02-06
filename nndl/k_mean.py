"""
This file performs the k-mean clustering and evaluates its performance(5 measurements and contingency matrix).
Also, by running this file, it answers question 2 and 3
"""
from sklearn.cluster import KMeans
import numpy as np
from scipy.sparse import coo_matrix
from all_tfidf_labels import get_tfidf_labels
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment

def k_means_clustering(tf_idf_matrix, k=2, random_state=0, max_iter=1000, n_init=30):
    kmeans = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter, n_init=n_init)
    kmeans.fit(tf_idf_matrix)
    return kmeans.labels_, kmeans.cluster_centers_

def show_A(A, note):

    plt.figure(figsize=(max(6, A.shape[1] / 2), max(6, A.shape[0] / 2)))
    plt.imshow(A, cmap='Blues', interpolation='nearest')
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, str(A[i, j]), ha='center', va='center', color='black' if A[i, j] < A.max() / 2 else 'white')
    
    plt.xlabel("Predicted Clusters")
    plt.ylabel("True Classes")
    plt.title("Contingency Table"+ note)
    plt.colorbar()
    plt.show()

def cluster_measures(true_labels, pred_labels, note):
   
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

def k_mean_evaluate(true_labels, pred_labels, note= ''):
    result = np.array(pred_labels)
    ground_truth = np.array(true_labels)
    true_classes, true_indices = np.unique(ground_truth, return_inverse=True)
    pred_classes, pred_indices = np.unique(result, return_inverse=True)
    A = coo_matrix((np.ones_like(result), (true_indices, pred_indices)),
                   shape=(len(true_classes), len(pred_classes)),
                   dtype=int).toarray()
    if A.shape[0] == A.shape[1]:
        row_ind, col_ind = linear_sum_assignment(A, maximize=True)
        A = A[row_ind[:, np.newaxis], col_ind]                  
    show_A(A, note)
    cluster_measures(true_labels, pred_labels, note)

if __name__ == "__main__":
    tfidf, ground_truth = get_tfidf_labels()
    print(np.unique(ground_truth).shape)
    labels, _ = k_means_clustering(tfidf)
    k_mean_evaluate(ground_truth, labels)