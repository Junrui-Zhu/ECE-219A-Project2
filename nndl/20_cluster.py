from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nmf import nmf_dim_reduction
from pca_svd import PCA_SVD
from k_mean import k_means_clustering, k_mean_evaluate
newsgroups_all = fetch_20newsgroups(subset='all', remove = ('headers', 'footers', 'quotes'))
    
vectorizer = TfidfVectorizer(min_df = 3, stop_words = 'english')
tfidf_matrix = vectorizer.fit_transform(newsgroups_all.data)
true_labels = newsgroups_all.target

# r_values = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, ]
# results_nmf, results_svd = evaluate_nmf_svd(tfidf_matrix, true_labels, r_values, k=20)
# plot_results(r_values, results_nmf, results_svd)

# The above code will show the opitimal settings. SVD:25 NMF:10
SVD_dim = 25
NMF_dim = 10
svd_model, SVD_matrix = PCA_SVD(tfidf_matrix, n_components=SVD_dim)
nmf_model, NMF_matrix = nmf_dim_reduction(tfidf_matrix, n_components=NMF_dim)
labels_svd, _ = k_means_clustering(SVD_matrix, k=20)
labels_nmf, _ = k_means_clustering(NMF_matrix, k=20)
k_mean_evaluate(true_labels, labels_svd, note = ' for SVD')
k_mean_evaluate(true_labels, labels_nmf, note = ' for NMF')