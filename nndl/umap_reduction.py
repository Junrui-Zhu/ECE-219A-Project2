import umap
from sklearn.datasets import fetch_20newsgroups
from all_tfidf_labels import *
from k_mean import *

def umap_dim_reduce(X, n_components, metric, random_state=42):
    umap_model = umap.UMAP(n_components=n_components, metric=metric, random_state=random_state)
    X_reduced = umap_model.fit_transform(X)
    return umap_model, X_reduced

if __name__ == "__main__":
    X, y = get_tfidf_labels()
    n_components_list = [5, 20, 200]
    metrics = ["cosine", "euclidean"]
    measurements = []
    for n_components in n_components_list:
        for metric in metrics:
            _, X_reduced = umap_dim_reduce(X, n_components, metric)
            y_pred, _ = k_means_clustering(X_reduced, k=20)
            measurements.append(k_mean_evaluate(y, y_pred, note=' for UMAP '+str(n_components)+' '+metric))

