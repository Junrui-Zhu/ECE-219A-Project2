"""
This file utilizes SBERT for feature extraction of the text dataset, then reduce the dimension and cluster.
By running the file, you will get the data for question 18.
"""

import torch
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from k_mean import cluster_measures
from agg_cluster import agg
from umap_reduction import umap_dim_reduce

def load_sbert_model(model_name="all-MiniLM-L6-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model

def extract_text_embeddings(model, texts):    
    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)  
    return embeddings

if __name__ == "__main__":
    newsgroups_all = fetch_20newsgroups(subset='all', remove = ('headers', 'footers', 'quotes'))
    model = load_sbert_model()
    X = extract_text_embeddings(model, newsgroups_all.data)
    y = newsgroups_all.target
    _, X = umap_dim_reduce(X, n_components=20, metric='cosine')
    y_pred, _ = agg(X, n_clusters=20)
    measurement = cluster_measures(y, y_pred, f'for SBERT extractor & UMAP projection & agg clustering, n_components={X.shape[1]}')

