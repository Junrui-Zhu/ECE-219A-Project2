"""
This file performs NMF dimensionality reduction.
"""

from sklearn.decomposition import NMF
import numpy as np

def nmf_dim_reduction(tfidf_matrix, n_components=10, random_state=42):
    nmf_model = NMF(n_components=n_components, random_state=random_state, max_iter=500, init='nndsvd')# set max_iter = 500 to ensure convergence
    W = nmf_model.fit_transform(tfidf_matrix)
    return nmf_model, W