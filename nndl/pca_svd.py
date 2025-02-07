"""
This file performs SVD algorithm.
By running the file, you will get answer for question 4
"""

from all_tfidf_labels import get_tfidf_labels
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

def PCA_SVD(all_tfidf, n_components = 1000):
  svd = TruncatedSVD(n_components=n_components)
  svd_reducted_matrix = svd.fit_transform(all_tfidf)
  return svd, svd_reducted_matrix

if __name__ == "__main__":
  all_tfidf, _ = get_tfidf_labels()
  svd, svd_matrix = PCA_SVD(all_tfidf)
  cumulative_evr = np.cumsum(svd.explained_variance_ratio_[:1000])
  Ks = [1, 10, 20, 50, 100, 200, 350, 500, 750, 1000]
  markers = [cumulative_evr[k - 1] for k in Ks]

  plt.figure(figsize=(12,9))
  plt.plot(range(1, 1001), cumulative_evr)
  plt.plot(Ks, markers, ls="", marker="o", label="points")

  for K, evr in zip(Ks, markers):
    plt.text(K, evr, str(round(evr, 2)))

  plt.title('explained_variance_ratio change with different Ks')
  plt.xlabel("Value of K")
  plt.ylabel("Cumulative explained_variance_ratio")
  plt.xticks(Ks, rotation = 90)
  plt.show() 