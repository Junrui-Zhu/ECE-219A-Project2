# Project 2: Data Representations and Clustering

## Table of Contents
1. [How to Run the Code](#how-to-run-the-code)
2. [Dependencies](#dependencies)
3. [File Structure](#file-structure)
4. [Authors](#authors)

---

## How to Run the Code
Necessary steps to run the code:

1. Navigate to the project directory:
   ```bash
   cd ECE-219A-Project2
   ```

2. Install the required dependencies:
   ```bash
   conda env create -f environment.yml
   ```

3. Each python script performs a certain algorithm, performs a certain task, or answers several questions by running it. Code comments sre provided at the beginning of each python script. For example:
 Question 1: all_tfidf_labels.py
 Question 2,3: k_mean.py
 Question 4: pca_svd.py
 Question 5,6,7: nmf_svd_kmeans.py
 Question 8,9: visualize_kmean.py
 Question 10: 20_cluster.py
 Question 11,12,13: umap_reduction.py
 Question 14: agg_cluster.py
 Question 15,16: hdbscan_cluster.py
 Question 17: text_dim_reduction_and_cluster.py
 Question 21: helper-code.ipynb
 Question 23: tSNE_visualization.py
 Question 24: images_dim_reduction_and_cluster.py
 Question 25: torch_model.py
 Question 26: pokemon_image_select.py
 Question 27: pokemon_type_select.py
 Question 28: pokemon_cluster_visualization.py
---

## Dependencies
This project requires the following libraries and tools. Use the provided environment.yml to install these packages:

- Python <3.13
- NumPy
- pandas
- matplotlib
- scikit-learn
- umap-learn
- plotly
- scipy
- hdbscan
- torch
- clip
- sentence_transformers
---

## File Structure
This Project is organized as follows:
```bash
├── pokemon_images/...                
├── flower_photos/...   
├── nndl/                 # Source code
│   ├── 20_cluster.py         
│   ├── agg_cluster.py
│   ├── all_tfidf_labels.py
│   ├── hdbscan_cluster.py
│   ├── images_dim_reduction_and_cluster.py
│   ├── k_mean.py
│   ├── load_images.py
│   ├── nmf_svd_kmeans.py
│   ├── nmf.py
│   ├── pca_svd.py
│   ├── plotmat.py
│   └── pokemon_cluster_visualization.py
│   └── pokemon_image_select.py 
│   └── pokemon_type_select.py 
│   └── text_dim_reduction_and_cluster.py 
│   └── torch_model.py 
│   └── tSNE_visualization.py 
│   └── umap_reduction.py 
│   └── visualize_kmean.py 
├── ECE219-proj2.pdf  #report in pdf format
├── flowers_features_and_labels.npz 
├── helper-code.ipynb
├── pokedex_helper.ipynb
├── Pokemon.csv 
├── README.md            # Documentation
└── environment.yml      # Python dependencies file
```

Notes:
- All source code is located in the `nndl/` folder.
- Use .ipynb files to download and use the dataset.

---

## Authors

This project was collaboratively developed by the following contributors:

| Name                | UID                       |  Contact               |
|---------------------|---------------------------|------------------------|
| **LiWei Tan**       | 206530851                 | 962439602@qq.com       |
| **TianXiang Xing**  | 006530550                 | andrewxing43@g.ucla.edu|
| **Junrui Zhu**      | 606530444                 | zhujr24@g.ucla.edu     |

---
