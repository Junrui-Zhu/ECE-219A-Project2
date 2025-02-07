"""
This file implements the function to load images data of flower photos dataset.
"""

import numpy as np
import os

def load_images_data(filename:str='./flowers_features_and_labels.npz'):
    if os.path.exists(filename):
        file = np.load(filename)
        f_all, y_all = file['f_all'], file['y_all']
        return f_all, y_all
    else:
        print("Please run help-code.ipynb to download flower_photos images and preprocess the data")
        return None, None