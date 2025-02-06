import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import clip
import torch
from tqdm import tqdm
from scipy.special import softmax

# load csv file and image paths to construct pokedex, use type_to_load=None to load all types, else use a list of types 1 to load
def construct_pokedex(csv_path='Pokemon.csv', image_dir='./images/', type_to_load=None):
    pokedex = pd.read_csv(csv_path)
    image_paths = []

    for pokemon_name in pokedex["Name"]:
        imgs = glob(f"{image_dir}/{pokemon_name}/0.jpg")
        if len(imgs) > 0:
            image_paths.append(imgs[0])
        else:
            image_paths.append(None)

    pokedex["image_path"] = image_paths
    pokedex = pokedex[pokedex["image_path"].notna()].reset_index(drop=True)

    # only keep pokemon with distinct id
    ids, id_counts = np.unique(pokedex["ID"], return_counts=True)
    ids, id_counts = np.array(ids), np.array(id_counts)
    keep_ids = ids[id_counts == 1]

    pokedex = pokedex[pokedex["ID"].isin(keep_ids)].reset_index(drop=True)
    pokedex["Type2"] = pokedex["Type2"].str.strip()
    if type_to_load is not None:
        pokedex = pokedex[pokedex["Type1"].isin(type_to_load)].reset_index(drop=True)
    return pokedex

# load clip model
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess, device

# inference clip model on a list of image path
def clip_inference_image(model, preprocess, image_paths, device):
    image_embeddings = []
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            img = Image.open(img_path)
            img_preprocessed = preprocess(img).unsqueeze(0).to(device)
            image_embedding = model.encode_image(img_preprocessed).detach().cpu().numpy()
            image_embeddings += [image_embedding]
            
    image_embeddings = np.concatenate(image_embeddings, axis=0)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=-1, keepdims=True)
    return image_embeddings

# inference clip model on a list of texts
def clip_inference_text(model, preprocess, texts, device):
    with torch.no_grad():
        text_embeddings = model.encode_text(clip.tokenize(texts).to(device)).detach().cpu().numpy()
    text_embeddings /= np.linalg.norm(text_embeddings, axis=-1, keepdims=True)
    return text_embeddings

# compute similarity of texts to each image
def compute_similarity_text_to_image(image_embeddings, text_embeddings):
    similarity = softmax((100.0 * image_embeddings @ text_embeddings.T), axis=-1)
    return similarity

# compute similarity of iamges to each text
def compute_similarity_image_to_text(image_embeddings, text_embeddings):
    similarity = softmax((100.0 * image_embeddings @ text_embeddings.T), axis=0)
    return similarity

def plot_images_horizontally(paths, titles, main_title, save_path=None, figsize=(15, 5)):
    """
    Plot images horizontally with titles and save the plot if a save path is provided.

    Parameters:
    - paths (list of str): List of paths to image files.
    - titles (list of str): List of titles for each image.
    - save_path (str, optional): Path to save the final plot (default is None, meaning no saving).
    - main_title (str, optional): The main title for the entire figure.
    - figsize (tuple): Size of the figure (default is (15, 5)).
    """
    if len(paths) != len(titles):
        raise ValueError("The number of paths must match the number of titles.")

    # Create a figure to contain subplots
    fig, axes = plt.subplots(1, len(paths), figsize=figsize)

    # Loop through the axes and images to plot them
    for i, ax in enumerate(axes):
        img = mpimg.imread(paths[i])  # Read image from path
        ax.imshow(img)  # Display the image
        ax.set_title(titles[i])  # Set the title
        ax.axis('off')  # Hide the axis

    # Set the main title
    plt.suptitle(main_title, fontsize=16, fontweight='bold')

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the main title

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot
        plt.show()

if __name__ == '__main__':
    pokedex = construct_pokedex('Pokemon.csv', 'pokemon_images')
    images_path = pokedex['image_path']
    model, preprocess, device = load_clip_model()
    # texts = ['Bug', 'Fire', 'Grass']
    texts = ['Dark', 'Dragon']
    texts_embeddings = clip_inference_text(model, preprocess, texts, device)
    images_embeddings = clip_inference_image(model, preprocess, images_path, device)
    similarity = compute_similarity_image_to_text(images_embeddings, texts_embeddings)

    top_5_indices = np.argsort(similarity, axis=0)[-5:][::-1].T  # Indices of top 5 values (sorted)

    for i in range(len(texts)):
        text = texts[i]
        titles = []
        paths = []
        for j in range(5):
            indice = top_5_indices[i][j]
            name, path = pokedex['Name'].iloc[indice], pokedex['image_path'].iloc[indice]
            type1, type2 = pokedex['Type1'].iloc[indice], pokedex['Type2'].iloc[indice]
            image_path = pokedex['image_path'].iloc[indice]
            title = name + ' ' + type1 + ' ' + type2
            titles.append(title)
            paths.append(image_path)
        plot_images_horizontally(paths, titles, text)
    