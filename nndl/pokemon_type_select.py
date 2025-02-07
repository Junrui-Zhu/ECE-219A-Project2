"""
This file predicts most related types for selected pokemon images.
By running the file, you will get answer for question 27
"""
from pokemon_image_select import *

def get_all_types(pokedex):
    type1_list = pokedex['Type1'].to_list()
    type2_list = pokedex['Type1'].to_list()
    type_list = list(set(type1_list + type2_list))
    return type_list

def plot_images(paths, titles, main_title, save_path=None, figsize=(15, 5)):
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
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()

    # Loop through the axes and images to plot them
    for i, ax in enumerate(axes):
        img = mpimg.imread(paths[i])  # Read image from path
        ax.imshow(img)  # Display the image
        ax.set_title(titles[i], fontsize=10)  # Set the title
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

if __name__ == "__main__":
    pokedex = construct_pokedex('Pokemon.csv', 'pokemon_images')
    texts = get_all_types(pokedex)
    pokedex = pokedex.sample(n=10, random_state=42)
    images_path = pokedex['image_path'].to_list()
    model, preprocess, device = load_clip_model()
    texts_embeddings = clip_inference_text(model, preprocess, texts, device)
    images_embeddings = clip_inference_image(model, preprocess, images_path, device)
    similarity = compute_similarity_text_to_image(images_embeddings, texts_embeddings)
    top_5_indices = np.argsort(similarity, axis=1)[:, -5:][:, ::-1] # Indices of top 5 values (sorted)

    titles = []
    for i in range(len(images_path)):
        name, path = pokedex['Name'].iloc[i], pokedex['image_path'].iloc[i]
        type1, type2 = pokedex['Type1'].iloc[i], pokedex['Type2'].iloc[i]
        image_path = pokedex['image_path'].iloc[i]
        title = name + '\n' + type1 + ' ' + type2
        for j in range(5):
            indice = top_5_indices[i][j]
            title = title + '\n' + f'{texts[indice]} {similarity[i][indice]:.4f}'
        titles.append(title)
    plot_images(images_path, titles, "Selected Pokemons and Predicted Types", figsize=(15, 6))