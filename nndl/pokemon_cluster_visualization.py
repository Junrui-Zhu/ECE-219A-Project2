"""
This file performs UMAP projection and visualization of pokemon image features.
By running the file, you will get answer for question 28
"""
from pokemon_image_select import *
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

def df_filter(df, types):
    indices = []
    for i in range(df.shape[0]):
        if df['Type1'].iloc[i] in types: #  or df['Type2'].iloc[i] in types
            indices.append(i)
    df_filtered = df[['Name', 'Type1', 'image_path']].iloc[indices]
    return df_filtered

# Use TSNE to project CLIP embeddings to 2D space
def umap_projection(image_embeddings, n_neighbors=15, min_dist=0.1, metric='cosine'):
    distance_matrix = np.zeros((image_embeddings.shape[0], image_embeddings.shape[0]))
    for i in range(image_embeddings.shape[0]):
        for j in range(image_embeddings.shape[0]):
            if i == j:
                distance_matrix[i, j] = 1
            else:
                distance_matrix[i, j] = np.dot(image_embeddings[i], image_embeddings[j])
    distance_matrix = 1 - distance_matrix
    reducer = TSNE(n_components=2, metric="precomputed", init="random", random_state=42)
    visualization_data = reducer.fit_transform(distance_matrix)
    return visualization_data

if __name__ == "__main__":
    pokedex = construct_pokedex('Pokemon.csv', 'pokemon_images')
    types = ['Bug', 'Fire', 'Grass']
    pokedex = df_filter(pokedex, types)
    images_path = pokedex['image_path']
    model, preprocess, device = load_clip_model()
    images_embeddings = clip_inference_image(model, preprocess, images_path, device)
    X_2d = umap_projection(images_embeddings)
    pokedex['x'] = X_2d[:, 0]
    pokedex['y'] = X_2d[:, 1]

    # Create hover text with Pokémon name and types
    pokedex["hover_text"] = pokedex.apply(
        lambda row: f"{row['Name']} ({row['Type1']})", axis=1
    )

    # Create scatter plot
    fig = px.scatter(
        pokedex,
        x="x",
        y="y",
        color="Type1",  # Color based on primary type
        hover_name="hover_text",  # Display Pokémon name & types on hover
        title="t-SNE Visualization of Pokémon Clusters (Bug, Fire, Grass)",
        labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
        width=900,
        height=600
    )

    # Show the plot
    fig.show()
