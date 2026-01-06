import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# import mplcyberpunk
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from fractal_recursion import generate_fractal_points


def get_node_colors(graph, cmap='coolwarm'):
    degrees = np.array([d for _, d in graph.degree()])
    norm_degrees = (degrees - degrees.min()) / (degrees.max() - 1e-5)
    return plt.get_cmap(cmap)(norm_degrees)

def create_mesh_network_with_coordinates(coordinates, max_distance):
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)
    for i, j in combinations(range(len(coordinates)), 2):
        distance = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        if distance <= max_distance:
            capacity = 1 / (1 + distance)
            G.add_edge(i, j, capacity=capacity, weight=1 / capacity)
    return G

def visualize_fractal_network(G, save_path=None, cmap='coolwarm'):
    pos = nx.get_node_attributes(G, 'pos')

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')

    node_colors = get_node_colors(G, cmap)

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=30,
        edgecolors='black',  
        linewidths=0.8,
        alpha=0.9,
        ax=ax
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color='#005f99', 
        width=0.5,
        alpha=0.7,
        ax=ax
    )

    # mplcyberpunk.make_lines_glow(ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Fractal Mesh Network", fontsize=12, color='black', pad=10)


    plt.show()

if __name__ == "__main__":
    D = 2
    num_points = 1000
    max_distance = 0.15

    x, y = generate_fractal_points(D, num_points)
    coordinates = list(zip(x, y))
    G = create_mesh_network_with_coordinates(coordinates, max_distance)

    visualize_fractal_network(G, save_path="art.pdf", cmap='coolwarm')

