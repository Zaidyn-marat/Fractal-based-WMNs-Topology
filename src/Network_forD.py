import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mplcyberpunk
from itertools import combinations
from matplotlib.colors import Normalize
from adjustText import adjust_text
from fractal_recursion import generate_fractal_points
plt.rcParams.update({
    "font.family": "serif",  # 改成 serif
    "font.serif": ["Libertinus Serif", "Linux Libertine", "Times New Roman", "Times"],
})


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
            G.add_edge(i, j)
    return G

def visualize_fractal_network(G, ax, cmap='coolwarm'):
    
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = get_node_colors(G, cmap)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, edgecolors='black', linewidths=0.8, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#005f99', width=0.5, alpha=0.7, ax=ax)
    mplcyberpunk.make_lines_glow(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"D = {D}", color='black', pad=5)
    # ax.set_aspect('equal')

import matplotlib.gridspec as gridspec

# 🌟 画布 & 布局
fig = plt.figure(figsize=(9, 9), facecolor='white')
gs = fig.add_gridspec(3, 3, wspace=0.05, hspace=0.15)

for i, D in enumerate(range(1, 10)):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    
    x, y = generate_fractal_points(D, 100)
    coords = list(zip(x, y))
    G = create_mesh_network_with_coordinates(coords, 0.5)

    visualize_fractal_network(G, ax)
    ax.set_aspect('equal', adjustable='box')  # 正方形坐标轴
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"D = {D}", fontsize=11, pad=10)
    ax.axis('off')

plt.savefig("Network.pdf", dpi=600, bbox_inches='tight')
plt.show()

# 性能指标数据


