import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# import mplcyberpunk
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from fractal_recursion import generate_fractal_points

# 颜色映射方案
def get_node_colors(graph, cmap='coolwarm'):
    degrees = np.array([d for _, d in graph.degree()])
    norm_degrees = (degrees - degrees.min()) / (degrees.max() - 1e-5)
    return plt.get_cmap(cmap)(norm_degrees)

# 生成 2D 分形点


# 创建网络（带权重）
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

    # 创建白色背景画布
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')

    # 获取节点颜色
    node_colors = get_node_colors(G, cmap)

    # 画节点
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=30,
        edgecolors='black',  # 黑色边缘增强对比
        linewidths=0.8,
        alpha=0.9,
        ax=ax
    )

    # 画边
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#005f99',  # 浅蓝色边缘适配白色背景
        width=0.5,
        alpha=0.7,
        ax=ax
    )

    # 发光效果（适配白色背景）
    # mplcyberpunk.make_lines_glow(ax=ax)

    # 关闭坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Fractal Mesh Network", fontsize=12, color='black', pad=10)
    
    # 保存或展示
    # if save_path:
    #     plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    #     print(f"图像已保存至: {save_path}")

    plt.show()

# 运行主程序
if __name__ == "__main__":
    D = 2
    num_points = 1000
    max_distance = 0.15

    # 生成分形点
    x, y = generate_fractal_points(D, num_points)
    coordinates = list(zip(x, y))

    # 创建网络
    G = create_mesh_network_with_coordinates(coordinates, max_distance)

    # 可视化
    visualize_fractal_network(G, save_path="art.pdf", cmap='coolwarm')
