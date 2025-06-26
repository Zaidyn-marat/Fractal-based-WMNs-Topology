import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

# 设置绘图风格
plt.rcParams.update({
    "font.family": "serif",  # 改成 serif
    "font.serif": ["Libertinus Serif", "Linux Libertine", "Times New Roman", "Times"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 14,
    "figure.titlesize": 13
})


# 分形维度
D_values = np.arange(1, 10.5, 0.5)

# 生成分形点集
def generate_fractal_points(dimension, num_points=86, max_depth=5):
    points = []
    def generate_recursive(x, y, scale, depth):
        if depth == 0: return
        points.append((x, y))
        new_scale = scale * (0.25 ** (1/dimension))
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                generate_recursive(x + dx*new_scale, y + dy*new_scale, new_scale, depth-1)
    generate_recursive(0.5, 0.5, 0.5, max_depth)
    points = np.array(points[:num_points])
    x = (points[:, 0] - points[:, 0].min()) / (points[:, 0].ptp())
    y = (points[:, 1] - points[:, 1].min()) / (points[:, 1].ptp())
    return x, y

# 构建网络
def create_mesh_network(coords, max_distance=0.5):
    G = nx.Graph()
    for i, (x, y) in enumerate(coords):
        G.add_node(i, pos=(x, y))
    for i, j in combinations(range(len(coords)), 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist <= max_distance:
            capacity = 1 / (1 + dist)
            G.add_edge(i, j, capacity=capacity, weight=1/capacity)
    return G

# 指标函数
def weighted_clustering(G):
    return nx.average_clustering(G, weight='capacity')

def global_efficiency(G):
    return nx.global_efficiency(G)

def algebraic_connectivity(G):
    return nx.algebraic_connectivity(G, weight='capacity')

def capacity_betweenness(G):
    bc = nx.betweenness_centrality(G, weight='capacity')
    return np.mean(list(bc.values()))

# 计算拓扑指标
metrics = {
    # 'Clustering': [],
    'Efficiency': [],
    # 'Betweenness': [],
    'Connectivity': []
}

print("Calculating metrics...")
for D in D_values:
    x, y = generate_fractal_points(D)
    coords = np.column_stack([x, y])
    G = create_mesh_network(coords)

    # metrics['Clustering'].append(weighted_clustering(G))
    metrics['Efficiency'].append(global_efficiency(G))
    # metrics['Betweenness'].append(capacity_betweenness(G))
    metrics['Connectivity'].append(algebraic_connectivity(G))


# 绘图
# palette = sns.color_palette("coolwarm", n_colors=len(metrics))


fig, axes = plt.subplots(1, 2, figsize=(9, 3), facecolor='white')

palette = sns.color_palette("coolwarm", n_colors=4)
metric_names = list(metrics.keys())

for i, ax in enumerate(axes.flatten()):
    name = metric_names[i]
    values = metrics[name]
    

    ax.plot(D_values, values, 'o-', color=palette[i], linewidth=2.5,
            markersize=7, markeredgecolor='black')

    ax.set_xlabel('Fractal Dimension (D)', fontsize=14)
    ax.set_ylabel(name, fontsize=14)
    ax.grid(False)
    # ax.set_title(f"{name} vs. Fractal Dimension", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
# plt.minorticks_on()
# plt.tick_params(axis='both', direction='in', which='both', top=True, right=True)
plt.tight_layout()
# plt.savefig("Performance_Metrics_Subplots.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

