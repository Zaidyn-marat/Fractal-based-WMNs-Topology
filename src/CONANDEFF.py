import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator

# 设置风格统一为 serif，顶刊风格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

# 分形维度
D_values = np.arange(1, 10.5, 0.5)

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
    x = (points[:, 0] - points[:, 0].min()) / points[:, 0].ptp()
    y = (points[:, 1] - points[:, 1].min()) / points[:, 1].ptp()
    return x, y

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

def global_efficiency(G):
    return nx.global_efficiency(G)

def algebraic_connectivity(G):
    return nx.algebraic_connectivity(G, weight='capacity')

# 计算两个指标
metrics = {
    'Efficiency': [],
    'Connectivity': []
}

print("Calculating metrics...")
for D in D_values:
    x, y = generate_fractal_points(D)
    coords = np.column_stack([x, y])
    G = create_mesh_network(coords)

    metrics['Efficiency'].append(global_efficiency(G))
    metrics['Connectivity'].append(algebraic_connectivity(G))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 3), facecolor='white')

palette = sns.color_palette("coolwarm", n_colors=len(metrics))
metric_names = list(metrics.keys())

# for i, ax in enumerate(axes):
#     name = metric_names[i]
#     values = metrics[name]

#     ax.plot(D_values, values, 'o-', color=palette[i], linewidth=2,
#             markersize=6, markeredgecolor='black')
    
#     ax.minorticks_on()  # 启用次要刻度
#     ax.tick_params(axis='both', which='both', direction='in', 
#                    top=True, right=True,  # 在顶部和右侧显示刻度
#                    labelsize=9)
    
#     # 更精细的刻度控制
#     ax.tick_params(which='minor', length=3, color='gray', width=0.5,labelsize=16)  # 次要刻度
#     ax.tick_params(which='major', length=3, color='black', width=0.8,labelsize=16)  # 主要刻度

#     ax.set_xlabel('Fractal dimension (D)', fontsize=16,color='black')
#     ax.set_ylabel(name, fontsize=16,color='black')
#     axes[0].yaxis.set_major_locator(MultipleLocator(0.5))
    
#     ax.tick_params(labelsize=16,color='black')
    
#     ax.grid(False)
    
for i, ax in enumerate(axes):
    name = metric_names[i]
    values = metrics[name]

    ax.plot(D_values, values, 'o-', color=palette[i], linewidth=2,
            markersize=6, markeredgecolor='black')

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', 
                   top=True, right=True, labelsize=9)
    ax.tick_params(which='minor', length=3, color='gray', width=0.5)
    ax.tick_params(which='major', length=3, color='black', width=0.8)
    ax.set_xlabel('Fractal dimension (D)', fontsize=16, color='black')
    ax.set_ylabel(name, fontsize=16, color='black')
    ax.tick_params(labelsize=16, color='black')
    ax.grid(False)

    # Y-axis tick control
    if name == 'Efficiency':
        ax.yaxis.set_major_locator(MultipleLocator(0.04))  # e.g., 0.65, 0.70, 0.75...
    elif name == 'Connectivity':
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))  # adaptive
    

plt.tight_layout()

plt.savefig("Topological_Metrics.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
