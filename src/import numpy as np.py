import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from matplotlib.colors import Normalize
from fractal_recursion import generate_fractal_points

# 1. 生成分形点
dimension = 2.5       # 可调参数，代表分形维度
n_points = 64         # 点的数量，可以与超立方体中节点数匹配，如 2^6 = 64
coordinates = generate_fractal_points(dimension, n_points)

# 2. 构建网状网络
max_distance = 0.3     # 距离阈值（你可能需要调参来保证图是连通的）
G_fractal = create_mesh_network_with_coordinates(coordinates, max_distance)

# 3. 生成 longest-matching traffic matrix
TM_fractal = longest_matching_tm(G_fractal)

# 4. 估算 throughput
throughput_fractal = estimate_throughput(G_fractal, TM_fractal)

# 5. 输出吞吐量
print(f"Fractal network throughput: {throughput_fractal:.4f}")

# 可视化分形网状网络
pos = nx.get_node_attributes(G_fractal, 'pos')
colors = get_node_colors(G_fractal)
plt.figure(figsize=(8, 8))
nx.draw(G_fractal, pos, node_color=colors, with_labels=False, node_size=50, edge_color='gray')
plt.title(f"Fractal Mesh Network (d={dimension}, Throughput={throughput_fractal:.4f})")
plt.show()


dimensions = np.linspace(1.5, 3.5, 10)
TH_fractal = []

for dim in dimensions:
    coords = generate_fractal_points(dim, n_points)
    Gf = create_mesh_network_with_coordinates(coords, max_distance)
    if not nx.is_connected(Gf):
        TH_fractal.append(0)
        continue
    TM = longest_matching_tm(Gf)
    th = estimate_throughput(Gf, TM)
    TH_fractal.append(th)

plt.plot(dimensions, TH_fractal, marker='o')
plt.xlabel("Fractal Dimension")
plt.ylabel("Throughput")
plt.title("Throughput vs Fractal Dimension")
plt.grid(True)
plt.show()
