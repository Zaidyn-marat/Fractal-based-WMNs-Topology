import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import re


# 设置图形风格
sns.set(style="whitegrid", font_scale=1.2)

# 固定节点数
n_nodes = 85
seed = 42

# 保证生成连通的 ER 网络
def generate_connected_er_graph(n_nodes, avg_degree, seed=None):
    p = avg_degree / (n_nodes - 1)
    rng = np.random.default_rng(seed)
    while True:
        G = nx.erdos_renyi_graph(n_nodes, p, seed=int(rng.integers(100000)))
        if nx.is_connected(G):
            return G

# 保证生成连通的 WS 网络
def generate_connected_ws_graph(n_nodes, k, p, seed=None):
    rng = np.random.default_rng(seed)
    while True:
        G = nx.watts_strogatz_graph(n_nodes, k, p, seed=int(rng.integers(100000)))
        if nx.is_connected(G):
            return G

# ER 参数设置
avg_degree_er = 4.5  # 稍微提高，提升连通性
G_er = generate_connected_er_graph(n_nodes, avg_degree_er, seed)

# WS 参数设置
k_ws = 4
p_ws = 0.1
G_ws = generate_connected_ws_graph(n_nodes, k_ws, p_ws, seed)

# BA 网络
m_ba = 2
G_ba = nx.barabasi_albert_graph(n_nodes, m_ba, seed=seed)

# 网络集合
networks = {
    "Erdős–Rényi (ER)": G_er,
    "Watts–Strogatz (WS)": G_ws,
    "Barabási–Albert (BA)": G_ba
}

# 分析函数
def analyze_network(G):
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    clustering = nx.average_clustering(G)
    try:
        path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        path_length = np.nan
    return avg_degree, clustering, path_length

# 拓扑特性表
results = []
for name, G in networks.items():
    avg_deg, clust, path_len = analyze_network(G)
    results.append({
        "Model": name,
        "Avg Degree": round(avg_deg, 2),
        "Clustering Coefficient": round(clust, 4),
        "Avg Path Length": round(path_len, 4) if not np.isnan(path_len) else "N/A"
    })

df_results = pd.DataFrame(results)

# # 可视化：制度分布
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# for ax, (name, G) in zip(axes, networks.items()):
#     degrees = [d for _, d in G.degree()]
#     sns.histplot(degrees, bins=range(min(degrees), max(degrees)+2), kde=False,
#                  ax=ax, color="skyblue", edgecolor="black")
#     ax.set_title(f"{name}\nDegree Distribution")
#     ax.set_xlabel("Degree")
#     ax.set_ylabel("Count")
#     ax.grid(True)

# plt.tight_layout()
# plt.show()

# 输出网络结构可视化
plt.figure(figsize=(18, 6))
for i, (name, G) in enumerate(networks.items(), 1):
    plt.subplot(1, 3, i)
    pos = nx.spring_layout(G, seed=seed)  # 稳定布局
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("网络拓扑分析结果：")
print(df_results.to_string(index=False))


output_dir = r"C:\DESKTOP_MARAT\Fractal Recursion Topology\Code\basline_topology"
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

saved_files = []

for name, G in networks.items():
    pos = nx.spring_layout(G, seed=seed)
    coords = np.array([pos[n] for n in sorted(G.nodes())])
    
    safe_name = name.lower()
    safe_name = re.sub(r"[–\s\(\)]+", "_", safe_name)
    safe_name = re.sub(r"[^a-z0-9_]", "", safe_name)
    
    filename = f"{safe_name}_coordinates.txt"
    filepath = os.path.join(output_dir, filename)
    
    np.savetxt(filepath, coords, fmt="%.6f", delimiter="\t",
                header=f"Coordinates for {name}", comments="", encoding="utf-8")
    
    saved_files.append(filepath)

print("以下文件已保存为 NS3 输入坐标格式：")
for f in saved_files:
    print(f)

#------------------------------------------------------------------------------------
# Runtime was reset; re-import necessary modules and re-define everything


# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import pandas as pd
# import os
# import re

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
# })

# # === 颜色映射 ===
# def get_node_colors(graph, cmap='coolwarm'):
#     degrees = np.array([d for _, d in graph.degree()])
#     norm_degrees = (degrees - degrees.min()) / (degrees.max() - 1e-5)
#     return plt.get_cmap(cmap)(norm_degrees)

# # === 可视化风格函数 ===
# def visualize_styled_network(G, title="Network", cmap='coolwarm'):
#     pos = nx.spring_layout(G, seed=42)  # 稳定布局
#     fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor='white')
#     node_colors = get_node_colors(G, cmap)

#     nx.draw_networkx_nodes(
#         G, pos,
#         node_color=node_colors,
#         node_size=30,
#         edgecolors='black',
#         linewidths=0.8,
#         alpha=0.9,
#         ax=ax
#     )
#     nx.draw_networkx_edges(
#         G, pos,
#         edge_color='#005f99',
#         width=0.5,
#         alpha=0.7,
#         ax=ax
#     )
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(title, fontsize=18, color='black', pad=10)


# # === 网络生成函数 ===
# def generate_connected_er_graph(n_nodes, avg_degree, seed=None):
#     p = avg_degree / (n_nodes - 1)
#     rng = np.random.default_rng(seed)
#     while True:
#         G = nx.erdos_renyi_graph(n_nodes, p, seed=int(rng.integers(100000)))
#         if nx.is_connected(G):
#             return G

# def generate_connected_ws_graph(n_nodes, k, p, seed=None):
#     rng = np.random.default_rng(seed)
#     while True:
#         G = nx.watts_strogatz_graph(n_nodes, k, p, seed=int(rng.integers(100000)))
#         if nx.is_connected(G):
#             return G

# # === 网络创建 ===
# n_nodes = 85
# seed = 42
# G_er = generate_connected_er_graph(n_nodes, avg_degree=4.5, seed=seed)
# G_ws = generate_connected_ws_graph(n_nodes, k=4, p=0.1, seed=seed)
# G_ba = nx.barabasi_albert_graph(n_nodes, m=2, seed=seed)

# # === 可视化调用 ===
# visualize_styled_network(G_er, title="Erdős–Rényi (ER)")
# visualize_styled_network(G_ws, title="Watts–Strogatz (WS)")
# visualize_styled_network(G_ba, title="Barabási–Albert (BA)")

# # 合并三个网络为一个横排图

# fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')

# graphs = [G_er, G_ws, G_ba]
# titles = ["Erdős–Rényi (ER)", "Watts–Strogatz (WS)", "Barabási–Albert (BA)"]

# for ax, G, title in zip(axes, graphs, titles):
#     pos = nx.spring_layout(G, seed=42)
#     node_colors = get_node_colors(G)

#     nx.draw_networkx_nodes(
#         G, pos,
#         node_color=node_colors,
#         node_size=30,
#         edgecolors='black',
#         linewidths=0.8,
#         alpha=0.9,
#         ax=ax
#     )
#     nx.draw_networkx_edges(
#         G, pos,
#         edge_color='#005f99',
#         width=0.5,
#         alpha=0.7,
#         ax=ax
#     )

#     ax.set_title(title, fontsize=20, color='black', pad=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout()
# plt.show()
# plt.savefig("basline_network.pdf", dpi=600, bbox_inches='tight', facecolor='white')
