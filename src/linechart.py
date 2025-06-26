import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from itertools import combinations
from scipy import stats
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# 配置科学绘图样式
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.constrained_layout.use': True
})

# 分形维度和性能数据
D_values = np.arange(1, 10.5, 0.5)
Throughput = [30.80, 28.55, 28.32, 29.43, 31.65, 32.40, 32.82, 32.62, 34.70, 34.68,
              34.89, 35.95, 33.61, 35.94, 33.35, 34.52, 35.23, 35.55, 35.17]
Delay = [2.2459, 1.3111, 1.3022, 1.1768, 1.1285, 1.1068, 1.1277, 1.1084, 1.0234, 1.0623,
         1.0478, 1.0198, 1.0222, 0.9821, 1.0068, 1.0111, 1.0493, 1.0095, 1.0175]

# 生成分形点集的函数（已优化）
def generate_fractal_points(dimension, num_points=86, max_depth=5):
    points = []
    
    def generate_recursive(x, y, scale, depth):
        if depth == 0:
            return
        points.append((x, y))
        
        new_scale = scale * (0.25 ** (1/dimension))
        generate_recursive(x - new_scale, y - new_scale, new_scale, depth-1)
        generate_recursive(x + new_scale, y - new_scale, new_scale, depth-1)
        generate_recursive(x - new_scale, y + new_scale, new_scale, depth-1)
        generate_recursive(x + new_scale, y + new_scale, new_scale, depth-1)
    
    generate_recursive(0.5, 0.5, 0.5, max_depth)
    
    # 归一化并截取所需点数
    points = np.array(points[:num_points])
    x = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
    y = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
    return x, y

# 创建网络
def create_mesh_network(coords, max_distance=0.2):
    G = nx.Graph()
    for i, (x, y) in enumerate(coords):
        G.add_node(i, pos=(x, y))
    
    for i, j in combinations(range(len(coords)), 2):
        dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
        if dist <= max_distance:
            capacity = 1 / (1 + dist)
            G.add_edge(i, j, capacity=capacity, weight=1/capacity)
    return G

# 拓扑指标计算函数
def weighted_clustering(G):
    return nx.average_clustering(G, weight='capacity')

def spatial_entropy(coords, bins=10):
    hist, xedges, yedges = np.histogram2d(coords[:,0], coords[:,1], bins=bins)
    prob = hist / hist.sum()
    return entropy(prob.flatten() + 1e-10)  # 避免零概率

def global_efficiency(G):
    return nx.global_efficiency(G)

def algebraic_connectivity(G):
    return nx.algebraic_connectivity(G, weight='capacity')

def capacity_betweenness(G):
    bc = nx.betweenness_centrality(G, weight='capacity')
    return np.mean(list(bc.values()))

def synergy_factor(coords, G):
    # 几何距离矩阵
    geo_dist = squareform(pdist(coords))
    
    # 拓扑距离矩阵
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    topo_dist = nx.floyd_warshall_numpy(G, weight='weight')
    topo_dist[np.isinf(topo_dist)] = np.nan  # 处理不连通情况
    
    # 计算相关系数（仅有效值）
    valid_mask = ~np.isnan(topo_dist)
    return np.corrcoef(geo_dist[valid_mask], topo_dist[valid_mask])[0, 1]

# 计算所有拓扑指标
print("Calculating topological metrics...")
topo_metrics = {
    'Clustering': [],
    'SpatialEntropy': [],
    'Efficiency': [],
    'AlgebraicConn': [],
    'Betweenness': [],
    'Synergy': []
}

for i, D in enumerate(D_values):
    print(f"Processing D={D:.1f} ({i+1}/{len(D_values)})")
    x, y = generate_fractal_points(D)
    coords = np.column_stack([x, y])
    G = create_mesh_network(coords)
    
    topo_metrics['Clustering'].append(weighted_clustering(G))
    topo_metrics['SpatialEntropy'].append(spatial_entropy(coords))
    topo_metrics['Efficiency'].append(global_efficiency(G))
    topo_metrics['AlgebraicConn'].append(algebraic_connectivity(G))
    topo_metrics['Betweenness'].append(capacity_betweenness(G))
    topo_metrics['Synergy'].append(synergy_factor(coords, G))

# 创建科学可视化 - 使用更合理的布局
fig = plt.figure(figsize=(16, 20))
gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.2, 1, 1.5, 1.2],
                      width_ratios=[1, 1], hspace=0.35, wspace=0.25)

# 1. 拓扑指标与维度关系
ax1 = plt.subplot(gs[0, :])
colors = plt.cm.viridis(np.linspace(0, 1, len(topo_metrics)))
for i, (metric, values) in enumerate(topo_metrics.items()):
    ax1.plot(D_values, values, 'o-', color=colors[i], label=metric, 
             linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1)

ax1.axvline(x=5, color='r', linestyle='--', alpha=0.7, label='Critical Dimension (D=5)')
ax1.set_xlabel('Fractal Dimension (D)', fontsize=12)
ax1.set_ylabel('Normalized Metric Value', fontsize=12)
ax1.set_title('Topological Metrics Evolution with Fractal Dimension', fontsize=14, pad=15)
ax1.legend(ncol=3, loc='upper center', frameon=True, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(1, 10)
ax1.tick_params(axis='both', which='major', labelsize=10)

# 2. 性能与拓扑相关性矩阵
ax2 = plt.subplot(gs[1, 0])
# 组合所有指标
all_metrics = {**topo_metrics, 'Throughput': Throughput, 'Delay': Delay}
metric_names = list(all_metrics.keys())
corr_matrix = np.zeros((len(metric_names), len(metric_names)))

for i, m1 in enumerate(metric_names):
    for j, m2 in enumerate(metric_names):
        r, p = stats.pearsonr(all_metrics[m1], all_metrics[m2])
        corr_matrix[i, j] = r

# 简化标签
short_names = {
    'Clustering': 'Clust',
    'SpatialEntropy': 'SpatEnt',
    'Efficiency': 'Eff',
    'AlgebraicConn': 'AlgConn',
    'Betweenness': 'Between',
    'Synergy': 'Synergy',
    'Throughput': 'Thru',
    'Delay': 'Delay'
}

short_labels = [short_names.get(name, name) for name in metric_names]

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
            xticklabels=short_labels, yticklabels=short_labels,
            vmin=-1, vmax=1, ax=ax2, cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
ax2.set_title('Metric Correlation Matrix', fontsize=14, pad=12)
ax2.tick_params(axis='both', which='major', labelsize=9)

# 3. 3D相变图：维度-协同因子-吞吐量
ax3 = plt.subplot(gs[1, 1], projection='3d')
synergy = np.array(topo_metrics['Synergy'])
sc = ax3.scatter(D_values, synergy, Throughput, 
                c=Delay, cmap='plasma', s=50, alpha=0.9, depthshade=False)

# 添加临界区域
critical_mask = (D_values >= 4.5) & (D_values <= 5.5)
ax3.plot(D_values[critical_mask], synergy[critical_mask], Throughput[critical_mask], 
         'r-', linewidth=3, label='Critical Region', alpha=0.8)

ax3.set_xlabel('Fractal Dimension', fontsize=11)
ax3.set_ylabel('Geometry-Topology Synergy', fontsize=11)
ax3.set_zlabel('Throughput (Mbps)', fontsize=11)
ax3.set_title('Phase Transition in Performance-Synergy Space', fontsize=14, pad=12)
cbar = fig.colorbar(sc, ax=ax3, pad=0.1, shrink=0.7)
cbar.set_label('End-to-End Delay (ms)', fontsize=10)
ax3.legend(loc='upper left', fontsize=9)
ax3.tick_params(axis='both', which='major', labelsize=9)
ax3.view_init(elev=25, azim=-45)  # 调整视角

# 4. 拓扑结构演化示例 - 使用单独的子图网格
ax4 = plt.subplot(gs[2, :])
ax4.axis('off')  # 隐藏主轴

# 创建嵌套网格用于拓扑示例
inner_gs = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[2, :], 
                                          wspace=0.15, hspace=0.25)

sample_dims = [1, 3, 5, 7, 9]
n_samples = len(sample_dims)

for i, D in enumerate(sample_dims):
    idx = np.where(np.isclose(D_values, D))[0][0]
    x, y = generate_fractal_points(D)
    G = create_mesh_network(np.column_stack([x, y]))
    
    # 拓扑图
    ax_top = fig.add_subplot(inner_gs[0, i])
    pos = {i: (x[i], y[i]) for i in range(len(x))}
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray', ax=ax_top)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue', edgecolors='k', linewidths=0.5, ax=ax_top)
    ax_top.set_title(f'D={D}\nEff={topo_metrics["Efficiency"][idx]:.2f}', fontsize=10)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_aspect('equal')
    
    # 拓扑距离矩阵
    ax_dist = fig.add_subplot(inner_gs[1, i])
    topo_dist = nx.floyd_warshall_numpy(G, weight='weight')
    np.fill_diagonal(topo_dist, 0)
    im = ax_dist.imshow(topo_dist, cmap='viridis', origin='lower', aspect='auto')
    ax_dist.set_title(f'Synergy={topo_metrics["Synergy"][idx]:.2f}', fontsize=10)
    ax_dist.set_xticks([])
    ax_dist.set_yticks([])
    
    # 添加颜色条
    if i == 4:
        cax = fig.add_axes([ax_dist.get_position().x1 + 0.01, 
                            ax_dist.get_position().y0, 
                            0.01, 
                            ax_dist.get_position().height])
        plt.colorbar(im, cax=cax)
        cax.set_ylabel('Topological Distance', fontsize=8)

# 添加整体标题
fig.text(0.5, 0.92, 'Topology Evolution and Distance Matrices', 
         ha='center', va='center', fontsize=14)

# 5. 临界维度前后性能对比
ax5 = plt.subplot(gs[3, 0])
critical_idx = np.where(D_values >= 5)[0][0]  # D>=5的起始索引

# 计算性能提升百分比
throughput_gain = (np.mean(Throughput[critical_idx:]) - np.mean(Throughput[:critical_idx])) / np.mean(Throughput[:critical_idx]) * 100
delay_reduction = (np.mean(Delay[:critical_idx]) - np.mean(Delay[critical_idx:])) / np.mean(Delay[:critical_idx]) * 100

metrics_pre = {
    'Throughput': np.mean(Throughput[:critical_idx]),
    'Delay': np.mean(Delay[:critical_idx]),
    'Efficiency': np.mean(topo_metrics['Efficiency'][:critical_idx]),
    'Synergy': np.mean(topo_metrics['Synergy'][:critical_idx])
}

metrics_post = {
    'Throughput': np.mean(Throughput[critical_idx:]),
    'Delay': np.mean(Delay[critical_idx:]),
    'Efficiency': np.mean(topo_metrics['Efficiency'][critical_idx:]),
    'Synergy': np.mean(topo_metrics['Synergy'][critical_idx:])
}

x = np.arange(len(metrics_pre))
width = 0.35

# 创建更美观的柱状图
ax5.bar(x - width/2, list(metrics_pre.values()), width, label='D < 5', 
        alpha=0.85, color='#1f77b4', edgecolor='k', linewidth=0.7)
ax5.bar(x + width/2, list(metrics_post.values()), width, label='D ≥ 5', 
        alpha=0.85, color='#ff7f0e', edgecolor='k', linewidth=0.7)

ax5.set_xticks(x)
ax5.set_xticklabels(metrics_pre.keys(), fontsize=10)
ax5.set_ylabel('Metric Value', fontsize=11)
title_text = (f'Performance Enhancement at Critical Dimension\n'
              f'Throughput: +{throughput_gain:.1f}% | Delay: -{delay_reduction:.1f}%')
ax5.set_title(title_text, fontsize=12, pad=12)
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, axis='y', linestyle='--', alpha=0.4)
ax5.tick_params(axis='both', which='major', labelsize=9)

# 添加性能增益标注
for i, (pre, post) in enumerate(zip(metrics_pre.values(), metrics_post.values())):
    gain = post - pre if i != 1 else pre - post  # Delay是减少
    sign = '+' if gain > 0 else ''
    ax5.text(i, max(pre, post) + 0.05*max(pre, post), 
             f'{sign}{gain:.2f}', 
             ha='center', fontsize=8)

# 6. 理论模型验证
ax6 = plt.subplot(gs[3, 1])
# 几何-拓扑协同模型
x_synergy = np.array(topo_metrics['Synergy'])
y_throughput = np.array(Throughput)

# 拟合模型
coeff = np.polyfit(x_synergy, y_throughput, 2)
poly = np.poly1d(coeff)
x_fit = np.linspace(min(x_synergy), max(x_synergy), 100)

# 分形维度投影误差模型
proj_error = 0.5 * np.exp(-0.3 * np.array(D_values))  # 理论模型

# 创建散点图
scatter = ax6.scatter(x_synergy, y_throughput, c=D_values, cmap='viridis', 
                      s=70, alpha=0.9, edgecolors='w', linewidth=0.5)
ax6.plot(x_fit, poly(x_fit), 'r--', linewidth=2, 
         label=f'Quadratic Fit: $R^2$={np.corrcoef(y_throughput, poly(x_synergy))[0,1]**2:.3f}')

# 添加临界点标记
critical_points = (D_values >= 4.5) & (D_values <= 5.5)
ax6.scatter(x_synergy[critical_points], y_throughput[critical_points], 
            s=90, edgecolors='red', facecolors='none', linewidths=1.5, 
            label='Critical Region (D=4.5-5.5)', zorder=10)

ax6.set_xlabel('Geometry-Topology Synergy Factor', fontsize=11)
ax6.set_ylabel('Throughput (Mbps)', fontsize=11)
ax6.set_title('Synergy-Performance Relationship', fontsize=12, pad=12)
ax6.legend(loc='best', fontsize=9)
ax6.grid(True, linestyle='--', alpha=0.4)
ax6.tick_params(axis='both', which='major', labelsize=9)

# 添加投影误差曲线
ax6b = ax6.twinx()
ax6b.plot(x_synergy, proj_error, 'm-', linewidth=2, alpha=0.8)
ax6b.set_ylabel('Projection Error $\\varepsilon$', fontsize=11, color='m')
ax6b.tick_params(axis='y', labelcolor='m')
ax6b.tick_params(axis='both', which='major', labelsize=9)

# 添加数学公式注释
ax6.text(0.05, 0.15, 
         r'$\mathcal{P} = \beta_0 + \beta_1 \Gamma + \beta_2 \Gamma^2$' + '\n' +
         r'$\Gamma = \mathrm{corr}(\mathbf{D_g}, \mathbf{D_t})$' + '\n' +
         r'$\varepsilon \propto D^{-1/2}$',
         transform=ax6.transAxes, fontsize=11, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=5))

# 添加颜色条
cax = fig.add_axes([ax6.get_position().x1 + 0.01, 
                    ax6.get_position().y0, 
                    0.015, 
                    ax6.get_position().height])
cbar = fig.colorbar(scatter, cax=cax)
cbar.set_label('Fractal Dimension (D)', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# 整体调整和保存
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('fractal_network_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出关键统计结果
critical_idx = np.where(D_values >= 5)[0][0]
print(f"\nCritical Dimension Analysis (D ≥ 5):")
print(f"Throughput Improvement: {np.mean(Throughput[critical_idx:]) - np.mean(Throughput[:critical_idx]):.2f} Mbps ({throughput_gain:.1f}%)")
print(f"Delay Reduction: {np.mean(Delay[:critical_idx]) - np.mean(Delay[critical_idx:]):.3f} ms ({delay_reduction:.1f}%)")
print(f"Synergy Factor Increase: {np.mean(topo_metrics['Synergy'][critical_idx:]) - np.mean(topo_metrics['Synergy'][:critical_idx]):.3f}")
print(f"Algebraic Connectivity Increase: {np.mean(topo_metrics['AlgebraicConn'][critical_idx:]) - np.mean(topo_metrics['AlgebraicConn'][:critical_idx]):.2f}")

# 计算拓扑-性能相关性
synergy_throughput_r, synergy_throughput_p = stats.pearsonr(topo_metrics['Synergy'], Throughput)
efficiency_delay_r, efficiency_delay_p = stats.pearsonr(topo_metrics['Efficiency'], Delay)
print(f"\nCorrelation Analysis:")
print(f"Synergy vs Throughput: r = {synergy_throughput_r:.3f}, p = {synergy_throughput_p:.4f}")
print(f"Global Efficiency vs Delay: r = {efficiency_delay_r:.3f}, p = {efficiency_delay_p:.4f}")

# 输出临界点附近的详细数据
critical_data = D_values[(D_values >= 4.5) & (D_values <= 5.5)]
print(f"\nCritical Region Data (D=4.5-5.5):")
for D in critical_data:
    idx = np.where(D_values == D)[0][0]
    print(f"D={D:.1f}: Synergy={topo_metrics['Synergy'][idx]:.3f}, "
          f"Throughput={Throughput[idx]:.2f}, Delay={Delay[idx]:.4f}, "
          f"Efficiency={topo_metrics['Efficiency'][idx]:.3f}")