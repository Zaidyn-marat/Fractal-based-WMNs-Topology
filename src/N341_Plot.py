# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# import mplcyberpunk
# from itertools import combinations
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# from fractal_recursion import generate_fractal_points
# import matplotlib.gridspec as gridspec

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
# })

# # 颜色映射方案
# def get_node_colors(graph, cmap='coolwarm'):
#     degrees = np.array([d for _, d in graph.degree()])
#     norm_degrees = (degrees - degrees.min()) / (degrees.max() - 1e-5)
#     return plt.get_cmap(cmap)(norm_degrees)

# class FractalNetworkFlow:
#     def __init__(self, dimension=1, connect_radius=0.2, cmap='coolwarm'):
#         self.dimension = dimension
#         self.connect_radius = connect_radius
#         self.cmap = cmap

#     def generate_network(self, num_points):
#         """生成分形网络"""
#         # x, y = generate_fractal_points(self.dimension, num_points + 1)

#         if num_points == 1:
#             x, y = [0.5], [0.5]
#         else:
#         # 其他情况正常生成分形点
#             x, y = generate_fractal_points(self.dimension, num_points + 1)
        
#         self.points = list(zip(x, y))

#         self.G = nx.Graph()
#         for i, (x, y) in enumerate(self.points):
#             self.G.add_node(i, pos=(x, y))

#         if num_points > 1:
#             for (i, j) in combinations(self.G.nodes, 2):
#                 p1 = np.array(self.G.nodes[i]['pos'])
#                 p2 = np.array(self.G.nodes[j]['pos'])
#                 distance = np.linalg.norm(p1 - p2)
#                 if distance < self.connect_radius:
#                     self.G.add_edge(i, j)

#     def visualize_flow(self, num_points_list, save_path=None):
#         """可视化分形网络生成流程图"""
        
#         fig = plt.figure(figsize=(9, 6), facecolor='white')
#         # 使用3列布局，第一行3个子图，第二行2个子图（居中）
#         gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
#         # 第一行的子图位置
#         ax1 = plt.subplot(gs[0, 0])  # N=1
#         ax2 = plt.subplot(gs[0, 1])  # N=5
#         ax3 = plt.subplot(gs[0, 2])  # N=21
        
#         # 第二行的子图位置 - 使用跨列来居中
#         ax4 = plt.subplot(gs[1, 0:2])  # N=85 (跨越前两列)
#         ax5 = plt.subplot(gs[1, 1:3])  # N=341 (跨越后两列)
        
#         axs = [ax1, ax2, ax3, ax4, ax5]
#         positions = [1, 5, 21, 85, 341]
        
#         for idx, (ax, num_points) in enumerate(zip(axs, positions)):
#             self.generate_network(num_points)
#             pos = nx.get_node_attributes(self.G, 'pos')
#             node_colors = get_node_colors(self.G, self.cmap)
            
#             nx.draw_networkx_nodes(
#                 self.G, pos,
#                 node_color=node_colors,
#                 node_size=15 if num_points == 341 else 20,
#                 edgecolors='black',
#                 linewidths=0.8,
#                 alpha=0.9,
#                 ax=ax
#             )
            
#             if num_points > 1:
#                 edge_width = 0.01 if num_points == 341 else 0.5
#                 nx.draw_networkx_edges(
#                     self.G, pos,
#                     edge_color='#005f99',
#                     width=0.004,
#                     alpha=0.9,
#                     ax=ax
#                 )
            
#             ax.set_title(f"N = {num_points}", fontsize=12, color='black', pad=10)
#             ax.set_xlim(-0.05, 1.05)
#             ax.set_ylim(-0.05, 1.05)
#             ax.set_aspect('equal')
#             ax.axis('off')
            
#             mplcyberpunk.make_lines_glow(ax=ax)
            
#             # 仅对341生成放大子图
#             if num_points == 341:
#                 sub_nodes = [n for n in self.G.nodes if pos[n][0] < 1 and pos[n][1] < 1]
#                 subG = self.G.subgraph(sub_nodes)
#                 sub_pos = {n: pos[n] for n in sub_nodes}
                
#                 axin = inset_axes(ax, width="60%", height="60%", 
#                                   bbox_to_anchor=(-0.5, -0.5, 1, 1),
#                                   bbox_transform=ax.transAxes, loc='lower left')
                
#                 nx.draw_networkx_nodes(
#                     subG, sub_pos,
#                     node_color='white',
#                     node_size=30,
#                     edgecolors='black',
#                     linewidths=1,
#                     alpha=0.9,
#                     ax=axin
#                 )
                
#                 nx.draw_networkx_edges(
#                     subG, sub_pos,
#                     edge_color='b',
#                     width=0.01,
#                     alpha=1,
#                     ax=axin
#                 )
                
#                 axin.set_xlim(-0.03, 0.2)
#                 axin.set_ylim(-0.03, 0.2)
#                 axin.set_xticks([])
#                 axin.set_yticks([])
#                 axin.set_aspect('equal')
                
#                 # mark_inset(ax, axin, loc1=2, loc2=3, fc="none", ec="black", lw=1)
        
#         # 调整子图间距
#         plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
#         if save_path:
#             plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
#             print(f"图像已保存至: {save_path}")
        
#         plt.show()

# if __name__ == "__main__":
#     fractal_flow = FractalNetworkFlow(dimension=1.6, connect_radius=0.4, cmap='coolwarm')
#     num_points_list = [1, 5, 21, 85, 341]
#     fractal_flow.visualize_flow(num_points_list, save_path="TEST.pdf")

#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mplcyberpunk
from itertools import combinations
from fractal_recursion import generate_fractal_points  # 确保此模块可用

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.serif": ["Libertinus Serif", "Linux Libertine", "Times New Roman", "Times"],
})

def get_node_colors(graph, cmap='coolwarm'):
    degrees = np.array([d for _, d in graph.degree()])
    norm_degrees = (degrees - degrees.min()) / (degrees.max() - 1e-5)
    return plt.get_cmap(cmap)(norm_degrees)

class FractalNetworkFlow:
    def __init__(self, dimension=1.6, connect_radius=0.4, cmap='coolwarm'):
        self.dimension = dimension
        self.connect_radius = connect_radius
        self.cmap = cmap

    def generate_network(self, num_points):
        if num_points == 1:
            x, y = [0.5], [0.5]
        else:
            x, y = generate_fractal_points(self.dimension, num_points + 1)
        self.points = list(zip(x, y))
        self.G = nx.Graph()
        for i, (x, y) in enumerate(self.points):
            self.G.add_node(i, pos=(x, y))
        if num_points > 1:
            for (i, j) in combinations(self.G.nodes, 2):
                p1 = np.array(self.G.nodes[i]['pos'])
                p2 = np.array(self.G.nodes[j]['pos'])
                distance = np.linalg.norm(p1 - p2)
                if distance < self.connect_radius:
                    self.G.add_edge(i, j)

    def visualize_flow(self, num_points_list, save_path=None):
        fig, axs = plt.subplots(1, len(num_points_list), figsize=(13, 3.8), facecolor='white')
        if len(num_points_list) == 1:
            axs = [axs]

        for ax, num_points in zip(axs, num_points_list):
            self.generate_network(num_points)
            pos = nx.get_node_attributes(self.G, 'pos')
            node_colors = get_node_colors(self.G, self.cmap)

            nx.draw_networkx_nodes(
                self.G, pos,
                node_color=node_colors,
                node_size=18,
                edgecolors='black',
                linewidths=0.6,
                alpha=0.9,
                ax=ax
            )

            if num_points > 1:
                nx.draw_networkx_edges(
                    self.G, pos,
                    edge_color='#005f99',
                    width=0.4,
                    alpha=0.8,
                    ax=ax
                )

            ax.set_title(f"N = {num_points}", fontsize=12, color='black', pad=6)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.axis('off')

            mplcyberpunk.make_lines_glow(ax=ax)

        plt.subplots_adjust(wspace=0.3)
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"图像已保存至: {save_path}")
        plt.show()

# 执行：一行横排版本
if __name__ == "__main__":
    fractal_flow = FractalNetworkFlow(dimension=1.6, connect_radius=0.4, cmap='coolwarm')
    num_points_list = [1, 5, 21, 85]
    fractal_flow.visualize_flow(num_points_list, save_path="fractalzoom.pdf")
