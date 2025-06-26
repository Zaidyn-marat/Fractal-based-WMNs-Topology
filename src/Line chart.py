import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [ "Times New Roman"],
})


# 数据
D_values = np.arange(1, 10.5, 0.5)
Throughput = [30.80, 28.55, 28.32, 29.43, 31.65, 32.40, 32.82, 32.62, 34.70, 34.68,
              34.89, 35.95, 33.61, 35.94, 33.35, 34.52, 35.23, 35.55, 35.17]
PDR = [0.3617, 0.3640, 0.3601, 0.3816, 0.3975, 0.4094, 0.4089, 0.4071, 0.4299, 0.4264,
        0.4351, 0.4428, 0.4215, 0.4329, 0.4214, 0.4325, 0.4313, 0.4366, 0.4317]
Delay = [2.2459, 1.3111, 1.3022, 1.1768, 1.1285, 1.1068, 1.1277, 1.1084, 1.0234, 1.0623,
          1.0478, 1.0198, 1.0222, 0.9821, 1.0068, 1.0111, 1.0493, 1.0095, 1.0175]
Jitter = [0.00094, 0.00084, 0.00084, 0.00073, 0.00071, 0.00059, 0.00063, 0.00069, 0.00044, 0.00059,
          0.00055, 0.00046, 0.00049, 0.00042, 0.00051, 0.00046, 0.00059, 0.00046, 0.00057]

# # # 色图和颜色映射
cmap = sns.color_palette("coolwarm", as_cmap=True)
norm = plt.Normalize(vmin=min(D_values), vmax=max(D_values))
colors = [cmap(norm(d)) for d in D_values]



import itertools
from matplotlib import patheffects
from adjustText import adjust_text

metrics = {
    "Throughput (Mbps)": Throughput,
    "PDR (%)": PDR,
    "Delay (ms)": Delay,
    "Jitter (ms)": [j * 1000 for j in Jitter]  # 单位从秒转毫秒
}

excluded_pairs = [("Throughput(Mbps)", "Delay(ms)"), ("PDR(%)", "Delay(ms)")]
pairs = [pair for pair in itertools.combinations(metrics.keys(), 2) if pair not in excluded_pairs]

# 修改这里为 2 行 2 列
fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white', constrained_layout=True)
axes = axes.flatten()

for ax, (x_name, y_name) in zip(axes, pairs):
    x = metrics[x_name]
    y = metrics[y_name]

    texts = []
    
    for i, (xi, yi, d) in enumerate(zip(x, y, D_values)):
        ax.scatter(xi, yi, color=colors[i], edgecolors='black', linewidth=0.6,
                  s=220, alpha=0.85, marker='o' if d >= 5 else '^')

    # 设置刻度参数
    ax.minorticks_on()  # 启用次要刻度
    ax.tick_params(axis='both', which='both', direction='in', 
                   top=True, right=True,  # 在顶部和右侧显示刻度
                   labelsize=9)
    
    # 更精细的刻度控制
    ax.tick_params(which='minor', length=3, color='gray', width=0.5,labelsize=16)  # 次要刻度
    ax.tick_params(which='major', length=3, color='black', width=0.8,labelsize=16)  # 主要刻度
    
    ax.set_xlabel(x_name, fontsize=16)
    ax.set_ylabel(y_name, fontsize=16)
    ax.grid(False)

# 隐藏多余子图
for i in range(len(pairs), len(axes)):
    axes[i].axis('off')

# 颜色条设置
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', shrink=0.8, pad=0.01)
cbar.set_label("Fractal Dimension (D)", fontsize=16)
cbar.ax.tick_params(labelsize=10)

# 保存和显示
plt.savefig("SCAT.pdf", dpi=600, bbox_inches='tight')
plt.show()


#-------------------------------------------------------------------------

# import pandas as pd

# # 构建 DataFrame
# df = pd.DataFrame({
#     "Throughput": Throughput,
#     "PDR": PDR,
#     "Delay": Delay,
#     "Jitter": Jitter
# })

# # # 计算相关系数矩阵
# corr_matrix = df.corr()

# # 绘制热力图
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
#             linewidths=0.5, cbar_kws={"shrink": 0.8})
# # plt.title("Correlation Matrix of Network Performance Metrics", fontsize=14)
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)
# plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
# plt.savefig("colleralation.pdf", dpi=600)

#-------------------------------------------------------------

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE
# from adjustText import adjust_text

# # 数据标准化
# features = ["Throughput", "PDR", "Delay", "Jitter"]
# X_std = StandardScaler().fit_transform(df[features])

# # t-SNE 降维
# tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=1000, random_state=42)
# X_tsne = tsne.fit_transform(X_std)

# # 绘图
# fig, ax = plt.subplots(figsize=(7, 5))

# # 高低维形状映射
# for (x, y, d) in zip(X_tsne[:, 0], X_tsne[:, 1], D_values):
#     marker = 'o' if d >= 5 else '^'
#     sc = ax.scatter(x, y, c=[d], cmap="coolwarm", s=80,
#                     edgecolors='k', marker=marker, vmin=min(D_values), vmax=max(D_values))

#     # 在右上偏移文本，避免遮挡标记
#     ax.text(x + 5, y - 0.5, f"{d:.1f}", fontsize=10,
#             ha='left', va='center', color='black')

# # 添加 x=0 的分界线
# ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# # 添加 colorbar
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('Fractal Dimension (D)', fontsize=12)
# cbar.ax.tick_params(labelsize=10)

# # 图像美化
# ax.set_xlabel("t-SNE 1", fontsize=11)
# ax.set_ylabel("t-SNE 2", fontsize=11)
# ax.tick_params(labelsize=9)

# plt.tight_layout()
# # plt.savefig("Projection.pdf", dpi=600)
# plt.show()



#------------------------------


