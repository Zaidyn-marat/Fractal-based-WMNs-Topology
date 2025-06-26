import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置论文级绘图风格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Linux Libertine", "Libertinus Serif", "Times"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False
})

# 原始性能数据
D_values = np.arange(1, 10.5, 0.5)
Throughput = [30.80, 28.55, 28.32, 29.43, 31.65, 32.40, 32.82, 32.62, 34.70, 34.68,
              34.89, 35.95, 33.61, 35.94, 33.35, 34.52, 35.23, 35.55, 35.17]
PDR = [0.3617, 0.3640, 0.3601, 0.3816, 0.3975, 0.4094, 0.4089, 0.4071, 0.4299, 0.4264,
        0.4351, 0.4428, 0.4215, 0.4329, 0.4214, 0.4325, 0.4313, 0.4366, 0.4317]
Delay = [2.2459, 1.3111, 1.3022, 1.1768, 1.1285, 1.1068, 1.1277, 1.1084, 1.0234, 1.0623,
          1.0478, 1.0198, 1.0222, 0.9821, 1.0068, 1.0111, 1.0493, 1.0095, 1.0175]
Jitter = [0.00094, 0.00084, 0.00084, 0.00073, 0.00071, 0.00059, 0.00063, 0.00069, 0.00044, 0.00059,
          0.00055, 0.00046, 0.00049, 0.00042, 0.00051, 0.00046, 0.00059, 0.00046, 0.00057]

# 构建 DataFrame
df = pd.DataFrame({
    "Throughput": Throughput,
    "PDR": PDR,
    "Delay": Delay,
    "Jitter": [j * 1000 for j in Jitter]  # 转为毫秒
})

# 计算相关系数矩阵
corr_matrix = df.corr(method="pearson")

# 创建图像和子图对象
fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=False)

# 绘制热力图
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f",
    cmap="coolwarm", square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8}, ax=ax,
    annot_kws={"fontsize": 12}
)

# 坐标轴细节
ax.tick_params(axis='both', direction='in', top=True, right=True, length=4, width=0.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=12)

# 调整子图边距（避免 colorbar 与边界冲突）
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

# 保存为 PDF 图像
plt.savefig("correlation.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
