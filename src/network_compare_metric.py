import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})
# 标签和数据
labels = ["BA", "WS", "ER", "D=2", "D=6.5", "D=7.5"]
metrics = {
    "Throughput (Mbps)": [19.37, 23.83, 23.81, 28.32, 35.95, 35.94],
    "PDR (%)": [0.267, 0.308, 0.305, 0.3601, 0.4428, 0.4329],
    "Delay (ms)": [2.01, 1.75, 1.56, 1.3022, 1.0198, 0.9821],
    "Jitter (ms)": [1.55, 1.67, 1.19, 0.84, 0.46, 0.42]
}

# 颜色选取
cmap = get_cmap("coolwarm")
selected_indices = np.linspace(0.25, 0.75, 6)
colors = [cmap(i) for i in selected_indices]

x = np.arange(len(labels))
width = 0.5

fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), dpi=300)

for ax, (title, data) in zip(axes.flatten(), metrics.items()):
    bars = ax.bar(x, data, color=colors, width=width, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel(title, fontsize=9)  # 设置y轴标签为指标名称
    ax.set_facecolor('white')
    
    # 设置刻度字体大小
    ax.tick_params(axis='both', labelsize=8)
    
    # 设置所有边框为黑色
    for spine in ['bottom', 'left', 'top', 'right']:
        ax.spines[spine].set_color('black')

plt.tight_layout()
plt.savefig("network_comparison_metrics.pdf", dpi=600, bbox_inches='tight', transparent=False)
plt.show()
