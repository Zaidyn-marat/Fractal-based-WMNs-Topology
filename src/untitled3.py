import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from matplotlib import gridspec

# 设置统一论文风格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

# ========================
# 子图1：Box-counting维度拟合图
# ========================
def generate_fractal_points(dimension, num_points):
    points = []
    def generate_recursive(x, y, scale, depth):
        if depth == 0:
            return
        points.append((x, y))
        new_scale = scale / 4**(1 / dimension)
        generate_recursive(x - new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x - new_scale, y + new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y + new_scale, new_scale, depth - 1)
    initial_scale = 0.5
    max_depth = int(np.log((3 * num_points + 1) - 1) / np.log(4))
    generate_recursive(0.5, 0.5, initial_scale, max_depth)
    points = np.array(points[:num_points])
    x = (points[:, 0] - points[:, 0].min()) / points[:, 0].ptp()
    y = (points[:, 1] - points[:, 1].min()) / points[:, 1].ptp()
    return x, y

def box_counting_dimension(positions):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(positions)
    S = np.ptp(standardized, axis=0).max()
    epsilons = S / np.logspace(0.3, 2.2, num=40)
    N_eps = []
    for eps in epsilons:
        mins = np.min(standardized, axis=0)
        indices = np.floor((standardized - mins) / eps).astype(int)
        unique_boxes = set(map(tuple, indices))
        N_eps.append(len(unique_boxes))
    log_L = np.log(epsilons)
    log_N = np.log(N_eps)
    fit_mask = (epsilons > 0.01) & (epsilons < 0.2)
    slope, _, r_value, _, _ = linregress(-log_L[fit_mask], log_N[fit_mask])
    return slope

# 维度采样
DD = np.arange(1, 2.1, 0.1)
FD = []
for D in DD:
    x, y = generate_fractal_points(D, 100000)
    positions = np.column_stack((x, y))
    fd = box_counting_dimension(positions)
    FD.append(fd)

# ========================
# 子图2：性能指标相关性热力图
# ========================
D_values = np.arange(1, 10.5, 0.5)
Throughput = [30.80, 28.55, 28.32, 29.43, 31.65, 32.40, 32.82, 32.62, 34.70, 34.68,
              34.89, 35.95, 33.61, 35.94, 33.35, 34.52, 35.23, 35.55, 35.17]
PDR = [0.3617, 0.3640, 0.3601, 0.3816, 0.3975, 0.4094, 0.4089, 0.4071, 0.4299, 0.4264,
        0.4351, 0.4428, 0.4215, 0.4329, 0.4214, 0.4325, 0.4313, 0.4366, 0.4317]
Delay = [2.2459, 1.3111, 1.3022, 1.1768, 1.1285, 1.1068, 1.1277, 1.1084, 1.0234, 1.0623,
          1.0478, 1.0198, 1.0222, 0.9821, 1.0068, 1.0111, 1.0493, 1.0095, 1.0175]
Jitter = [0.00094, 0.00084, 0.00084, 0.00073, 0.00071, 0.00059, 0.00063, 0.00069, 0.00044, 0.00059,
          0.00055, 0.00046, 0.00049, 0.00042, 0.00051, 0.00046, 0.00059, 0.00046, 0.00057]

df = pd.DataFrame({
    "Throughput": Throughput,
    "PDR": PDR,
    "Delay": Delay,
    "Jitter": [j * 1000 for j in Jitter]  # 转为毫秒
})
corr_matrix = df.corr(method="pearson")
# ======================== 第一张图：Box-counting维度拟合 ========================
fig1, ax1 = plt.subplots(figsize=(6.5, 5))

ax1.plot(DD, DD, '--', color='gray', zorder=1)
ax1.plot(DD, FD, 'o-', color='#4455bf', markeredgecolor='black', linewidth=2, zorder=2)

ax1.minorticks_on()
ax1.tick_params(axis='both', which='both', direction='in',
               top=True, right=True, labelsize=9)
ax1.tick_params(which='minor', length=3, color='gray', width=0.5)
ax1.tick_params(which='major', length=3.5, color='black', width=0.8)

ax1.set_xlabel("D$_{target}$", fontsize=14)
ax1.set_ylabel("D$_{BC}$", fontsize=14)
ax1.set_xlim(min(DD)-0.05, max(DD)+0.05)
ax1.set_ylim(min(FD)-0.1, max(FD)+0.1)
# ✅ 移除 set_aspect，避免变形
ax1.grid(False)

plt.tight_layout()
# plt.savefig("boxcounting_plot.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()


# ======================== 第二张图：性能指标热力图 ========================
fig2, ax2 = plt.subplots(figsize=(6.5, 5))

sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
    linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax2,
    annot_kws={"fontsize": 14}
)

ax2.tick_params(axis='both', direction='in', top=True, right=True, length=4, width=0.8)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, ha="center", fontsize=13)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, va="center", fontsize=13)

plt.tight_layout()
# plt.savefig("correlation_heatmap.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
