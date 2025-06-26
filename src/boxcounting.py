import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

# 统一风格设置
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

# 采样范围
DD = np.arange(1, 2.1, 0.1)
FD = []
for D in DD:
    x, y = generate_fractal_points(D, 100000)
    positions = np.column_stack((x, y))
    fd = box_counting_dimension(positions)
    FD.append(fd)

# 画图
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(DD, DD, '--', color='gray', zorder=1)
ax.plot(DD, FD, 'o-', color='#4455bf', markeredgecolor='black', linewidth=2, zorder=2)
ax.set_xlabel("Fractal Dimension", fontsize=12)
ax.set_ylabel("Box-counting Dimension", fontsize=12)
ax.set_xlim(min(DD)-0.05, max(DD)+0.05)
ax.set_ylim(min(FD)-0.1, max(FD)+0.1)
ax.set_aspect('equal', adjustable='box')  # 保持方形比例
ax.grid(False)
plt.tight_layout()  # 建议去掉或用 constrained_layout
# plt.savefig("boxcounting.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
