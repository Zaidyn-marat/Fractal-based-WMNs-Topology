import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 输入数据
D = np.arange(1, 11)
Throughput = [34.59, 33.75, 40.12, 39.66, 41.50, 41.91, 41.66, 39.74, 41.01, 38.67]
Delay = [1.97, 1.10, 0.89, 0.96, 0.93, 0.94, 0.91, 0.91, 0.93, 1.04]
Jitter = [0.85, 0.49, 0.35, 0.33, 0.43, 0.34, 0.36, 0.35, 0.48, 0.47]
PDR = [0.39, 0.41, 0.48, 0.47, 0.48, 0.49, 0.49, 0.47, 0.48, 0.46]

# 综合指标计算
# Q-Score
alpha, beta, gamma = 0.5, 0.3, 0.2
Q_scores = alpha*np.array(PDR) + beta*(1/np.array(Delay)) + gamma*(1/np.array(Jitter))

# 网络效率
theoretical_capacity = 45
efficiency = (np.array(Throughput)/theoretical_capacity) * np.array(PDR)

# 时延-抖动耦合系数
cov_matrix = np.cov(Delay, Jitter)
C_DJ = cov_matrix[0,1]/(np.std(Delay)*np.std(Jitter))

# 性能稳定性
throughput_stability = 1 - (np.std(Throughput)/np.mean(Throughput))

# 可视化设置
plt.style.use('default')
sns.set_palette("husl")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig)

# 子图1: 多指标雷达图
ax1 = fig.add_subplot(gs[0, :], polar=True)
categories = ['Throughput','Delay','Jitter','PDR','Q-Score']
N = len(categories)
angles = [n/float(N)*2*np.pi for n in range(N)]
angles += angles[:1]

def normalize(data):
    return (data - np.min(data))/(np.max(data)-np.min(data))

values = np.array([
    normalize(Throughput),
    normalize([-d for d in Delay]),  # 时延取负向指标
    normalize([-j for j in Jitter]),
    normalize(PDR),
    normalize(Q_scores)
]).T

ax1.set_theta_offset(np.pi/2)
ax1.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
ax1.set_rlabel_position(0)

for d in range(10):
    ax1.plot(angles, np.append(values[d], values[d][0]), linewidth=1, linestyle='solid', 
             label=f'D={d+1}' if d<5 else None)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Multi-Dimensional Performance Radar Chart", y=1.2)

# 子图2: 时延-抖动相位图
ax2 = fig.add_subplot(gs[1, 0])
sc = ax2.scatter(Delay, Jitter, c=D, cmap='viridis', s=100)
plt.colorbar(sc, label='Dimension D')
ax2.set_xlabel("Delay (s)")
ax2.set_ylabel("Jitter (s)")
plt.title("Delay-Jitter Phase Diagram\nCoupling Coefficient: {:.3f}".format(C_DJ))

# 子图3: Q-Score与效率关系
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(Q_scores, efficiency, c=D, cmap='plasma', s=100)
ax3.set_xlabel("Q-Score")
ax3.set_ylabel("Network Efficiency")
plt.title("QoS vs Efficiency Trade-off")

# 子图4: 性能稳定性
ax4 = fig.add_subplot(gs[1, 2])
ax4.bar(D, Throughput, color='skyblue', label='Throughput')
ax4.plot(D, np.ones(10)*np.mean(Throughput), 'r--', 
        label='Stability: {:.2f}'.format(throughput_stability))
ax4.set_xticks(D)
ax4.legend()
plt.title("Throughput Stability Analysis")

# 子图5: 指标相关性热力图
ax5 = fig.add_subplot(gs[2, :])
corr_matrix = np.corrcoef([Throughput, Delay, Jitter, PDR, Q_scores])
sns.heatmap(corr_matrix, annot=True, xticklabels=['Throughput','Delay','Jitter','PDR','Q-Score'],
           yticklabels=['Throughput','Delay','Jitter','PDR','Q-Score'], cmap="coolwarm")
plt.title("Cross-Metric Correlation Matrix")

plt.tight_layout()
plt.show()