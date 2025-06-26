import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from itertools import combinations

from gudhi import RipsComplex, SimplexTree
from gudhi.representations import PersistenceImage, Landscape
from sklearn.decomposition import PCA

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
    x = (points[:, 0] - min(points[:, 0])) / (max(points[:, 0]) - min(points[:, 0]))
    y = (points[:, 1] - min(points[:, 1])) / (max(points[:, 1]) - min(points[:, 1]))
    return x, y

def create_mesh_network_with_coordinates(coordinates, max_distance):
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)
    for i, j in combinations(range(len(coordinates)), 2):
        distance = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        if distance <= max_distance:
            capacity = 1 / (1 + distance)
            G.add_edge(i, j, capacity=capacity, weight=1 / capacity)
    return G

def compute_topological_features(coordinates, max_dimension=9):
    """
    计算点云的拓扑特征
    :param coordinates: 点坐标列表 [(x1,y1), (x2,y2), ...]
    :param max_dimension: 计算的最大同调维度
    :return: 包含拓扑特征的字典
    """
    # 转换为numpy数组
    points = np.array(coordinates)
    
    # 创建Rips复形
    rips_complex = RipsComplex(points=points, max_edge_length=0.5)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    # 计算持续同调
    persistence = simplex_tree.persistence()
    
    # 提取各维度的条形码
    barcode = {dim: [] for dim in range(max_dimension+1)}
    for interval in persistence:
        dim = interval[0]
        if dim <= max_dimension:
            birth = interval[1][0]
            death = interval[1][1] if not np.isinf(interval[1][1]) else birth + 1.0  # 处理无限持续
            barcode[dim].append((birth, death))
    
    # 计算拓扑特征
    features = {}
    
    # Betti数（各维度拓扑特征的个数）
    features['betti_0'] = len([b for b in barcode[0] if b[1] - b[0] > 0.01]) 
    features['betti_1'] = len(barcode[1])
    features['betti_2'] = len(barcode[2])
    
    # 持续熵（衡量拓扑结构的复杂性）
    for dim in range(max_dimension+1):
        lifetimes = [death - birth for birth, death in barcode[dim]]
        total_lifetime = sum(lifetimes)
        if total_lifetime > 0:
            probs = [lt / total_lifetime for lt in lifetimes]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            features[f'persistence_entropy_{dim}'] = entropy
        else:
            features[f'persistence_entropy_{dim}'] = 0.0
    
    # 平均持续时间和最大持续时间
    for dim in range(max_dimension+1):
        if barcode[dim]:
            lifetimes = [death - birth for birth, death in barcode[dim]]
            features[f'mean_persistence_{dim}'] = np.mean(lifetimes)
            features[f'max_persistence_{dim}'] = np.max(lifetimes)
        else:
            features[f'mean_persistence_{dim}'] = 0.0
            features[f'max_persistence_{dim}'] = 0.0
    
    return features, barcode, persistence

# 主分析流程
def analyze_fractal_dimensions(dim_range, num_points=86, max_distance=0.2):
    results = []
    
    for D in dim_range:
        print(f"Analyzing dimension D = {D:.1f}")
        
        # 生成分形点集
        x, y = generate_fractal_points(D, num_points)
        coordinates = list(zip(x, y))
        
        # 创建网状网络
        G = create_mesh_network_with_coordinates(coordinates, max_distance)
        
        # 计算拓扑特征
        features, barcode, persistence = compute_topological_features(coordinates)
        
        # 存储结果
        result = {
            'dimension': D,
            'network': G,
            'coordinates': coordinates,
            'features': features,
            'barcode': barcode,
            'persistence': persistence
        }
        results.append(result)
    
    return results

# 可视化函数
def visualize_results(results):
    # 准备绘图数据
    dimensions = [res['dimension'] for res in results]
    
    # Betti数变化
    betti_0 = [res['features']['betti_0'] for res in results]
    betti_1 = [res['features']['betti_1'] for res in results]
    betti_2 = [res['features']['betti_2'] for res in results]
    
    # 持续熵变化
    entropy_0 = [res['features']['persistence_entropy_0'] for res in results]
    entropy_1 = [res['features']['persistence_entropy_1'] for res in results]
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # Betti数变化图
    plt.subplot(2, 2, 1)
    plt.plot(dimensions, betti_0, 'o-', label='Betti-0 (flux)')
    plt.plot(dimensions, betti_1, 's-', label='Betti-1 (circle)')
    plt.plot(dimensions, betti_2, 'd-', label='Betti-2 (empty)')
    plt.xlabel('D')
    plt.ylabel('Betti')
    # plt.title('拓扑特征数量随分形维度的变化')
    plt.legend()
    plt.grid(True)
    
    # 持续熵变化图
    plt.subplot(2, 2, 2)
    plt.plot(dimensions, entropy_0, 'o-', label='0E')
    plt.plot(dimensions, entropy_1, 's-', label='1E')
    plt.xlabel('D')
    plt.ylabel('entropy')
    # plt.title('拓扑复杂性随分形维度的变化')
    plt.legend()
    plt.grid(True)
    
    # 网络可视化示例（取第一个和最后一个维度）
    plt.subplot(2, 2, 3)
    pos = {i: results[0]['coordinates'][i] for i in range(len(results[0]['coordinates']))}
    nx.draw(results[0]['network'], pos, node_size=20, edge_color='gray')
    plt.title(f'D = {dimensions[0]:.1f} ')
    
    plt.subplot(2, 2, 4)
    pos = {i: results[-1]['coordinates'][i] for i in range(len(results[-1]['coordinates']))}
    nx.draw(results[-1]['network'], pos, node_size=20, edge_color='gray')
    plt.title(f'D = {dimensions[-1]:.1f} ')
    
    plt.tight_layout()
    # plt.savefig('fractal_network_analysis.png', dpi=300)
    plt.show()
    
    # 绘制条形码图示例
    plt.figure(figsize=(12, 6))
    for i, dim in enumerate([0, 1]):
        plt.subplot(1, 2, i+1)
        for j, (birth, death) in enumerate(results[-1]['barcode'][dim]):
            plt.plot([birth, death], [j, j], 'b-', linewidth=2)
        plt.title(f'D = {dimensions[-1]:.1f} 维{dim}条形码')
        plt.xlabel('filter parameter')
        plt.ylabel('topology features')
    plt.tight_layout()
    # plt.savefig('persistence_barcodes.png', dpi=300)
    plt.show()

# 执行分析
if __name__ == "__main__":
    # 设置分形维度范围 (1.0到3.0，步长0.2)
    dim_range = np.arange(1.0, 8, 1)
    
    # 运行分析
    results = analyze_fractal_dimensions(dim_range)
    
    # 可视化结果
    visualize_results(results)
    
    # 打印特征摘要
    print("\n拓扑特征摘要:")
    print("维度 | Betti0 | Betti1 | Betti2 | 熵0 | 熵1")
    for res in results:
        f = res['features']
        print(f"{res['dimension']:.1f} | {f['betti_0']} | {f['betti_1']} | {f['betti_2']} | "
              f"{f['persistence_entropy_0']:.3f} | {f['persistence_entropy_1']:.3f}")