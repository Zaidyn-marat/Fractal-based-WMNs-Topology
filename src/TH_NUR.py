# -*- coding: utf-8 -*-
"""
Created on Sat May 31 15:38:09 2025

@author: marat
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import mplcyberpunk
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from fractal_recursion import generate_fractal_points



from scipy.optimize import linprog




# 2. Сгенерируем traffic matrix (дальние пары - longest matching)
def longest_matching_tm(G):
    pairs = list(combinations(G.nodes, 2))
    pairs.sort(key=lambda x: nx.shortest_path_length(G, *x), reverse=True)
    matched = set()
    TM = []
    for u, v in pairs:
        if u not in matched and v not in matched:
            TM.append((u, v))
            matched.add(u)
            matched.add(v)
    return TM

# 3. Простейший расчёт throughput (наивный: считаем загруженность рёбер)
def estimate_throughput(G, TM):
    edge_load = {e: 0 for e in G.edges}

    for s, t in TM:
        path = nx.shortest_path(G, s, t)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            e = tuple(sorted((u, v)))
            edge_load[e] += 1

    # Найдём самый перегруженный линк
    max_load = max(edge_load.values())
    if max_load == 0:
        return 0
    # Мы предполагаем, что каждый поток в TM имеет вес 1
    # Тогда throughput = 1 / макс. загрузка линка
    return 1 / max_load


# Визуализация
#pos = nx.spring_layout(G, seed=42)
#nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')


# 创建网络（带权重）
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


num_points = 100
max_distance = 0.2
D=np.arange(2, 8, 0.1 )
TH=[]
for i in D:
    x, y = generate_fractal_points(i, num_points)
    coordinates = list(zip(x, y))
    G = create_mesh_network_with_coordinates(coordinates, max_distance)
    TM = longest_matching_tm(G)
    throughput = estimate_throughput(G, TM)
    TH.append(throughput)
plt.plot(D, TH)
# nx.draw(G, pos=coordinates)