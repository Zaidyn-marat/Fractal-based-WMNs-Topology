import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os 
import random
def generate_fractal_points(dimension, num_points):

    points = []


    def generate_recursive(x, y, scale, depth):
        if depth == 0:
            return

        points.append((x, y))

        new_scale = scale / 4** (1 / dimension)
        generate_recursive(x - new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y - new_scale, new_scale, depth - 1)
        generate_recursive(x - new_scale, y + new_scale, new_scale, depth - 1)
        generate_recursive(x + new_scale, y + new_scale, new_scale, depth - 1)
        
        # points.append((x, y))
        
    initial_scale = 0.5
    # max_depth =int(np.log2(num_points))
    max_depth = int(np.log((3 * num_points + 1) - 1) / np.log(4))
    # max_depth = int(np.log(3 * num_points + 1) / np.log(4)) - 1
    generate_recursive(0.5, 0.5, initial_scale, max_depth)


    points = np.array(points[:num_points])
    
    x = (points[:, 0] - min(points[:, 0])) / (max(points[:, 0]) - min(points[:, 0]))
    y = (points[:, 1] - min(points[:, 1])) / (max(points[:, 1]) - min(points[:, 1]))
    return x,y

# dimensions = np.arange(1, 11, 0.1)
# num_points = 86
# # saved_files = []

# for d in dimensions:
#     x, y = generate_fractal_points(d, num_points)
#     coords = np.column_stack((x, y))
#     filename = f"fractal_coordinates_dim_{round(d, 1)}.txt"
#     np.savetxt(filename, coords, fmt="%.6f", delimiter="\t")
#     # saved_files.append(filename)

# # saved_files


import networkx as nx
from scipy.special import erfc
def capacity_shannon(d):  # f = 2.4 ГГц, Pt = 0 dBW, N = -90 dBm ≈ 10^-9 W
    
    # Constants
    Pt_dBm = 38  # Transmit power in dBm (100 mW)
    Pt = 10 ** (Pt_dBm / 10) / 1000  # Power in Watts
    Gt = -1.7  # Transmit antenna gain (linear)
    Gr = -1.7  # Receive antenna gain (linear)
    f = 2.4e9  # Frequency in Hz (2.4 GHz)
    c = 3e8  # Speed of light in m/s
    k = 1.38e-23  # Boltzmann's constant
    T = 290  # Temperature in Kelvin
    B = 2.7e6  # Bandwidth in Hz (1 MHz)
    # Noise power
    N = k * T * B
    
    #Потеря энергиии по пути
    L = (4 * np.pi * d * f / c) ** 2
    
    #Мощность принятого сигнала
    Pr=Pt * Gt * Gr / L
    
   
    
    SNR = Pr / N
    
    # BER=(7/12) * erfc(np.sqrt((1/7) * SNR))
    C=B * np.log2(1 + SNR)
    return C


# # Диапазон значений от 1.0 до 10.0 с шагом 0.1
# dims = np.arange(1.0, 10.1, 0.1)

# for dim in dims:
#     # Формируем имя файла с точностью до 1 знака после запятой
#     filename = f"fractal_coordinates_dim_{dim:.1f}.txt"

#     try:
#         coords = np.loadtxt(filename)
#         print(f"Загружено из файла {filename}:")
#         print(coords)
#     except OSError:
#         print(f"Файл {filename} не найден.")

# Задаём радиус для соединения точек
epsilon = 200  # можно настроить под конкретную плотность сети
num_points = 86
# Обработка всех файлов с размерностью от 1.0 до 2.0 с шагом 0.1
dimensions = np.arange(1,10,0.1)  # округлено до включительно 10.0
AvgC=[]
for d in dimensions:

    
    X,Y = generate_fractal_points(d,num_points)
    x,y = X*500,Y*500
    coords = list(zip(x, y))
    # print(coords)



    # Создание пустого графа
    G = nx.Graph()

    # Добавляем узлы
    for i, (x, y) in enumerate(coords):
        G.add_node(i, pos=(x, y))
    
    # Добавляем рёбра между точками, расстояние между которыми меньше epsilon
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            # dist = np.linalg.norm(coords[i] - coords[j])
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            if dist < epsilon:
                G.add_edge(i, j, length=dist)
    
    
    pos = nx.get_node_attributes(G, 'pos')



# plt.figure(1)
# nx.draw_networkx(G, pos=pos, with_labels=True)
    
# edge_labels = nx.get_edge_attributes(G, "length")
# nx.draw_networkx_edge_labels(G, pos, edge_labels)
# # nx.draw_networkx_node_labels(G, pos+0.1, Px.values())
# plt.show()




    
    Nodes=G.nodes()
    # print(Nodes)
    from itertools import combinations 
    a=list(combinations(Nodes,2))
    # print(len(a))
    
    TotalC=[]
    
    for i in a:
        source=i[0]
        target=i[1]
            # Вычислим пропускную способность между двумя узлами (например, от 'A' до 'D')
        path = nx.shortest_path(G, source, target, weight='distance')
    
        path_edges = list(zip(path[:-1], path[1:]))
        # print(path_edges)
        
    
    
    
    
    
        # Вычислим пропускную способность для каждого линка
        link_capacities = []
        for u, v in path_edges:
            d = G[u][v]['length']
            # print(d)
            
            
            C = capacity_shannon(d)/ 1e6
            
            link_capacities.append(C)
        # print(link_capacities)
        # Минимальная пропускная способность по пути
        min_capacity = min(link_capacities)/len(link_capacities)
        TotalC.append(min_capacity)
    print(np.mean(TotalC))
    AvgC.append(np.mean(TotalC))
    
# Plot

# th = []
plt.figure(figsize=(10, 6))
plt.plot(dimensions, AvgC)
plt.xlabel("Dimension")
plt.ylabel("Пропускная способность (Мбит/с)")
plt.title("Пропускная способность от расстояния (64-QAM, 1 МГц)")
plt.grid(True)
plt.tight_layout()
plt.show()



