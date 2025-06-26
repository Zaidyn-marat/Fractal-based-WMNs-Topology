import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
import os 

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

# dimensions = np.arange(1, 10, 1)
# num_points = 86
# saved_files = []

# for d in dimensions:
#     x, y = generate_fractal_points(d, num_points)
#     coords = np.column_stack((x, y))
#     filename = f"fractal_coordinates_dim_{round(d, 1)}.txt"
#     np.savetxt(filename, coords, fmt="%.6f", delimiter="\t", header=f"Dimension: {d}", comments="")
#     saved_files.append(filename)











