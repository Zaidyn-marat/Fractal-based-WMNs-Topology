import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fractal_recursion import generate_fractal_points

def box_dimension(x, y, plot=False):


    sizes = np.logspace(np.log10(1), np.log10(0.01), num=40)
    counts = []
    

    for s in sizes:

        xi = np.floor(x / s).astype(int)
        yi = np.floor(y / s).astype(int)
        
        boxes = set(zip(xi, yi))
        counts.append(len(boxes))
    

    sizes = np.array(sizes)
    counts = np.array(counts)
    valid = counts > 0
    
    log_s = np.log(sizes[valid])
    log_n = np.log(counts[valid])
    

    model = LinearRegression().fit(log_s.reshape(-1, 1), log_n)
    dimension = -model.coef_[0]
    
    
    return dimension




