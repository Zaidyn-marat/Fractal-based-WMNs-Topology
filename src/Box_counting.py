import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fractal_recursion import generate_fractal_points

# def information_dimension(x, y, plot=False):


#     # Logarithmically spaced box sizes (epsilon)
#     sizes = np.logspace(np.log10(1), np.log10(0.5), num=100)
#     I_values = []  # Information function values

#     for epsilon in sizes:
#         # Discretize points into grid cells
#         xi = np.floor(x / epsilon).astype(int)
#         yi = np.floor(y / epsilon).astype(int)
#         boxes, counts = np.unique(list(zip(xi, yi)), return_counts=True)

#         # Compute probabilities P_i(epsilon)
#         P = counts / len(x)
        
#         # Compute information function I(epsilon)
#         I = -np.sum(P * np.log(P))
#         I_values.append(I)

#     # Fit a linear model to estimate d_inf
#     log_eps = np.log(sizes)
#     log_I = np.array(I_values)

#     # slope, _ = np.polyfit(log_eps, log_I, 1)
#     # dimension = -slope

#     model = LinearRegression().fit(log_eps.reshape(-1, 1), log_I)
#     dimension = -model.coef_[0]

#     if plot:
#         plt.figure(figsize=(8, 6))
#         plt.scatter(log_eps, log_I, c='blue', label='Data points')
#         plt.plot(log_eps, model.predict(log_eps.reshape(-1, 1)), 
#                   color='red', label=f'Fit (D_inf = {dimension:.2f})')
#         plt.xlabel('log(ε)')
#         plt.ylabel('I(ε)')
#         plt.legend()
#         plt.show()

#     return dimension

# dimension = 1.8
# num_points = 100
# x, y = generate_fractal_points(dimension, num_points)

# info_dim = information_dimension(x, y, plot=True)



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


theoretical_dims = np.round(np.arange(1.0, 6.01, 0.1), 2)
num_points = 5000 # more points for better estimation
estimated_dims = []

for d in theoretical_dims:
    x, y = generate_fractal_points(d, num_points)
    D_est = box_dimension(x, y, plot=False)
    estimated_dims.append(D_est)


plt.figure(figsize=(8, 6))
plt.plot(theoretical_dims, theoretical_dims, 'k--')
plt.scatter(theoretical_dims, estimated_dims,
            s=100, c=estimated_dims, cmap='viridis', edgecolor='k', alpha=0.8)
plt.xlabel('Fractal Dimension (D)')
plt.ylabel(' box Dimension (D_inf)')
plt.tight_layout()
plt.show()

