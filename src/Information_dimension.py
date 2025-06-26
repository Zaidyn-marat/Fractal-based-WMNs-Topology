import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fractal_recursion import generate_fractal_points


def information_dimension(x, y, plot=False):
    sizes = np.logspace(np.log10(1), np.log10(0.1), num=1000)
    I_values = []

    for epsilon in sizes:
        xi = np.floor(x / epsilon).astype(int)
        yi = np.floor(y / epsilon).astype(int)
        boxes, counts = np.unique(list(zip(xi, yi)), return_counts=True)

        P = counts / len(x)
        I = -np.sum(P * np.log(P))
        I_values.append(I)

    log_eps = np.log(sizes)
    log_I = np.array(I_values)

    model = LinearRegression().fit(log_eps.reshape(-1, 1), log_I)
    dimension = -model.coef_[0]

    if plot:
        plt.figure(figsize=(6, 5))
        plt.scatter(log_eps, log_I, c='blue', label='Data points', s=10)
        plt.plot(log_eps, model.predict(log_eps.reshape(-1, 1)),
                  color='red', label=f'Fit (D_inf = {dimension:.2f})')
        plt.xlabel('log(ε)')
        plt.ylabel('I(ε)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title('Information Dimension Fit')
        plt.tight_layout()
        plt.show()

    return dimension




theoretical_dims = np.round(np.arange(1.0, 2.01, 0.1), 2)
num_points = 1000 # more points for better estimation
estimated_dims = []

for d in theoretical_dims:
    x, y = generate_fractal_points(d, num_points)
    D_est = information_dimension(x, y, plot=False)
    estimated_dims.append(D_est)


plt.figure(figsize=(8, 6))
plt.plot(theoretical_dims, theoretical_dims, 'k--')
plt.scatter(theoretical_dims, estimated_dims,
            s=100, c=estimated_dims, cmap='viridis', edgecolor='k', alpha=0.8)
plt.xlabel('Fractal Dimension (D)')
plt.ylabel(' Information Dimension (D_inf)')
plt.tight_layout()
plt.show()
