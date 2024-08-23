import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data from CSV
df = pd.read_csv('wind_data.csv')
wind_speeds = df['WindSpeed'].dropna()

# Define functions for empirical and theoretical CDF
def empirical_cdf(data):
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y

def theoretical_cdf(x, shape, scale):
    return 1 - np.exp(- (x / scale) ** shape)

# Define the objective function for optimization
def objective(params, x, y):
    shape, scale = params
    y_pred = theoretical_cdf(x, shape, scale)
    return np.sum((y - y_pred) ** 2)

# Calculate empirical CDF
x_empirical, y_empirical = empirical_cdf(wind_speeds)

# Initial guess for shape and scale parameters
initial_guess = [1.0, np.mean(wind_speeds)]

# Perform optimization to fit Weibull distribution
result = minimize(objective, initial_guess, args=(x_empirical, y_empirical), bounds=[(0.1, None), (0.1, None)])
shape_estimated, scale_estimated = result.x

# Display the estimated parameters
print(f'Estimated shape parameter (k): {shape_estimated:.4f}')
print(f'Estimated scale parameter (λ): {scale_estimated:.4f}')

# Plotting the empirical and fitted CDFs
plt.figure(figsize=(10, 6))
plt.step(x_empirical, y_empirical, label='Empirical CDF', where='post')
plt.plot(x_empirical, theoretical_cdf(x_empirical, shape_estimated, scale_estimated), 'r-', label=f'Fitted Weibull CDF\nShape={shape_estimated:.2f}, Scale={scale_estimated:.2f}')
plt.xlabel('Wind Speed')
plt.ylabel('CDF')
plt.title('Empirical vs Fitted Weibull CDF')
plt.legend()
plt.grid(True)
plt.show()
