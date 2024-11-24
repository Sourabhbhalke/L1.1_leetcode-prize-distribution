import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Dataset: ranks and corresponding prizes
ranks = [1, 2, 3] + list(range(4, 51)) + list(range(51, 101)) + list(range(101, 201))
prizes = [5000, 2500, 1000] + [300] * (50 - 4 + 1) + [100] * (100 - 51 + 1) + [50] * (200 - 101 + 1)

# Define curve models
def linear_model(rank, m, c):
    return m * rank + c

def quadratic_model(rank, a, b, c):
    return a * rank**2 + b * rank + c

def exponential_decay(rank, P1, k):
    rank_clipped = np.clip(rank - 1, 0, 100)  # Prevent overflow in exp()
    return P1 * np.exp(-k * rank_clipped)

def gaussian_model(rank, A, mu, sigma):
    return A * np.exp(-((rank - mu)**2) / (2 * sigma**2))

# Function to evaluate and plot fits
def evaluate_fit(model_name, model_func, x_data, y_data, p0=None, bounds=(-np.inf, np.inf), maxfev=2000):
    try:
        # Fit the model
        params, _ = curve_fit(model_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=maxfev)
        fitted_y = model_func(x_data, *params)
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((y_data - fitted_y)**2)
        
        # Calculate R-squared
        ss_total = np.sum((y_data - np.mean(y_data))**2)
        ss_residual = np.sum((y_data - fitted_y)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Plot the result
        plt.figure(figsize=(8, 5))
        plt.scatter(x_data, y_data, color='blue', label='Actual Data')
        plt.plot(x_data, fitted_y, color='red', label=f'{model_name} Fit')
        plt.xlabel('Rank')
        plt.ylabel('Prize (LeetCoins)')
        plt.title(f'{model_name} Fit')
        plt.legend()
        plt.show()
        
        # Output result quality
        print(f"{model_name}:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R-squared: {r_squared:.4f}")
        
        if r_squared > 0.85:  # Threshold for a "good fit"
            print("  Result: Good fit\n")
        else:
            print("  Result: Not great fit\n")
    except RuntimeError as e:
        print(f"{model_name}: Fit failed ({e})\n")

# Prepare data for fitting
x_data = np.array(ranks)
y_data = np.array(prizes)

# Evaluate each model
evaluate_fit("Linear", linear_model, x_data, y_data)

evaluate_fit("Quadratic", quadratic_model, x_data, y_data)

evaluate_fit("Exponential Decay", exponential_decay, x_data, y_data,
             p0=[5000, 0.01], bounds=([0, 0], [np.inf, np.inf]))

evaluate_fit("Gaussian", gaussian_model, x_data, y_data,
             p0=[5000, np.mean(x_data), np.std(x_data)],
             bounds=([0, min(x_data), 0], [np.inf, max(x_data), np.inf]))