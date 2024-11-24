import numpy as np
import matplotlib.pyplot as plt

# Data: Rank and corresponding prizes
ranks = [1, 2, 3] + list(range(4, 51)) + list(range(51, 101)) + list(range(101, 201))
prizes = [5000, 2500, 1000] + [300] * (50 - 4 + 1) + [100] * (100 - 51 + 1) + [50] * (200 - 101 + 1)

# Fit an exponential decay function
def prize_distribution(rank, P1=5000, k=0.1):
    return P1 * np.exp(-k * (rank - 1))

# Generate fitted curve
fitted_prizes = [prize_distribution(r) for r in ranks]

# Plot original data and fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(ranks, prizes, color='blue', label='Actual Prizes')
plt.plot(ranks, fitted_prizes, color='red', label='Fitted Curve')
plt.xlabel('Rank')
plt.ylabel('Prize (LeetCoins)')
plt.title('LeetCode Prize Distribution Curve')
plt.legend()
plt.show()