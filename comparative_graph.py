import matplotlib.pyplot as plt
import numpy as np

# Define ranks
ranks = np.arange(1, 201)

# Define prize distributions
winner_takes_all = [5000 if i == 0 else 0 for i in range(len(ranks))]
fifty_percent_rule = [5000 / (2**i) for i in range(len(ranks))]
long_tail = [5000 / (i + 10) for i in range(len(ranks))]
flat_distribution = [100 for _ in range(len(ranks))]

# Plot each distribution
plt.figure(figsize=(12, 8))

plt.plot(ranks, winner_takes_all, label="Winner-Takes-All", color="red")
plt.plot(ranks, fifty_percent_rule, label="50% Rule", color="blue")
plt.plot(ranks, long_tail, label="Long Tail Distribution", color="green")
plt.plot(ranks, flat_distribution, label="Flat Distribution", color="orange")

# Add labels and legend
plt.title("Comparative Prize Distribution Models", fontsize=16)
plt.xlabel("Rank", fontsize=14)
plt.ylabel("Prize", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Show plot
plt.show()