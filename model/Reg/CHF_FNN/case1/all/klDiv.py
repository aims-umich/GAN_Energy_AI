import numpy as np
import pandas as pd
from scipy.stats import entropy

# Load test (true) and GAN-predicted datasets
test_df = pd.read_csv("/home/unabila/FFchf/all/test_all.csv")  # Replace with actual file path
gan_df = pd.read_csv("/home/unabila/CondGan/1h/generated_samples_(0.001, 3, 200, 64).csv")  # Replace with actual file path

# Extract CHF column (last column)
true_chf = test_df.iloc[:, -1].dropna().values  # True CHF values
pred_chf = gan_df.iloc[:, -1].dropna().values  # GAN-predicted CHF values

# Define number of bins for probability estimation
num_bins = 50

# Compute histograms (probability distributions)
hist_true, bin_edges = np.histogram(true_chf, bins=num_bins, density=True)
hist_pred, _ = np.histogram(pred_chf, bins=bin_edges, density=True)

# Avoid zero probabilities by adding a small constant
hist_true += 1e-10
hist_pred += 1e-10

# Compute KL divergence (measuring difference between distributions)
kl_divergence = entropy(hist_pred, hist_true)  # D_KL(P_G || P_R)

# Display result
print(f"KL Divergence (CHF): {kl_divergence}")
