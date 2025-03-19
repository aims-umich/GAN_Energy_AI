import os
import pandas as pd
import numpy as np

# Root directory to start the search
root_dir = "/home/unabila/GAN_Energy_AI/model/Reg/CHF_FNN/"

# Walk through all directories and files
for dirpath, _, filenames in os.walk(root_dir):
    for file_name in filenames:
        if file_name.endswith('.csv'):
            file_path = os.path.join(dirpath, file_name)

            try:
                # Load CSV
                df = pd.read_csv(file_path)

                # Add 1% Gaussian noise
                noisy_df = df + 0.01 * df * np.random.randn(*df.shape)

                # Overwrite the original file
                noisy_df.to_csv(file_path, index=False)

                print(f"Noised and replaced: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

print("All CSV files updated with noise.")
